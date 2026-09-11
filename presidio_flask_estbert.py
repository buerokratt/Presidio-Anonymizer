import yaml
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    PatternRecognizer,
    Pattern,
)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider
from presidio_analyzer.entity_recognizer import EntityRecognizer
from typing import List, Optional
import logging
import os
import re

logger = logging.getLogger("presidio-flask-api")

# Estonian abbreviations whose full stop does not end a sentence. Used when
# deciding where a model span may be cut, so "Pärnu mnt. 12" stays one span.
ABBREVIATIONS = frozenset(
    {
        "tn",  # tänav
        "mnt",  # maantee
        "pst",  # puiestee
        "kt",  # kaubatänav
        "nr",  # number
        "lk",  # lehekülg
        "vt",  # vaata
        "jne",  # ja nii edasi
        "jm",  # ja muud
        "sh",  # sealhulgas
        "nt",  # näiteks
        "dr",  # doktor
        "hr",  # härra
        "pr",  # proua
    }
)

# Per-entity score thresholds read from the config. AnalyzerEngine only supports
# one threshold for everything, so the engine is built with the lowest threshold
# in play and apply_score_thresholds() does the real per-entity filtering. The
# app builds exactly one analyzer per process, so these live at module level
# instead of being threaded through every call site.
entity_score_thresholds: dict[str, float] = {}
default_score_threshold: float = 0.8


class EstBERTRecognizerONNX(EntityRecognizer):
    """Custom recognizer for tartuNLP/EstBERT_NER model with ONNX optimization"""

    ENTITIES = ["PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "GPE"]

    # XLM-RoBERTa has 514 position embeddings. Feeding it more makes the ONNX
    # session throw, which previously wiped out every model detection for the
    # whole request, so longer text is analysed in overlapping windows and the
    # spans are mapped back onto the original offsets. The overlap gives an
    # entity that lands on a window edge a second chance in the next window.
    WINDOW_TOKENS = 400
    WINDOW_OVERLAP_TOKENS = 50

    # An Estonian registration plate, which the CAR_NUMBER recognizer owns.
    PLATE_SHAPE = re.compile(r"[0-9]{2,3}\s?[A-ZÕÄÖÜ]{3}")

    def __init__(
        self, model_name: str = "tartuNLP/EstBERT_NER", supported_language: str = "xx"
    ) -> None:
        super().__init__(
            supported_entities=self.ENTITIES,
            supported_language=supported_language,
            name="EstBERT_NER_ONNX_Recognizer",
        )

        logger.info(f"Loading EstBERT model with ONNX optimization: {model_name}")

        # Log cache configuration
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", "default")
        logger.info(f"Using model cache directory: {cache_dir}")

        try:
            # Load tokenizer - will use TRANSFORMERS_CACHE env var automatically
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)

            # Configure ONNX session options
            inter_threads = int(os.getenv("ONNX_INTER_OP_THREADS", "2"))
            intra_threads = int(os.getenv("ONNX_INTRA_OP_THREADS", "2"))

            logger.info(
                f"ONNX config: inter_op_threads={inter_threads}, intra_op_threads={intra_threads}"
            )

            # Load ONNX model with optimized settings
            # This will download to cache on first run, then reuse from volume
            self.model = ORTModelForTokenClassification.from_pretrained(
                model_name,
                export=True,  # Export to ONNX if not already
                provider="CPUExecutionProvider",  # Use CPU
            )

            # Configure session after loading (works with optimum 1.23+)
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "get_session_options"
            ):
                try:
                    session_options = self.model.model.get_session_options()  # type: ignore
                    session_options.inter_op_num_threads = inter_threads
                    session_options.intra_op_num_threads = intra_threads
                    logger.info(
                        f"✓ ONNX session configured with {inter_threads} inter-op and {intra_threads} intra-op threads"
                    )
                except Exception as e:
                    logger.warning(f"Could not configure session options: {e}")

            logger.info("Model loaded with ONNX optimization")

            # Create pipeline with ONNX model
            # "simple" does not merge sentencepiece continuations for XLM-R, so
            # a name came back as several fragments scored separately - the ones
            # below the threshold were dropped and the rest of the name survived
            # in clear text. "first" takes the first subword's label for the
            # whole word and keeps names intact.
            self.nlp_pipeline = pipeline(
                "ner",
                model=self.model,  # type: ignore
                tokenizer=self.tokenizer,
                aggregation_strategy="first",
                device=-1,  # CPU
            )

            self.label_mapping = {
                "PER": "PERSON",
                "ORG": "ORGANIZATION",
                "LOC": "LOCATION",
                "GPE": "GPE",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            }

            logger.info(
                f"EstBERT ONNX recognizer initialized for language: {supported_language}"
            )
            logger.info(f"  Model: {model_name}")
            logger.info("  ONNX optimization enabled")
        except Exception as e:
            logger.error(f"Failed to initialize EstBERT ONNX recognizer: {e}")
            raise

    def load(self) -> None:
        """Load method - required by Presidio"""
        pass

    def _char_windows(self, text: str) -> List[tuple[int, str]]:
        """Whitespace-aligned fallback split, used when offsets are unavailable."""
        budget = self.WINDOW_TOKENS * 3  # conservative chars-per-token estimate
        if len(text) <= budget:
            return [(0, text)]

        windows = []
        start = 0
        while start < len(text):
            end = min(start + budget, len(text))
            if end < len(text):
                split = text.rfind(" ", start, end)
                if split > start:
                    end = split
            windows.append((start, text[start:end]))
            start = end
        return windows

    def _windows(self, text: str) -> List[tuple[int, str]]:
        """Split text into model-sized windows as (char offset, window text)."""
        try:
            encoded = self.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = [
                (start, end) for start, end in encoded["offset_mapping"] if end > start
            ]
        except Exception as e:
            logger.warning(f"Token offsets unavailable, splitting on whitespace: {e}")
            return self._char_windows(text)

        if not offsets:
            return [(0, text)]
        if len(offsets) <= self.WINDOW_TOKENS:
            return [(0, text)]

        step = self.WINDOW_TOKENS - self.WINDOW_OVERLAP_TOKENS
        windows = []
        for first in range(0, len(offsets), step):
            window = offsets[first : first + self.WINDOW_TOKENS]
            if not window:
                break
            char_start = min(start for start, _ in window)
            char_end = max(end for _, end in window)
            windows.append((char_start, text[char_start:char_end]))
            if first + self.WINDOW_TOKENS >= len(offsets):
                break

        logger.info(
            f"Text of {len(text)} chars / {len(offsets)} tokens analysed "
            f"in {len(windows)} windows"
        )
        return windows

    def _sentence_chunks(
        self, text: str, start: int, end: int
    ) -> List[tuple[int, int]]:
        """Cut a span at newlines and sentence ends.

        Word-level aggregation can run a span straight through a full stop into
        the next line, so in a transcript "…Tolliametis.\\nNõustaja: Teie võlg…"
        came back as one ORGANIZATION and swallowed the speaker label.

        A newline is always a boundary - no entity name contains one. A full
        stop is only a boundary when whitespace follows, which keeps a domain
        such as www.eesti.ee intact, and not when what precedes it is an initial
        or a known abbreviation, which keeps "J. Tamm" and "Pärnu mnt. 12"
        whole.
        """
        chunks = []
        chunk_start = start

        for match in re.finditer(r"\n+|(?<=[.!?])[ \t]+", text[start:end]):
            boundary = start + match.start()
            is_newline = "\n" in match.group(0)
            # A newline is always a boundary; a full stop only when it really
            # ends a clause rather than abbreviating the word before it.
            if not is_newline and self._ends_in_abbreviation(
                text[chunk_start:boundary]
            ):
                continue
            chunks.append((chunk_start, boundary))
            chunk_start = start + match.end()

        chunks.append((chunk_start, end))
        return [(a, b) for a, b in chunks if text[a:b].strip()]

    @staticmethod
    def _ends_in_abbreviation(text: str) -> bool:
        """Whether a full stop here abbreviates a word rather than ends a clause."""
        last_word = text.rstrip().rsplit(maxsplit=1)[-1] if text.strip() else ""
        stem = last_word.rstrip(".!?")
        if len(stem) <= 1:  # an initial: "J."
            return True
        if re.fullmatch(r"(?:[^\W\d_]\.)+", last_word):  # "A.S."
            return True
        return stem.lower() in ABBREVIATIONS

    def _runs_without_addresses(
        self, text: str, start: int, end: int
    ) -> List[tuple[int, int]]:
        """Split a span into runs of words, excluding any that hold an address.

        Only the offending word is dropped, never the whole span: a span
        covering "Jaan Tamm jaan@eesti.ee" has to keep protecting the name.
        """
        runs: List[tuple[int, int]] = []
        run_start: Optional[int] = None
        run_end = start

        for token in re.finditer(r"\S+", text[start:end]):
            if "@" in token.group(0):
                if run_start is not None:
                    runs.append((run_start, run_end))
                    run_start = None
                continue
            if run_start is None:
                run_start = start + token.start()
            run_end = start + token.end()

        if run_start is not None:
            runs.append((run_start, run_end))
        return runs

    def _normalize_span(self, text: str, start: int, end: int) -> List[tuple[int, int]]:
        """Tidy one model span into zero or more spans worth reporting.

        Word-level aggregation is greedy: it takes in trailing punctuation and
        runs two neighbouring names together across a comma. It also labels an
        e-mail address as a person, because the local part often is a first
        name - and that span then outranks the e-mail recognizer's, replacing a
        [E-POST] placeholder with [ISIK]. So each span is cut at sentence and
        line boundaries, then on commas and semicolons, addresses are carved out
        word by word, and what is left is stripped back to alphanumeric edges.
        """
        spans = []
        clauses = [
            (sentence_start + clause.start(), sentence_start + clause.end())
            for sentence_start, sentence_end in self._sentence_chunks(text, start, end)
            for clause in re.finditer(r"[^,;]+", text[sentence_start:sentence_end])
        ]

        for chunk_start, chunk_end in clauses:
            for piece_start, piece_end in self._runs_without_addresses(
                text, chunk_start, chunk_end
            ):
                while piece_start < piece_end and not text[piece_start].isalnum():
                    piece_start += 1
                while piece_end > piece_start and not text[piece_end - 1].isalnum():
                    piece_end -= 1

                if piece_end <= piece_start:
                    continue
                # A registration plate is not an organisation name. The model
                # labels one ORGANIZATION at up to 1.00, which outranks any score
                # the CAR_NUMBER pattern can claim, so ownership has to be
                # settled here rather than by score.
                if self.PLATE_SHAPE.fullmatch(text[piece_start:piece_end]):
                    continue
                spans.append((piece_start, piece_end))
        return spans

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> List[RecognizerResult]:
        """Analyze text using EstBERT ONNX model"""
        results = []

        if not text or not text.strip():
            return results

        try:
            # ONNX inference - releases GIL, allows true parallel execution
            ner_results = []
            seen: set[tuple[str, int, int]] = set()
            for offset, window in self._windows(text):
                window_results = self.nlp_pipeline(window)
                if not isinstance(window_results, list):
                    logger.warning(f"Unexpected NER output format: {window_results}")
                    continue
                for entity in window_results:
                    entity["start"] += offset
                    entity["end"] += offset
                    # Overlapping windows can report the same span twice.
                    key = (
                        str(entity.get("entity_group", entity.get("entity", ""))),
                        entity["start"],
                        entity["end"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    ner_results.append(entity)

            for entity in ner_results:
                entity_type = (
                    entity.get("entity_group", entity.get("entity", ""))
                    .replace("B-", "")
                    .replace("I-", "")
                )
                presidio_entity = self.label_mapping.get(entity_type, entity_type)

                if presidio_entity not in entities:
                    continue

                for start, end in self._normalize_span(
                    text, entity["start"], entity["end"]
                ):
                    result = RecognizerResult(
                        entity_type=presidio_entity,
                        start=start,
                        end=end,
                        score=entity["score"],
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Error in EstBERT ONNX analysis: {e}")

        return results


class DenylistRecognizer(EntityRecognizer):
    """Custom recognizer for denylist words"""

    def __init__(
        self,
        denylist: List[str],
        entity_type: str = "DENYLIST_MATCH",
        score: float = 1.0,
        supported_language: str = "xx",
    ) -> None:
        super().__init__(
            supported_entities=[entity_type],
            supported_language=supported_language,
            name="Denylist_Recognizer",
        )
        self.denylist = [word.lower() for word in denylist]
        self.entity_type = entity_type
        self.score = score

    def load(self) -> None:
        """Load method - required by Presidio"""
        pass

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> List[RecognizerResult]:
        """Find denylist words in text"""
        results = []
        text_lower = text.lower()

        for word in self.denylist:
            start = 0
            while True:
                start = text_lower.find(word, start)
                if start == -1:
                    break

                end = start + len(word)
                is_start_boundary = start == 0 or not text[start - 1].isalnum()
                is_end_boundary = end == len(text) or not text[end].isalnum()

                if is_start_boundary and is_end_boundary:
                    result = RecognizerResult(
                        entity_type=self.entity_type,
                        start=start,
                        end=end,
                        score=self.score,
                    )
                    results.append(result)

                start = end

        return results


def apply_allowlist(
    results: List[RecognizerResult], text: str, allowlist: List[str]
) -> List[RecognizerResult]:
    """Filter out results that match words in the allowlist

    A detected span is dropped when it is entirely allowlisted. When an
    allowlisted term only sits at one edge of a wider span, the span is trimmed
    back instead of being kept whole - otherwise a model span such as
    "Phoenix Tallinnas" would anonymise an allowlisted "Tallinn" along with the
    codename next to it.
    """
    if not allowlist:
        return results

    allowlist_lower = [word.lower() for word in allowlist]
    filtered_results = []

    for result in results:
        start, end = result.start, result.end

        trimmed = True
        while trimmed and start < end:
            trimmed = False
            span_lower = text[start:end].lower()
            for term in allowlist_lower:
                if not term or len(term) > len(span_lower):
                    continue
                if span_lower == term:
                    start = end
                    trimmed = True
                    break
                if (
                    span_lower.endswith(term)
                    and not span_lower[-len(term) - 1].isalnum()
                ):
                    end -= len(term)
                    trimmed = True
                    break
                if span_lower.startswith(term) and not span_lower[len(term)].isalnum():
                    start += len(term)
                    trimmed = True
                    break
            # Whitespace and punctuation exposed by a trim are not PII either.
            while start < end and not text[start].isalnum():
                start += 1
            while end > start and not text[end - 1].isalnum():
                end -= 1

        if start >= end:
            logger.debug(
                f"Filtered out '{text[result.start : result.end]}' due to allowlist"
            )
            continue

        if (start, end) != (result.start, result.end):
            logger.debug(
                f"Trimmed '{text[result.start : result.end]}' to "
                f"'{text[start:end]}' due to allowlist"
            )
            result = RecognizerResult(
                entity_type=result.entity_type,
                start=start,
                end=end,
                score=result.score,
            )

        filtered_results.append(result)

    return filtered_results


def apply_score_thresholds(
    results: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Drop results below the threshold configured for their entity type."""
    kept = []
    for result in results:
        threshold = entity_score_thresholds.get(
            result.entity_type, default_score_threshold
        )
        if result.score >= threshold:
            kept.append(result)
        else:
            logger.debug(
                f"Dropped {result.entity_type} '{text[result.start : result.end]}' "
                f"scoring {result.score:.2f} below {threshold}"
            )
    return kept


def subtract_spans(
    result: RecognizerResult, blockers: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Cut the blocker spans out of `result` and return whatever is left of it.

    Used to carve a denylist match out of an overlapping detection without
    losing the rest of that detection, which may be real PII in its own right.
    """
    pieces = [(result.start, result.end)]

    for blocker in blockers:
        remaining = []
        for start, end in pieces:
            if blocker.end <= start or blocker.start >= end:
                remaining.append((start, end))
                continue
            if start < blocker.start:
                remaining.append((start, blocker.start))
            if blocker.end < end:
                remaining.append((blocker.end, end))
        pieces = remaining

    trimmed = []
    for start, end in pieces:
        while start < end and not text[start].isalnum():
            start += 1
        while end > start and not text[end - 1].isalnum():
            end -= 1
        if end > start:
            trimmed.append(
                RecognizerResult(
                    entity_type=result.entity_type,
                    start=start,
                    end=end,
                    score=result.score,
                )
            )
    return trimmed


def create_pattern_recognizers(config: dict, language: str) -> List[PatternRecognizer]:
    """Create pattern-based recognizers from configuration"""
    recognizers = []
    recognizers_config = config.get("recognizers", [])

    logger.info(
        f"Creating {len(recognizers_config)} pattern recognizers for language '{language}'"
    )

    for rec_config in recognizers_config:
        try:
            name = rec_config.get("name")
            supported_entity = rec_config.get("supported_entity")
            patterns_config = rec_config.get("patterns", [])

            patterns = []
            for pattern_config in patterns_config:
                pattern = Pattern(
                    name=pattern_config.get("name"),
                    regex=pattern_config.get("regex"),
                    score=pattern_config.get("score", 0.5),
                )
                patterns.append(pattern)

            recognizer = PatternRecognizer(
                supported_entity=supported_entity,
                patterns=patterns,
                name=name,
                supported_language=language,
            )
            recognizers.append(recognizer)
            logger.info(f"  ✓ Created: {name} for {supported_entity}")

        except Exception as e:
            logger.error(
                f"  ✗ Failed to create {rec_config.get('name', 'unknown')}: {e}"
            )

    return recognizers


def load_presidio_from_config(config_path: str) -> AnalyzerEngine:
    """Load complete Presidio analyzer from YAML configuration with ONNX"""
    logger.info("=" * 80)
    logger.info("LOADING PRESIDIO ANALYZER WITH ONNX OPTIMIZATION")
    logger.info("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    supported_languages = config.get("supported_languages", ["xx"])
    logger.info(f"Supported languages: {supported_languages}")

    # Create NLP engine
    logger.info("Creating NLP engine...")
    try:
        nlp_provider = NlpEngineProvider(nlp_configuration=config["nlp_configuration"])
        nlp_engine = nlp_provider.create_engine()
        logger.info("✓ NLP engine created")
    except Exception as e:
        logger.error(f"✗ NLP engine creation failed: {e}")
        raise

    # Score thresholds. The engine gets the lowest threshold any entity uses so
    # it cannot pre-filter a result that an entity-specific threshold would still
    # admit; apply_score_thresholds() then filters per entity.
    global entity_score_thresholds, default_score_threshold
    default_score_threshold = float(config.get("default_score_threshold", 0.8))
    entity_score_thresholds = {
        str(entity): float(score)
        for entity, score in (config.get("entity_score_thresholds") or {}).items()
    }
    engine_threshold = min([default_score_threshold, *entity_score_thresholds.values()])
    logger.info(
        f"Score thresholds: default {default_score_threshold}, "
        f"per-entity {entity_score_thresholds or 'none'}, engine {engine_threshold}"
    )

    # Create analyzer
    logger.info("Creating AnalyzerEngine...")
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=supported_languages,
        default_score_threshold=engine_threshold,
    )
    logger.info(" AnalyzerEngine created")

    # Remove unwanted recognizers.
    #
    # SpacyRecognizer wraps xx_ent_wiki_sm, the multilingual model that backs the
    # NlpEngine. Running the two NER sources separately traced every PERSON false
    # positive in the audit to it - sentence-initial verbs such as Soovin and
    # Palun, common nouns such as Otsus and Kaebuse, and worse, agencies labelled
    # as people: Töötukassas, Vabariigi Valitsus, Tallinna Linnavalitsusele. The
    # Estonian model labels all of those correctly. spaCy stays as the NlpEngine
    # for tokenisation and lemmas; it just no longer contributes entities.
    unwanted_recognizers = ["MedicalLicenseRecognizer", "SpacyRecognizer"]
    for recognizer_name in unwanted_recognizers:
        try:
            analyzer.registry.remove_recognizer(recognizer_name)
            logger.info(f"  Removed unwanted recognizer: {recognizer_name}")
        except Exception as e:
            logger.debug(f"  Could not remove {recognizer_name}: {e}")

    # Add recognizers for each supported language
    for lang in supported_languages:
        logger.info(f"\nAdding recognizers for language: {lang}")

        # 1. Add EstBERT ONNX recognizer
        try:
            estbert_config = config.get("estbert_configuration", {})
            model_name = estbert_config.get("model_name", "tartuNLP/EstBERT_NER")
            estbert_recognizer = EstBERTRecognizerONNX(
                model_name=model_name, supported_language=lang
            )
            analyzer.registry.add_recognizer(estbert_recognizer)
            logger.info("  EstBERT ONNX recognizer added")
        except Exception as e:
            logger.error(f"  EstBERT ONNX recognizer failed: {e}")

        # 2. Add pattern recognizers
        try:
            pattern_recognizers = create_pattern_recognizers(config, lang)
            for recognizer in pattern_recognizers:
                analyzer.registry.add_recognizer(recognizer)
            logger.info(f"  ✓ {len(pattern_recognizers)} pattern recognizers added")
        except Exception as e:
            logger.error(f"  ✗ Pattern recognizers failed: {e}")

    # Verify
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    for lang in supported_languages:
        try:
            recognizers = analyzer.get_recognizers(lang)
            entities = analyzer.get_supported_entities(lang)

            logger.info(f"\nLanguage: {lang}")
            logger.info(f"  Recognizers ({len(recognizers)}):")
            for rec in recognizers:
                logger.info(f"    - {rec.name}")
            logger.info(f"  Entities ({len(entities)}): {entities}")

            if len(recognizers) == 0:
                logger.error(f"  WARNING: No recognizers for language {lang}!")

        except Exception as e:
            logger.error(f"  ✗ Error verifying {lang}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYZER READY WITH ONNX OPTIMIZATION")
    logger.info("=" * 80 + "\n")

    return analyzer


def analyze_with_lists(
    analyzer: AnalyzerEngine,
    text: str,
    entities: List[str],
    language: str = "xx",
    allowlist: Optional[List[str]] = None,
    denylist: Optional[List[str]] = None,
    return_decision_process: bool = False,
    correlation_id: Optional[str] = None,
) -> List[RecognizerResult]:
    """Analyze text with allowlist and denylist support"""

    # Check recognizers
    try:
        recognizers = analyzer.get_recognizers(language)
        if len(recognizers) == 0:
            error_msg = f"No recognizers registered for language '{language}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"Failed to get recognizers: {e}")
        raise

    try:
        results = analyzer.analyze(
            text=text,
            entities=entities,
            language=language,
            return_decision_process=return_decision_process,
            correlation_id=correlation_id,
        )
        results = apply_score_thresholds(results, text)
        logger.info(f"  Found {len(results)} entities")
    except Exception as e:
        logger.error(f"  Analysis failed: {e}", exc_info=True)
        raise

    # Apply allowlist
    if allowlist:
        results = apply_allowlist(results, text, allowlist)

    # Add denylist
    if denylist:
        denylist_recognizer = DenylistRecognizer(
            denylist=denylist, entity_type="DENYLIST_MATCH", supported_language=language
        )
        denylist_results = denylist_recognizer.analyze(text, ["DENYLIST_MATCH"])

        # A denylist entry is a promise that the word is treated as PII, so it
        # has to win any overlap. Left to compete, the anonymizer's conflict
        # resolution preferred a longer overlapping NER span and dropped the
        # DENYLIST_MATCH result, which silently discarded the operator chosen
        # for it - with DEFAULT=keep and DENYLIST_MATCH=redact the denylisted
        # word was published verbatim.
        #
        # The overlapping result is cut back rather than discarded: a merged
        # span like "Kalle Kask Phoenixis" must keep protecting the name once
        # the denylisted word is carved out of it.
        if denylist_results:
            kept = []
            for result in results:
                clashes = [
                    d
                    for d in denylist_results
                    if result.start < d.end and d.start < result.end
                ]
                if clashes:
                    kept.extend(subtract_spans(result, clashes, text))
                else:
                    kept.append(result)
            results = kept

        results.extend(denylist_results)

    results.sort(key=lambda x: x.start)
    return results


def validate_config(config_path: str) -> tuple:
    """Validate Presidio configuration file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "supported_languages",
            "nlp_configuration",
            "estbert_configuration",
        ]

        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"

        return True, "Configuration is valid"

    except Exception as e:
        return False, f"Configuration error: {str(e)}"
