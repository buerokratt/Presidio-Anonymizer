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
from typing import Callable, Iterator, List, Optional
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

# Score thresholds. AnalyzerEngine supports one threshold for everything, which
# is the wrong shape for this pipeline: the NER model emits real probabilities,
# while a YAML pattern carries a constant its author typed. Comparing the two
# against one cut is what made a pattern silently unreachable whenever its
# score sat below the threshold - IP_ADDRESS was listed as a supported entity
# for months without ever being reportable, and eleven patterns in the stanza
# config are in that state today.
#
# So the engine threshold is 0 and all filtering happens in
# apply_score_thresholds():
#
#   pattern recognizers from the YAML   never filtered. A regex either matches
#                                       or it does not; there is no confidence
#                                       to weigh. If a pattern is not trusted,
#                                       the fix is the pattern, not a number
#                                       that quietly disables it.
#   everything else                     filtered per entity, falling back to
#                                       default_score_threshold. This is where
#                                       a score means something: the model's
#                                       own confidence, and the validated
#                                       built-ins, which score 1.0 when their
#                                       checksum passes and 0 when it fails.
#
# Pattern scores still matter for conflict resolution, where Presidio prefers
# the higher-scored of two overlapping spans.
#
# The app builds exactly one analyzer per process, so these live at module
# level instead of being threaded through every call site.
entity_score_thresholds: dict[str, float] = {}
default_score_threshold: float = 0.8
pattern_recognizer_names: set[str] = set()
# The subset whose matches have passed a checksum (`validator:` in the YAML).
# apply_pattern_precedence() lets them win a partial overlap.
checked_recognizer_names: set[str] = set()
# Each YAML recognizer's compiled patterns, by recognizer name, so a span cut
# down by apply_pattern_precedence() can be checked against its own shape.
pattern_regexes: dict[str, List[re.Pattern[str]]] = {}


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
    # IGNORECASE to match the CAR_NUMBER pattern this defends: Presidio
    # compiles that one case-insensitively, so without the flag here a
    # lower-case plate slipped past the guard and was reported as an
    # organisation.
    RECOGNIZER_NAME = "EstBERT_NER_ONNX_Recognizer"

    PLATE_SHAPE = re.compile(r"[0-9]{2,3}\s?[A-ZÕÄÖÜ]{3}", re.IGNORECASE)

    DEFAULT_LABEL_MAPPING = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "GPE",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
    }

    def __init__(
        self,
        model_name: str = "tartuNLP/EstBERT_NER",
        supported_language: str = "xx",
        label_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            supported_entities=self.ENTITIES,
            supported_language=supported_language,
            name=self.RECOGNIZER_NAME,
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

            # estbert_configuration.entity_mapping in the YAML was never read:
            # this mapping was hardcoded, so the config key looked live and was
            # not. The config now wins, and the hardcoded table is the fallback
            # for a config that omits it.
            self.label_mapping = dict(label_mapping or self.DEFAULT_LABEL_MAPPING)

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

    @staticmethod
    def _straddles(start: int, end: int, others: List[RecognizerResult]) -> bool:
        """True if [start, end) partly overlaps any of `others`.

        Disjoint is fine and so is full containment either way round; only a
        partial overlap is a problem, because the anonymizer has no rule for it
        and emits both spans.
        """
        for other in others:
            if other.end <= start or end <= other.start:
                continue
            if start <= other.start and other.end <= end:
                continue
            if other.start <= start and end <= other.end:
                continue
            return True
        return False

    @classmethod
    def _merge_adjacent_spans(
        cls, text: str, results: List[RecognizerResult]
    ) -> List[RecognizerResult]:
        """Join same-type spans separated only by spaces, keeping the best score.

        The model labels a multi-word name one word at a time often enough that
        "New Yorki" comes back as GPE("New") and GPE("Yorki") scored
        separately. Filtering them separately drops whichever falls below the
        threshold and leaves that half of the name in clear text - the defect
        this issue was opened for, still reachable for names the model has weak
        evidence for.

        Merging before the filter is what makes the scores agree with the
        output. AnonymizerEngine already rewrites same-type spans separated by
        spaces as one placeholder, so by the time the text is produced the
        pipeline has treated them as a single entity regardless; only the
        threshold filter was treating them as two. The merged span takes the
        higher score, so strong evidence for one word carries the rest of the
        name.

        A merge is abandoned when the result would half-cover some other span.
        Presidio resolves one span containing another, but not two that only
        partly overlap: it writes both, and the shared characters come out
        twice. The model does emit spans that sit inside each other - a
        one-letter LOCATION fragment inside a GPE, say - and absorbing the
        fragment into its neighbour is what turns containment into a partial
        overlap.

        A merged span that turns out to be plate-shaped is dropped, for the
        same reason _normalize_span() drops one: a registration number is not
        an organisation name, and the model outscores the CAR_NUMBER pattern.

        The gap has to be spaces alone, which is the anonymizer's own rule.
        That is also what keeps a deliberate split from being undone:
        _normalize_span cuts at commas, sentence ends and newlines, so its
        pieces are never separated by spaces alone.
        """
        if not results:
            return results

        ordered = sorted(results, key=lambda r: (r.start, r.end))
        merged = [ordered[0]]
        for index, current in enumerate(ordered[1:], start=1):
            previous = merged[-1]
            gap = text[previous.end : current.start]
            others = [
                other
                for position, other in enumerate(ordered)
                if other is not current and position != index and other is not previous
            ]
            if (
                current.entity_type == previous.entity_type
                and previous.end <= current.start
                and gap != ""
                and gap.strip(" ") == ""
                and not cls._straddles(previous.start, current.end, others)
            ):
                previous.end = current.end
                previous.score = max(previous.score, current.score)
            else:
                merged.append(current)

        # The plate guard has to run again here. _normalize_span() drops a piece
        # that is a plate, but the model splits "45 XYZ" into "45" and "XYZ",
        # and neither half is plate-shaped on its own - so the guard passed them
        # and this merge put them back together as one ORGANIZATION.
        return [
            span
            for span in merged
            if not cls.PLATE_SHAPE.fullmatch(text[span.start : span.end])
        ]

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

        return self._merge_adjacent_spans(text, results)


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


def is_pattern_result(result: RecognizerResult) -> bool:
    """True if one of the YAML pattern recognizers produced this result."""
    source = (result.recognition_metadata or {}).get(
        RecognizerResult.RECOGNIZER_NAME_KEY
    )
    return source in pattern_recognizer_names


def is_checked_result(result: RecognizerResult) -> bool:
    """True if a recognizer with a `validator:` produced this result."""
    source = (result.recognition_metadata or {}).get(
        RecognizerResult.RECOGNIZER_NAME_KEY
    )
    return source in checked_recognizer_names


def is_model_result(result: RecognizerResult) -> bool:
    """True if the NER model produced this result."""
    source = (result.recognition_metadata or {}).get(
        RecognizerResult.RECOGNIZER_NAME_KEY
    )
    return source == EstBERTRecognizerONNX.RECOGNIZER_NAME


def apply_pattern_precedence(
    results: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Let a pattern own the text it matched when a model span overlaps it.

    A pattern match is deterministic; a model span is a guess. Left to compete
    on score the model wins every time - it labels confidently enough to
    outrank any score a pattern can carry - so "123 abc" came back as
    [ORGANISATSIOON] rather than [AUTONUMBER], and "err.ee/uudised?id=12" as an
    organisation rather than a URL. Nothing leaked in either case; the
    placeholder was simply wrong, which is still a defect to anyone reading the
    output.

    The model span is cut back rather than discarded, the same way a denylist
    match is carved out of one: a span over "Auto 123 ABC" still has "Auto" to
    report once the plate is taken out of it.

    Only a clash of different types counts. Two spans of the same type are a
    duplicate, not a mislabel, and Presidio already settles those.

    The model's spans yield to every other recognizer, Presidio's built-ins
    included - those are deterministic too, and the validated ones (e-mail by
    its TLD check, and our own IBAN and card patterns by mod-97 and Luhn) are
    better evidence than anything here. Leaving them out was not an option either: a model span
    reading "Konto EE38" half-covers the IBAN beside it, and a partial overlap
    makes the anonymizer write both spans, so the text came back with "EE38"
    in it twice.

    What never happens is one non-model span carving another. Cutting a card
    match out of the IBAN around it split one account number into
    "[PANGAKONTO] [PANGAKAART]"; left alone, the card span sits wholly inside
    the IBAN and the anonymizer drops it as contained.
    """
    blockers = [result for result in results if not is_model_result(result)]
    if not blockers:
        return results
    results = trim_straddlers_of_checked(results, text)

    kept: List[RecognizerResult] = []
    for result in results:
        if not is_model_result(result):
            kept.append(result)
            continue
        clashes = [
            blocker
            for blocker in blockers
            if blocker.entity_type != result.entity_type
            and result.start < blocker.end
            and blocker.start < result.end
        ]
        if clashes:
            kept.extend(subtract_spans(result, clashes, text))
        else:
            kept.append(result)
    return kept


def apply_score_thresholds(
    results: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Drop results below the threshold configured for their entity type.

    Results produced by the YAML pattern recognizers are exempt - see the note
    on the threshold globals above.
    """
    kept = []
    for result in results:
        if is_pattern_result(result):
            kept.append(result)
            continue

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


def trim_straddlers_of_checked(
    results: List[RecognizerResult], text: str
) -> List[RecognizerResult]:
    """Cut a checksummed span out of an unchecked pattern span that half-covers it.

    In "EE38 2200 2210 2014 5685 2024" the IBAN ends before the year, and the
    phone pattern reads the last group and the year, "5685 2024", as a number.
    The two spans partly overlap, which the anonymizer has no rule for: it
    wrote "[PANGAKONTO][TELEFON]", swallowing the year and the space before
    it. The IBAN passed mod-97; the phone match is a shape, so the IBAN keeps
    the shared text.

    What is left of the phone span is kept only if it still is what its
    recognizer looks for - a whole match of one of its own patterns. "2024" is
    not a phone number and is left as text. In "+372 5512 3456 4242 4242 4242
    4242" the phone pattern runs into the card, and "+372 5512 3456" is one, so
    it stays anonymised: dropping the span whole would have printed it.

    Only a partial overlap with a different type counts. One span inside
    another is settled by the anonymizer, and two checked spans are left to
    it as well.
    """
    checked = [result for result in results if is_checked_result(result)]
    if not checked:
        return results

    kept: List[RecognizerResult] = []
    for result in results:
        if not is_pattern_result(result) or is_checked_result(result):
            kept.append(result)
            continue
        straddled = [
            other
            for other in checked
            if other.entity_type != result.entity_type
            and result.start < other.end
            and other.start < result.end
            and not (result.start <= other.start and other.end <= result.end)
            and not (other.start <= result.start and result.end <= other.end)
        ]
        if not straddled:
            kept.append(result)
            continue

        source = (result.recognition_metadata or {}).get(
            RecognizerResult.RECOGNIZER_NAME_KEY, ""
        )
        own_patterns = pattern_regexes.get(source, [])
        for piece in subtract_spans(result, straddled, text):
            # subtract_spans trims punctuation off both ends, which takes the
            # "+" off "+372 5512 3456"; give back what the original span had.
            while (
                piece.start > result.start
                and not text[piece.start - 1].isspace()
                and not text[piece.start - 1].isalnum()
            ):
                piece.start -= 1
            if any(
                regex.fullmatch(text, piece.start, piece.end) for regex in own_patterns
            ):
                piece.recognition_metadata = result.recognition_metadata
                kept.append(piece)
            else:
                logger.debug(
                    f"Dropped {result.entity_type} remainder "
                    f"[{piece.start}:{piece.end}] after a checksummed span "
                    f"took the rest"
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


def validate_ee_personal_code(text: str) -> bool:
    """Check an isikukood's mod-11 check digit.

    The regex can only describe the shape - eleven digits, a first digit of
    1-8, a month of 01-12 and a day of 01-31 - which leaves "39001010001"
    looking exactly like a real code. It is not one: the check digit is the
    thing that decides, and the standard defines nothing else to verify.

    An earlier version also required the birth date to be a real calendar date,
    which rejected the 30th of February. That is stricter than the standard,
    which specifies only 01-31 for the day, and strictness in that direction is
    the dangerous one here: a real code with an odd date would go unanonymised.
    It also earned nothing - every impossible date to hand fails the check digit
    anyway.

    The algorithm cannot catch every single-digit error: 51107121760 and
    61107121760 differ by one digit and both verify. That is a known flaw in
    the published method, so both are accepted here too.
    """
    if len(text) != 11 or not text.isdigit():
        return False

    digits = [int(character) for character in text]
    if not 1 <= digits[0] <= 8:
        return False

    # The published algorithm: weight the first ten digits, take modulo 11, and
    # if that lands on 10, weight them again on the second scale. A second 10
    # becomes 0.
    first = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1)
    second = (3, 4, 5, 6, 7, 8, 9, 1, 2, 3)
    remainder = sum(d * w for d, w in zip(digits[:10], first, strict=True)) % 11
    if remainder == 10:
        remainder = sum(d * w for d, w in zip(digits[:10], second, strict=True)) % 11
        if remainder == 10:
            remainder = 0
    return remainder == digits[10]


# Spaces, non-breaking spaces and hyphens are how card and account numbers are
# grouped when written out; none of them is part of the number.
_NUMBER_SEPARATORS = re.compile(r"[\s\-]")


def validate_luhn(text: str) -> bool:
    """Check a payment card number's Luhn check digit.

    Card numbers run 13-19 digits. Every second digit from the right is doubled
    (subtracting 9 when that exceeds 9) and the total must be a multiple of 10.

    One in ten digit runs passes Luhn by chance, and a rejected match is
    searched again from inside (ValidatingPatternRecognizer.analyze), so the
    lower bound is what keeps the tail of a rejected number - "5678 9012 3456"
    of "1234 5678 9012 3456" - from coming back as a card.
    """
    digits = _NUMBER_SEPARATORS.sub("", text)
    if not re.fullmatch(r"[0-9]{13,19}", digits):
        return False

    total = 0
    for position, character in enumerate(reversed(digits)):
        digit = int(character)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# IBAN length per country, from the ISO 13616 registry. A country that is not
# here does not issue IBANs, so two letters in front of a number are not one.
IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BI": 27, "BR": 29, "BY": 28, "CH": 21, "CR": 22,
    "CY": 28, "CZ": 24, "DE": 22, "DJ": 27, "DK": 18, "DO": 28, "EE": 20,
    "EG": 29, "ES": 24, "FI": 18, "FK": 18, "FO": 18, "FR": 27, "GB": 22,
    "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HN": 28, "HR": 21,
    "HU": 28, "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30,
    "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20,
    "LV": 21, "LY": 25, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MN": 20,
    "MR": 27, "MT": 31, "MU": 30, "NI": 28, "NL": 18, "NO": 15, "OM": 23,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "RU": 33,
    "SA": 24, "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24, "SM": 27,
    "SO": 23, "ST": 25, "SV": 28, "TL": 23, "TN": 24, "TR": 26, "UA": 29,
    "VA": 22, "VG": 24, "XK": 20, "YE": 30,
}  # fmt: skip


def _compact_iban(text: str) -> str:
    return _NUMBER_SEPARATORS.sub("", text).upper()


def validate_iban_shape(text: str) -> bool:
    """Check that an IBAN has a real country code and that country's length.

    This is what survives a typo: a mistyped digit leaves both intact, while a
    serial number such as "SN12 3456 7890 1234" has neither.
    """
    compact = _compact_iban(text)
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]+", compact):
        return False
    return IBAN_LENGTHS.get(compact[:2]) == len(compact)


def validate_iban(text: str) -> bool:
    """Check an IBAN in full: country, length and mod-97 check digits.

    Case and grouping are not part of the number: "ee38 2200 ..." is the same
    account as "EE382200...". The first four characters move to the end, each
    letter becomes two digits (A=10 ... Z=35), and the result modulo 97 must be 1.
    """
    if not validate_iban_shape(text):
        return False

    compact = _compact_iban(text)
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(character, 36)) for character in rearranged)
    return int(numeric) % 97 == 1


# Validators a YAML recognizer can ask for by name, with `validator:`. A
# pattern describes a shape; a validator decides whether that shape is a real
# identifier, which is the one thing a regex cannot do.
VALIDATORS: dict[str, Callable[[str], bool]] = {
    "ee_personal_code": validate_ee_personal_code,
    "luhn": validate_luhn,
    "iban": validate_iban,
    "iban_shape": validate_iban_shape,
}

# What a failed check does, set per recognizer with `validator_mode:`.
#
#   strict  the match is dropped. For shapes that occur by accident - sixteen
#           digits are as often an order or reference number as a card.
#   boost   the match is kept at the pattern's score; only a pass changes it,
#           to 1.0. For shapes distinctive enough that a failed check means a
#           typo rather than a coincidence: "EE38 2200 ..." with one digit
#           wrong is still someone's account number, and a missed entity leaks
#           where a false positive only over-redacts.
#
# A boost recognizer can also name a `plausible:` check, from VALIDATORS. A
# match that fails the validator is then kept only if it passes this one, so
# the benefit of the doubt goes to a typo and not to anything of the shape.
VALIDATOR_MODES = ("strict", "boost")


_WORD_CHARACTER = re.compile(r"\w")


def find_validated_matches(
    regex: re.Pattern[str],
    text: str,
    judge: Callable[[str], Optional[bool]],
) -> Iterator[tuple[int, int, Optional[bool]]]:
    """Yield (start, end, verdict) for each match `judge` does not reject.

    `judge` returns True (verified), None (kept at the pattern's score) or
    False (rejected). Presidio walks matches with finditer, which resumes after
    a match's end whether or not it was accepted, and that loses real numbers
    in two ways:

      too long   A greedy pattern takes whatever digits follow, and the whole
                 then fails: "EE38 2200 2210 2014 5685 2024" is too long for an
                 Estonian IBAN, so the account inside it was never reported.
                 A rejected match is tried again from the same start, shorter,
                 ending only where a match could end anyway - before a
                 separator, never mid-token - longest first.
      too early  A match that starts too soon hides one inside it: in
                 "2024 4242 4242 4242 4242" the year and three groups fail
                 Luhn, and the card one group later was never tried. If no
                 shorter match is accepted either, the search resumes one
                 character after the start rather than after the end.
    """
    position = 0
    while (match := regex.search(text, position)) is not None:
        start, end = match.span()
        if start == end:
            position = end + 1
            continue

        accepted: Optional[tuple[int, Optional[bool]]] = None
        verdict = judge(text[start:end])
        if verdict is not False:
            accepted = (end, verdict)
        else:
            for shorter_end in range(end - 1, start, -1):
                if _WORD_CHARACTER.match(text, shorter_end):
                    continue
                if regex.fullmatch(text, start, shorter_end) is None:
                    continue
                verdict = judge(text[start:shorter_end])
                if verdict is not False:
                    accepted = (shorter_end, verdict)
                    break

        if accepted is None:
            position = start + 1
            continue
        accepted_end, verdict = accepted
        yield start, accepted_end, verdict
        position = accepted_end


class ValidatingPatternRecognizer(PatternRecognizer):
    """A PatternRecognizer that checks its matches before reporting them.

    Presidio scores a validated match 1.0 and an invalidated one 0, so in
    strict mode a shape-only match disappears instead of being reported with
    the pattern's nominal confidence. In boost mode a failed check reports
    nothing, which leaves the match at the pattern's own score - unless a
    `plausible` check is given and fails too.
    """

    def __init__(
        self,
        validator: Callable[[str], bool],
        strict: bool = True,
        plausible: Optional[Callable[[str], bool]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._validator = validator
        self._strict = strict
        self._plausible = plausible

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        if self._validator(pattern_text):
            return True
        if self._strict:
            return False
        if self._plausible is not None and not self._plausible(pattern_text):
            return False
        return None

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
        regex_flags: Optional[int] = None,
    ) -> List[RecognizerResult]:
        """Match like PatternRecognizer, but look again after a rejected match.

        See find_validated_matches() for what "again" means.
        """
        flags: int = regex_flags or self.global_regex_flags or 0
        results: List[RecognizerResult] = []
        for pattern in self.patterns:
            regex = re.compile(pattern.regex, flags)
            for start, end, validation_result in find_validated_matches(
                regex, text, self.validate_result
            ):
                score = (
                    EntityRecognizer.MAX_SCORE if validation_result else pattern.score
                )
                description = self.build_regex_explanation(
                    self.name,
                    pattern.name,
                    pattern.regex,
                    pattern.score,
                    validation_result,  # type: ignore[arg-type]
                    flags,
                )
                description.score = score
                results.append(
                    RecognizerResult(
                        entity_type=self.supported_entities[0],
                        start=start,
                        end=end,
                        score=score,
                        analysis_explanation=description,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
        return EntityRecognizer.remove_duplicates(results)


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

            validator_name = rec_config.get("validator")
            validator = VALIDATORS.get(validator_name) if validator_name else None
            if validator_name and validator is None:
                logger.error(
                    f"  Unknown validator '{validator_name}' for {name}; "
                    f"reporting matches unvalidated"
                )

            # An unrecognised mode falls back to boost: a misspelt setting
            # should over-redact, not silently start dropping matches.
            validator_mode = rec_config.get("validator_mode", "strict")
            if validator_mode not in VALIDATOR_MODES:
                logger.error(
                    f"  Unknown validator_mode '{validator_mode}' for {name}; "
                    f"using 'boost'"
                )
                validator_mode = "boost"

            # Same fallback for the plausibility check: an unknown name keeps
            # every failed match, as boost mode without one would.
            plausible_name = rec_config.get("plausible")
            plausible = VALIDATORS.get(plausible_name) if plausible_name else None
            if plausible_name and plausible is None:
                logger.error(
                    f"  Unknown plausible check '{plausible_name}' for {name}; "
                    f"keeping every match that fails validation"
                )

            if validator is not None:
                recognizer = ValidatingPatternRecognizer(
                    validator=validator,
                    strict=validator_mode == "strict",
                    plausible=plausible,
                    supported_entity=supported_entity,
                    patterns=patterns,
                    name=name,
                    supported_language=language,
                )
            else:
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

    # Score thresholds. The engine itself filters nothing: it would apply one
    # cut to model spans and pattern matches alike, and it runs before
    # apply_score_thresholds() can tell them apart.
    global entity_score_thresholds, default_score_threshold, pattern_recognizer_names
    global checked_recognizer_names, pattern_regexes
    default_score_threshold = float(config.get("default_score_threshold", 0.8))
    entity_score_thresholds = {
        str(entity): float(score)
        for entity, score in (config.get("entity_score_thresholds") or {}).items()
    }
    pattern_recognizer_names = set()
    checked_recognizer_names = set()
    pattern_regexes = {}
    logger.info(
        f"Score thresholds: default {default_score_threshold}, "
        f"per-entity {entity_score_thresholds or 'none'}, "
        f"patterns exempt, engine 0"
    )

    # Create analyzer
    logger.info("Creating AnalyzerEngine...")
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=supported_languages,
        default_score_threshold=0.0,
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
    #
    # CreditCardRecognizer and IbanRecognizer are replaced by the YAML
    # recognizers of the same entities, which run the same Luhn and mod-97
    # checks. The built-ins only match a card starting 1, 3, 4, 50-55 or 6
    # (missing Mastercard 2221-2720 and Maestro 56-58), only an upper-case
    # IBAN, and only "-" or " " as a separator, so a lower-case account number
    # or a card pasted with non-breaking spaces went unanonymised. Keeping them
    # alongside would leave two owners for one entity.
    unwanted_recognizers = [
        "MedicalLicenseRecognizer",
        "SpacyRecognizer",
        "CreditCardRecognizer",
        "IbanRecognizer",
    ]
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
                model_name=model_name,
                supported_language=lang,
                label_mapping=estbert_config.get("entity_mapping"),
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
                pattern_recognizer_names.add(recognizer.name)
                if isinstance(recognizer, ValidatingPatternRecognizer):
                    checked_recognizer_names.add(recognizer.name)
                pattern_regexes[recognizer.name] = [
                    re.compile(pattern.regex, recognizer.global_regex_flags or 0)
                    for pattern in recognizer.patterns
                ]
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
        results = apply_pattern_precedence(results, text)
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
