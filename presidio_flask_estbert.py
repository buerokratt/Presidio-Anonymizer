import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
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

logger = logging.getLogger("presidio-flask-api")


class EstBERTRecognizer(EntityRecognizer):
    """Custom recognizer for tartuNLP/EstBERT_NER model"""

    ENTITIES = ["PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "GPE"]

    def __init__(
        self, model_name: str = "tartuNLP/EstBERT_NER", supported_language: str = "xx"
    ):
        super().__init__(
            supported_entities=self.ENTITIES,
            supported_language=supported_language,
            name="EstBERT_NER_Recognizer",
        )

        logger.info(f"Loading EstBERT model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.nlp_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
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
                f" EstBERT recognizer initialized for language: {supported_language}"
            )
        except Exception as e:
            logger.error(f"✗ Failed to initialize EstBERT recognizer: {e}")
            raise

    def load(self) -> None:
        """Load method - required by Presidio"""
        pass

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """Analyze text using EstBERT NER model"""
        results = []

        if not text or not text.strip():
            return results

        try:
            ner_results = self.nlp_pipeline(text)

            for entity in ner_results:
                entity_type = (
                    entity.get("entity_group", entity.get("entity", ""))
                    .replace("B-", "")
                    .replace("I-", "")
                )
                presidio_entity = self.label_mapping.get(entity_type, entity_type)

                if presidio_entity in entities:
                    result = RecognizerResult(
                        entity_type=presidio_entity,
                        start=entity["start"],
                        end=entity["end"],
                        score=entity["score"],
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Error in EstBERT analysis: {e}")

        return results


class DenylistRecognizer(EntityRecognizer):
    """Custom recognizer for denylist words"""

    def __init__(
        self,
        denylist: List[str],
        entity_type: str = "DENYLIST_MATCH",
        score: float = 1.0,
        supported_language: str = "xx",
    ):
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
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
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
    """Filter out results that match words in the allowlist"""
    if not allowlist:
        return results

    allowlist_lower = [word.lower() for word in allowlist]
    filtered_results = []

    for result in results:
        detected_text = text[result.start : result.end].lower()
        if detected_text not in allowlist_lower:
            filtered_results.append(result)
        else:
            logger.debug(f"Filtered out '{detected_text}' due to allowlist")

    return filtered_results


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


def load_presidio_from_config(config_path: str):
    """Load complete Presidio analyzer from YAML configuration"""
    logger.info("=" * 80)
    logger.info("LOADING PRESIDIO ANALYZER")
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
        logger.info(" NLP engine created")
    except Exception as e:
        logger.error(f" NLP engine creation failed: {e}")
        raise

    # Create analyzer WITHOUT registry (we'll add recognizers manually)
    logger.info("Creating AnalyzerEngine...")
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=supported_languages,
        default_score_threshold=config.get("default_score_threshold", 0.8),
    )
    logger.info(" AnalyzerEngine created")
    unwanted_recognizers = ["MedicalLicenseRecognizer"]
    for recognizer_name in unwanted_recognizers:
        try:
            analyzer.registry.remove_recognizer(recognizer_name)
            logger.info(f"  Removed unwanted recognizer: {recognizer_name}")
        except Exception as e:
            logger.debug(f"  Could not remove {recognizer_name}: {e}")
    # Add recognizers for EACH supported language
    for lang in supported_languages:
        logger.info(f"\nAdding recognizers for language: {lang}")

        # 1. Add EstBERT recognizer
        try:
            estbert_config = config.get("estbert_configuration", {})
            model_name = estbert_config.get("model_name", "tartuNLP/EstBERT_NER")
            estbert_recognizer = EstBERTRecognizer(
                model_name=model_name, supported_language=lang
            )
            analyzer.registry.add_recognizer(estbert_recognizer)
            logger.info("   EstBERT recognizer added")
        except Exception as e:
            logger.error(f"   EstBERT recognizer failed: {e}")

        # 2. Add pattern recognizers
        try:
            pattern_recognizers = create_pattern_recognizers(config, lang)
            for recognizer in pattern_recognizers:
                analyzer.registry.add_recognizer(recognizer)
            logger.info(f"   {len(pattern_recognizers)} pattern recognizers added")
        except Exception as e:
            logger.error(f"   Pattern recognizers failed: {e}")

    # Verify recognizers were added
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
                logger.error(f"    WARNING: No recognizers for language {lang}!")

        except Exception as e:
            logger.error(f"   Error verifying {lang}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYZER READY")
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

    logger.info("analyze_with_lists:")
    logger.info(f"  Text length: {len(text)}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Entities: {entities}")

    # Check recognizers BEFORE analysis
    try:
        recognizers = analyzer.get_recognizers(language)
        logger.info(f"  Available recognizers: {len(recognizers)}")

        if len(recognizers) == 0:
            error_msg = f"No recognizers registered for language '{language}'. This is a configuration error."
            logger.error(error_msg)
            raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"  Failed to get recognizers: {e}")
        raise

    # Run analysis
    try:
        results = analyzer.analyze(
            text=text,
            entities=entities,
            language=language,
            return_decision_process=return_decision_process,
            correlation_id=correlation_id,
        )
        logger.info(f"  Found {len(results)} entities")
    except Exception as e:
        logger.error(f"  Analysis failed: {e}", exc_info=True)
        raise

    # Apply allowlist
    if allowlist:
        results = apply_allowlist(results, text, allowlist)
        logger.info(f"  After allowlist: {len(results)} entities")

    # Add denylist
    if denylist:
        denylist_recognizer = DenylistRecognizer(
            denylist=denylist, entity_type="DENYLIST_MATCH", supported_language=language
        )
        denylist_results = denylist_recognizer.analyze(text, ["DENYLIST_MATCH"])
        logger.info(f"  Denylist found: {len(denylist_results)} entities")
        results.extend(denylist_results)

    results.sort(key=lambda x: x.start)
    logger.info(f"  Final: {len(results)} entities")

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
