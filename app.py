import os
import yaml
import logging


from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Resource, Api, fields

from werkzeug.exceptions import HTTPException

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from utils import synthesize_all

# Import your custom configuration loader
from presidio_flask_estbert import (
    load_presidio_from_config,
    validate_config,
    analyze_with_lists,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presidio-flask-api")

DEFAULT_PORT = "8000"

WELCOME_MESSAGE = r"""
    _______  _______  _______  _______ _________ ______  _________ _______ 
    (  ____ )(  ____ )(  ____ \(  ____ \\__   __/(  __  \ \__   __|(  ___  )
    | (    )|| (    )|| (    \/| (    \/   ) (   | (  \  )   ) (   | (   ) |
    | (____)|| (____)|| (__    | (_____    | |   | |   ) |   | |   | |   | |
    |  _____)|     __)|  __)   (_____  )   | |   | |   | |   | |   | |   | |
    | (      | (\ (   | (            ) |   | |   | |   ) |   | |   | |   | |
    | )      | ) \ \__| (____/\/\____) |___) (___| (__/  )___) (___| (___) |
    |/       |/   \__/(_______/\_______)\_______/(______/ \_______/(_______)

    Estonian Presidio API Server with Allowlist/Denylist Support
    Powered by EstBERT + spaCy
                                                              
"""


class EstonianPresidioFlaskServer:
    """Flask server for Presidio with Estonian EstBERT support"""

    def __init__(self, config_path: str = "/app/config/presidio-spacy-estbert.yml"):
        self.config_path = config_path
        self.app = Flask(__name__)

        # Initialize Flask-RESTX with proper configuration
        self.api = Api(
            self.app,
            version="1.0",
            title="Estonian Presidio API",
            description="API for PII detection and anonymization using EstBERT + spaCy with allowlist/denylist support",
            doc="/docs/",  # Swagger UI will be available at /docs/
            validate=True,  # Enable request validation
            ordered=True,  # Keep endpoint order in documentation
        )

        # Enable CORS
        CORS(self.app)

        # Configure Flask
        self.app.config["JSON_AS_ASCII"] = False  # Support for Estonian characters
        self.app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

        # Initialize engines
        self._initialize_engines()

        # Define API models
        self._define_api_models()

        # Setup routes using Flask-RESTX
        self._setup_routes()

        # Setup error handlers
        self._setup_error_handlers()

        logger.info(WELCOME_MESSAGE)

    def _initialize_engines(self):
        """Initialize Presidio analyzer and anonymizer engines"""
        try:
            # Validate configuration
            is_valid, message = validate_config(self.config_path)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {message}")

            # Load configuration
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            # Initialize analyzer with EstBERT + spaCy
            logger.info("Initializing Presidio Analyzer with EstBERT + spaCy...")
            self.analyzer = load_presidio_from_config(self.config_path)

            # Initialize anonymizer
            logger.info("Initializing Presidio Anonymizer...")
            self.anonymizer = AnonymizerEngine()

            logger.info("Presidio engines initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Presidio engines: {e}")
            raise

    def _define_api_models(self):
        """Define Flask-RESTX models for request/response validation and documentation"""

        # Health check response model
        self.health_model = self.api.model(
            "HealthResponse",
            {
                "status": fields.String(
                    required=True, description="Service status", example="healthy"
                ),
                "service": fields.String(
                    required=True,
                    description="Service name",
                    example="Estonian Presidio API",
                ),
                "version": fields.String(
                    required=True, description="API version", example="1.0.0"
                ),
                "supported_languages": fields.List(
                    fields.String, description="Supported languages", example=["et"]
                ),
                "model": fields.String(
                    description="Model information",
                    example="tartuNLP/EstBERT_NER + spaCy",
                ),
            },
        )

        # Anonymizer operator models for better documentation
        self.operator_replace_model = self.api.model(
            "ReplaceOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type",
                    example="replace",
                    enum=["replace"],
                ),
                "new_value": fields.String(
                    required=True,
                    description="Text to replace the PII with",
                    example="[PERSON]",
                ),
            },
        )

        self.operator_redact_model = self.api.model(
            "RedactOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type - completely removes the PII",
                    example="redact",
                    enum=["redact"],
                )
            },
        )

        self.operator_mask_model = self.api.model(
            "MaskOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type",
                    example="mask",
                    enum=["mask"],
                ),
                "masking_char": fields.String(
                    required=True,
                    description="Character to use for masking",
                    example="*",
                    default="*",
                ),
                "chars_to_mask": fields.Integer(
                    required=True, description="Number of characters to mask", example=4
                ),
                "from_end": fields.Boolean(
                    description="Mask from the end of the string (true) or beginning (false)",
                    example=True,
                    default=True,
                ),
            },
        )

        self.operator_hash_model = self.api.model(
            "HashOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type - creates deterministic hash (same input = same hash)",
                    example="hash",
                    enum=["hash"],
                ),
                "hash_type": fields.String(
                    required=True,
                    description="Hash algorithm to use",
                    example="sha256",
                    enum=["sha256", "sha512", "md5"],
                    default="sha256",
                ),
            },
        )

        self.operator_encrypt_model = self.api.model(
            "EncryptOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type - AES encryption (reversible with same key)",
                    example="encrypt",
                    enum=["encrypt"],
                ),
                "key": fields.String(
                    required=True,
                    description="Encryption key (16, 24, or 32 characters for 128, 192, or 256-bit encryption)",
                    example="WmZq4t7w!z%C&F)J",
                ),
            },
        )

        self.operator_keep_model = self.api.model(
            "KeepOperator",
            {
                "type": fields.String(
                    required=True,
                    description="Operator type - keeps original text unchanged",
                    example="keep",
                    enum=["keep"],
                )
            },
        )

        # Anonymization request model - NOW WITH ARRAY OF TEXTS
        self.anonymize_request_model = self.api.model(
            "AnonymizeRequest",
            {
                "texts": fields.List(
                    fields.String,
                    required=True,
                    description="Array of strings to analyze and anonymize",
                    example=[
                        "Kontakt Jaan Tamm email jaan@example.com või telefon +372 5555 5555",
                        "Mari Mets elab Tallinnas",
                    ],
                ),
                "language": fields.String(
                    default="xx", description="Language code", example="xx"
                ),
                "anonymizers": fields.Raw(
                    description="""Custom anonymization operators per entity type or DEFAULT for all entities.
                
Available operators:
- replace: {"type": "replace", "new_value": "[REDACTED]"}
- redact: {"type": "redact"} - completely removes PII
- mask: {"type": "mask", "masking_char": "*", "chars_to_mask": 4, "from_end": true}
- hash: {"type": "hash", "hash_type": "sha256"} - deterministic hashing
- encrypt: {"type": "encrypt", "key": "16-32_char_key"} - reversible encryption
- keep: {"type": "keep"} - no anonymization

Use "DEFAULT" to apply operator to all entities, or specify per entity type.
Entity-specific operators override DEFAULT.""",
                    example={
                        "DEFAULT": {"type": "replace", "new_value": "[REDACTED]"},
                        "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
                        "PHONE_NUMBER": {
                            "type": "mask",
                            "masking_char": "X",
                            "chars_to_mask": 4,
                            "from_end": True,
                        },
                    },
                ),
                "entities": fields.List(
                    fields.String,
                    description="Specific entity types to anonymize (if not provided, uses all configured)",
                    example=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
                ),
                "allowlist": fields.List(
                    fields.String,
                    description="Words/phrases to exclude from anonymization (case-insensitive)",
                    example=["Microsoft", "Tallinn"],
                ),
                "denylist": fields.List(
                    fields.String,
                    description="Words/phrases to force anonymization as DENYLIST_MATCH (case-insensitive)",
                    example=["Project Phoenix", "confidential"],
                ),
            },
        )

        # Anonymization item model (for individual PII entities found)
        self.anonymization_item_model = self.api.model(
            "AnonymizationItem",
            {
                "start": fields.Integer(
                    required=True,
                    description="Start position of original PII in text",
                    example=8,
                ),
                "end": fields.Integer(
                    required=True,
                    description="End position of original PII in text",
                    example=17,
                ),
                "entity_type": fields.String(
                    required=True,
                    description="Type of entity that was anonymized",
                    example="PERSON",
                ),
                "text": fields.String(
                    required=True,
                    description="The anonymized/replacement text",
                    example="[PERSON]",
                ),
                "operator": fields.String(
                    required=True,
                    description="Operator used for anonymization",
                    example="replace",
                ),
            },
        )

        # Single text result model
        self.single_anonymize_result_model = self.api.model(
            "SingleAnonymizeResult",
            {
                "text": fields.String(
                    required=True,
                    description="Fully anonymized text with all PII replaced",
                    example="Contact [PERSON] at [EMAIL] or call XXXX",
                ),
                "items": fields.List(
                    fields.Nested(self.anonymization_item_model),
                    description="List of all anonymization operations performed on this text",
                ),
            },
        )

        # Array response model - ARRAY OF RESULTS
        self.anonymize_response_model = self.api.model(
            "AnonymizeResponse",
            {
                "results": fields.List(
                    fields.Nested(self.single_anonymize_result_model),
                    required=True,
                    description="Array of anonymization results, one for each input text",
                )
            },
        )

        # Error model
        self.error_model = self.api.model(
            "Error",
            {
                "error": fields.String(
                    required=True,
                    description="Error message describing what went wrong",
                    example="Missing required field: texts",
                )
            },
        )

    def _setup_error_handlers(self):
        """Setup Flask error handlers"""

        @self.app.errorhandler(HTTPException)
        def handle_http_exception(error):
            logger.error(f"HTTP error: {error}")
            return jsonify(error=str(error)), error.code

        @self.app.errorhandler(Exception)
        def handle_generic_exception(error):
            logger.error(f"Unexpected error: {error}")
            return jsonify(error="Internal server error"), 500

    def _setup_routes(self):
        """Setup Flask-RESTX routes with proper documentation"""

        logger.info("Setting up routes...")

        # Store reference to self for use in route handlers
        server_instance = self

        # Test route to verify Flask is working
        @self.app.route("/test")
        def test_route():
            return jsonify({"status": "test route works"})

        # Health Check Route
        @self.api.route("/health")
        class HealthCheck(Resource):
            @self.api.doc(
                "health_check",
                description="Health check endpoint - verify API is running and get service information",
            )
            @self.api.marshal_with(server_instance.health_model, code=200)
            def get(resource_self):
                """Health check endpoint - verify API is running"""
                logger.info("Health check endpoint called")
                return {
                    "status": "healthy",
                    "service": "Estonian Presidio API",
                    "version": "1.0.0",
                    "supported_languages": server_instance.config.get(
                        "supported_languages", ["xx"]
                    ),
                    "model": "tartuNLP/EstBERT_NER + spaCy",
                }, 200

        logger.info(
            f"Routes registered: {[rule.rule for rule in self.app.url_map.iter_rules()]}"
        )

        # Anonymize Route
        @self.api.route("/anonymize")
        class Anonymize(Resource):
            @self.api.doc(
                "anonymize_text",
                description="""
**Tuvasta ja anonümiseeri isikuandmed tekstides kasutades konfigureeritavaid operaatoreid.**

See endpoint teostab kaks toimingut iga teksti kohta:
1. **Analüüs**: Tuvastab isikuandmed kasutades EstBERT + spaCy mudeleid
2. **Anonümiseerimine**: Muudab tuvastatud isikuandmed vastavalt määratud operaatoritele

**OLULINE: Endpoint võtab vastu MASSIIVI tekstidest ja tagastab MASSIIVI tulemusi.**

**Saadaolevad anonümiseerimise operaatorid:**

| Operaator | Kirjeldus | Pööratav | Näide |
|----------|-------------|----------|---------|
| **replace** | Asenda kohatäitjaga | Ei | `[ISIK]` |
| **redact** | Eemalda täielikult |  Ei | *(tühi)* |
| **mask** | Osaline varjamine |  Ei | `Jaan ****` |
| **hash** | Krüptograafiline räsi |  Ei | `8c9a6b5e...` |
| **encrypt** | AES krüpteerimine |  Jah | `AgQPxF7...` |
| **keep** | Jäta muutmata | — | `Juhan` |

**DEFAULT kasutamine:**
- Määra `"DEFAULT"` operaator kõigile entiteetidele
- Kirjuta üle entiteedi-spetsiifiliste operaatoritega
- Näide: Räsi kõik, välja arvatud e-mailid

**Operaatori näited massiividega:**
```json
// Lihtne asendamine kõigile entiteetidele mitme tekstiga
{
  "texts": [
    "Kontakt Jaan Tamm email jaan@example.com või telefon +372 5555 5555",
    "Mari Mets elab Tallinnas aadressil Liivalaia 2"
  ],
  "anonymizers": {
    "DEFAULT": {"type": "replace", "new_value": "[VARJATUD]"}
  }
}

// Räsi kõik, aga asenda e-mailid
{
  "texts": [
    "Töötaja Mari Mets (mari.mets@firma.ee) isikukood 39012315678",
    "Kontakt Peeter Kukk aadressil peeter@mail.ee"
  ],
  "anonymizers": {
    "DEFAULT": {"type": "hash", "hash_type": "sha256"},
    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"}
  }
}

// Lubatud sõnade ja keelatud sõnade kasutamine
{
  "texts": [
    "Microsoft töötaja Jaan Tamm projektis Phoenix saadab maili aadressil jaan@microsoft.com",
    "Tallinna kontoris töötab Mari Mets projektis Phoenix"
  ],
  "allowlist": ["Microsoft", "Tallinn"],
  "denylist": ["Phoenix"],
  "anonymizers": {
    "DEFAULT": {"type": "replace", "new_value": "[PII]"},
    "PERSON": {"type": "replace", "new_value": "[ISIK]"}
  }
}
```

**Sisendparameetrid:**

- **texts** (nõutud): List tekstidest, mida analüüsida ja anonümiseerida
- **language** (valikuline, vaikimisi "xx"): Keele kood, hetkel sobib xx (mitmekeelne/EstBERT)
- **anonymizers** (valikuline): Anonümiseerimise operaatorid entiteedi tüüpide kaupa või DEFAULT kõigile
- **entities** (valikuline): Konkreetsed entiteedi tüübid, mida anonümiseerida
- **allowlist** (valikuline): Sõnad/fraasid, mida EI anonümiseerita (automaatne käänamine)
- **denylist** (valikuline): Sõnad/fraasid, mida ALATI anonümiseeritakse (automaatne käänamine)

**Väljund:**

Endpoint tagastab `results` massiivi, kus iga element vastab ühele sisendtekstile:
```json
{
  "results": [
    {
      "text": "Kontakt [PERSON] email [EMAIL] või telefon XXXX",
      "items": [
        {"start": 8, "end": 16, "entity_type": "PERSON", "text": "[PERSON]", "operator": "replace"},
        {"start": 23, "end": 30, "entity_type": "EMAIL_ADDRESS", "text": "[EMAIL]", "operator": "replace"}
      ]
    },
    {
      "text": "[PERSON] elab [LOCATION] aadressil Liivalaia 2",
      "items": [...]
    }
  ]
}
```
                """,
            )
            @self.api.expect(server_instance.anonymize_request_model)
            @self.api.marshal_with(server_instance.anonymize_response_model, code=200)
            @self.api.response(400, "Bad Request", server_instance.error_model)
            @self.api.response(
                500, "Internal Server Error", server_instance.error_model
            )
            def post(resource_self):
                """
                Analyze and anonymize multiple texts using EstBERT + spaCy
                """
                try:
                    logger.info("Anonymize endpoint called")
                    # Parse request JSON
                    if not request.is_json:
                        server_instance.api.abort(400, "Request must be JSON")

                    data = request.get_json()

                    if "texts" not in data:
                        server_instance.api.abort(400, "Missing required field: texts")

                    texts = data["texts"]

                    # Validate that texts is a list
                    if not isinstance(texts, list):
                        server_instance.api.abort(400, "Field 'texts' must be an array")

                    if len(texts) == 0:
                        server_instance.api.abort(400, "Field 'texts' cannot be empty")

                    language = data.get("language", "xx")
                    anonymizers = data.get("anonymizers")
                    entities = data.get("entities")
                    allowlist = data.get("allowlist", [])
                    allowlist = synthesize_all(allowlist)
                    denylist = data.get("denylist", [])
                    denylist = synthesize_all(denylist)

                    # Use entities from request or default from config
                    entities_to_detect = entities or server_instance.config.get(
                        "entities_to_detect",
                        [
                            "PERSON",
                            "ORGANIZATION",
                            "LOCATION",
                            "DATE_TIME",
                            "EMAIL_ADDRESS",
                            "PHONE_NUMBER",
                            "URL",
                            "IP_ADDRESS",
                            "GPE",
                        ],
                    )

                    # Setup anonymization operators (same for all texts)
                    operators = {}
                    if anonymizers:
                        # Use custom anonymizers from request
                        for entity_type, config in anonymizers.items():
                            operator_type = config.get("type", "replace")
                            params = {}

                            if operator_type == "replace":
                                params["new_value"] = config.get(
                                    "new_value", f"<{entity_type}>"
                                )
                            elif operator_type == "mask":
                                params["masking_char"] = config.get("masking_char", "*")
                                params["chars_to_mask"] = config.get("chars_to_mask", 4)
                                params["from_end"] = config.get("from_end", True)
                            elif operator_type == "redact":
                                pass  # No parameters needed for redact
                            elif operator_type == "encrypt":
                                params["key"] = config.get("key", "")

                            operators[entity_type] = OperatorConfig(
                                operator_type, params
                            )
                    else:
                        # Use default Estonian anonymizers from config
                        default_anonymizers = server_instance.config.get(
                            "anonymization_config", {}
                        ).get("default_operators", {})
                        for entity_type, replacement in default_anonymizers.items():
                            operators[entity_type] = OperatorConfig(
                                "replace", {"new_value": replacement}
                            )

                        # Add default operator for DENYLIST_MATCH if not configured
                        if "DENYLIST_MATCH" not in operators:
                            operators["DENYLIST_MATCH"] = OperatorConfig(
                                "replace", {"new_value": "[PII]"}
                            )

                    # Process each text in the array
                    results_array = []

                    for idx, text in enumerate(texts):
                        logger.info(f"Processing text {idx + 1}/{len(texts)}")

                        # Analyze text with allowlist/denylist support
                        analyzer_results = analyze_with_lists(
                            analyzer=server_instance.analyzer,
                            text=text,
                            entities=entities_to_detect,
                            language=language,
                            allowlist=allowlist if allowlist else None,
                            denylist=denylist if denylist else None,
                        )

                        # Anonymize the text
                        anonymized_result = server_instance.anonymizer.anonymize(
                            text=text,
                            analyzer_results=analyzer_results,
                            operators=operators,
                        )

                        # Convert items to JSON format
                        items_json = []
                        for item in anonymized_result.items:
                            items_json.append(
                                {
                                    "start": item.start,
                                    "end": item.end,
                                    "entity_type": item.entity_type,
                                    "text": item.text,
                                    "operator": item.operator,
                                }
                            )

                        # Add this text's result to the array
                        results_array.append(
                            {"text": anonymized_result.text, "items": items_json}
                        )

                    logger.info(f"Successfully processed {len(results_array)} texts")

                    return {"results": results_array}, 200

                except Exception as e:
                    error_msg = f"Anonymization failed: {str(e)}"
                    logger.error(error_msg)
                    server_instance.api.abort(500, error_msg)

        # Recognizers Route
        @self.api.route("/recognizers")
        class Recognizers(Resource):
            @self.api.doc(
                "get_recognizers",
                description="""
**Get list of all active recognizers for a language.**

Recognizers are the detection engines that identify different types of PII:
- **EstBERT_NER_Recognizer**: Estonian NER model for PERSON, ORGANIZATION, LOCATION
- **Pattern-based recognizers**: Estonian personal codes, phone numbers, car numbers
- **Built-in recognizers**: Email, URL, IP address, credit cards, etc.

**Example Response:**
```json
{
  "recognizers": [
    "EstBERT_NER_Recognizer",
    "EstonianPersonalCode",
    "EstonianPhoneNumbers",
    "EstonianCarNumber",
    "EmailRecognizer",
    "UrlRecognizer"
  ],
  "language": "xx",
  "count": 6
}
```
                """,
            )
            @self.api.param(
                "language", "Language code (default: xx)", type="string", default="xx"
            )
            def get(resource_self):
                """Get list of available recognizers for a language"""
                try:
                    language = request.args.get("language", "xx")
                    recognizers_list = server_instance.analyzer.get_recognizers(
                        language
                    )
                    recognizer_names = [
                        recognizer.name for recognizer in recognizers_list
                    ]

                    return {
                        "recognizers": recognizer_names,
                        "language": language,
                        "count": len(recognizer_names),
                    }, 200

                except Exception as e:
                    error_msg = f"Failed to get recognizers: {str(e)}"
                    logger.error(error_msg)
                    server_instance.api.abort(500, error_msg)

        # Supported Entities Route
        @self.api.route("/supportedentities")
        class SupportedEntities(Resource):
            @self.api.doc(
                "get_supported_entities",
                description="""
**Get list of all entity types that can be detected.**

Returns all supported PII entity types including:

**Estonian-Specific:**
- EE_PERSONAL_CODE: Estonian personal ID (isikukood)
- CAR_NUMBER: Estonian license plates
- EST_ID_DOC: Estonian document numbers

**Person & Organization:**
- PERSON: Names of individuals
- ORGANIZATION: Company/organization names
- LOCATION: Addresses, cities, places
- GPE: Geopolitical entities

**Contact Information:**
- EMAIL_ADDRESS: Email addresses
- PHONE_NUMBER: Phone numbers (Estonian & international)
- URL: Web addresses

**Financial:**
- IBAN_CODE: Bank account numbers
- CREDIT_CARD: Credit card numbers
- CRYPTO: Cryptocurrency addresses

**Other:**
- DATE_TIME: Dates and times
- IP_ADDRESS: IP addresses

**Example Response:**
```json
{
  "entities": [
    "PERSON", "ORGANIZATION", "LOCATION", 
    "EMAIL_ADDRESS", "PHONE_NUMBER", 
    "EE_PERSONAL_CODE", "CAR_NUMBER",
    "IBAN_CODE", "CREDIT_CARD", "DATE_TIME"
  ],
  "language": "xx",
  "count": 15
}
```
                """,
            )
            @self.api.param(
                "language", "Language code (default: xx)", type="string", default="xx"
            )
            def get(resource_self):
                """Get list of supported entities for a language"""
                try:
                    language = request.args.get("language", "xx")
                    entities_list = server_instance.analyzer.get_supported_entities(
                        language
                    )
                    configured_entities = server_instance.config.get(
                        "entities_to_detect", []
                    )
                    filtered_entities = [
                        e for e in entities_list if e in configured_entities
                    ]
                    return {
                        "entities": filtered_entities,
                        "language": language,
                        "count": len(filtered_entities),
                    }, 200

                except Exception as e:
                    error_msg = f"Failed to get supported entities: {str(e)}"
                    logger.error(error_msg)
                    server_instance.api.abort(500, error_msg)

        # Configuration Route
        @self.api.route("/config")
        class Configuration(Resource):
            @self.api.doc(
                "get_configuration",
                description="""
**Get current API configuration (excluding sensitive information).**

Returns the active configuration including:
- Supported languages
- Detection threshold scores
- Entity types being detected
- Model information (EstBERT, spaCy)
- Custom recognizers

**Use this endpoint to:**
- Verify API configuration
- Check which models are loaded
- See detection thresholds
- List custom recognizers

**Example Response:**
```json
{
  "supported_languages": ["xx"],
  "default_score_threshold": 0.9,
  "entities_to_detect": ["PERSON", "EMAIL_ADDRESS", ...],
  "estbert_model": "tartuNLP/EstBERT_NER",
  "nlp_engine": "spacy",
  "custom_recognizers": [
    {"name": "EstonianPersonalCode", "type": "pattern", ...}
  ]
}
```
                """,
            )
            def get(resource_self):
                """Get current API configuration (excluding sensitive data)"""
                try:
                    safe_config = {
                        "supported_languages": server_instance.config.get(
                            "supported_languages"
                        ),
                        "default_score_threshold": server_instance.config.get(
                            "default_score_threshold"
                        ),
                        "entities_to_detect": server_instance.config.get(
                            "entities_to_detect"
                        ),
                        "estbert_model": server_instance.config.get(
                            "estbert_configuration", {}
                        ).get("model_name"),
                        "nlp_engine": server_instance.config.get(
                            "nlp_configuration", {}
                        ).get("nlp_engine_name"),
                        "custom_recognizers": [
                            {
                                "name": rec.get("name"),
                                "type": rec.get("type"),
                                "supported_entity": rec.get("supported_entity"),
                            }
                            for rec in server_instance.config.get(
                                "custom_recognizers", []
                            )
                        ],
                    }
                    return safe_config, 200

                except Exception as e:
                    error_msg = f"Failed to get configuration: {str(e)}"
                    logger.error(error_msg)
                    server_instance.api.abort(500, error_msg)

        logger.info("All routes setup complete")


# Global server instance
server = None


# Application factory
def create_app(config_path: str = "/app/config/presidio-spacy-estbert.yml") -> Flask:
    """Create and configure the Flask application"""
    global server
    try:
        server = EstonianPresidioFlaskServer(config_path)
        return server.app
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise


# For running directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estonian Presidio API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", DEFAULT_PORT)),
        help="Port to bind to",
    )
    parser.add_argument(
        "--config",
        default="/app/config/presidio-spacy-estbert.yml",
        help="Configuration file path",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Create Flask app
    app = create_app(args.config)

    logger.info(
        f"Starting Estonian Presidio Flask API server on {args.host}:{args.port}"
    )
    logger.info(
        f"Swagger documentation will be available at http://{args.host}:{args.port}/docs/"
    )

    # Run Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )
