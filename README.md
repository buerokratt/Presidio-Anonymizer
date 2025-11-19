# Estonian Presidio API

A privacy-focused Flask API server that combines Microsoft Presidio with the Estonian EstBERT NER model to detect and anonymize personally identifiable information (PII) in Estonian text.

## Features

* **Estonian NER Support** : Integrates `tartuNLP/EstBERT_NER_v2` for Estonian named entity recognition
* **Multilingual** : Supports Estonian via sPaCy
* **Custom Estonian Patterns** : Recognizes Estonian-specific entities like personal codes (isikukood), car numbers, and phone numbers
* **Batch Processing** : Process multiple texts in a single request for efficient anonymization
* **Allowlist/Denylist Support** : Fine-tune detection by excluding false positives or forcing specific words to be detected as PII
* **REST API** : Easy-to-use HTTP endpoints for text analysis and anonymization
* **Docker Support** : Ready-to-deploy containerized application
* **Configurable** : YAML-based configuration for entities, patterns, and anonymization rules
* **Swagger Documentation** : Interactive API documentation available at `/docs/`

## Supported Entities

### Estonian-Specific

* **EE_PERSONAL_CODE** : Estonian personal identification codes (isikukood)
* **CAR_NUMBER** : Estonian car registration numbers
* **PHONE_NUMBER** : Estonian and international phone numbers
* **EST_ID_DOC** : Estonian document numbers

### General Entities (via EstBERT + Presidio)

* **PERSON** : Personal names
* **ORGANIZATION** : Company and organization names
* **LOCATION** : Addresses and locations
* **GPE** : Geopolitical entities
* **EMAIL_ADDRESS** : Email addresses
* **URL** : Website URLs
* **IP_ADDRESS** : IP addresses
* **IBAN_CODE** : IBAN bank account numbers
* **DATE_TIME** : Dates and times
* **CRYPTO** : Cryptocurrency addresses
* **CREDIT_CARD** : Credit card numbers

### Special Entities

* **DENYLIST_MATCH** : Words from the denylist that are forced to be detected as PII

## Installation

### Using Docker Compose (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/buerokratt/Presidio-Anonymizer.git
cd Presidio-Anonymizer
```

2. Start the service:

```bash
docker-compose up --build -d
```

3. Test the API:

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Minu nimi on Jaan Tamm ja ma elan Tallinnas."],
    "language": "xx"
}'
```

4. Access Swagger documentation:

```
http://localhost:8000/docs/
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns API status and configuration information.

**Response:**

```json
{
  "status": "healthy",
  "service": "Estonian Presidio API",
  "version": "1.0.0",
  "supported_languages": ["xx"],
  "model": "tartuNLP/EstBERT_NER + spaCy"
}
```

### Anonymize Text

```http
POST /anonymize
```

Analyzes and replaces PII entities with anonymized placeholders. **Accepts an array of texts** and returns an array of results.

**Batch Processing:**

- Process multiple texts in a single request
- Same configuration (anonymizers, allowlist, denylist) applies to all texts
- Each text is processed independently
- Results maintain the same order as input texts

**Request Body:**

```json
{
  "texts": [
    "Minu nimi on Jaan Tamm ja ma elan Tallinnas. Projekti nimi on Liblikas.",
    "Mari Mets töötab Tartu Ülikoolis."
  ],
  "language": "xx",
  "anonymizers": {
    "PERSON": {"type": "replace", "new_value": "[ISIK]"},
    "LOCATION": {"type": "mask", "masking_char": "*", "chars_to_mask": 4},
    "DENYLIST_MATCH": {"type": "hash", "hash_type": "sha256"}
  },
  "entities": ["PERSON", "LOCATION"],
  "allowlist": ["Tallinn"],
  "denylist": ["Liblikas"]
}
```

**Using DEFAULT Operator:**

Apply the same anonymization to all detected entities:

```json
{
  "texts": ["Mu nimi on Mart Kask ja email on mart@example.com"],
  "language": "xx",
  "anonymizers": {
    "DEFAULT": {
      "type": "replace",
      "new_value": "[XXX]"
    }
  }
}
```

Or use `DEFAULT` with entity-specific overrides:

```json
{
  "texts": ["Võta ühendust Peeter Kukk, email peeter@example.com või helistades +372 5555 5555"],
  "language": "xx",
  "anonymizers": {
    "DEFAULT": {"type": "hash", "hash_type": "sha256"},
    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
    "PHONE_NUMBER": {"type": "mask", "masking_char": "X", "chars_to_mask": 4}
  }
}
```

**Parameters:**

* `texts` (required, array of strings): **Array of texts to anonymize** - each text is processed independently
* `language` (optional, string, default: "xx"): Language code
* `anonymizers` (optional, object): Custom anonymization operators for each entity type
  * `type`: Anonymization method - "replace", "mask", "redact", or "encrypt"
  * `new_value`: Replacement text (for "replace" type)
  * `masking_char`: Character to use for masking (for "mask" type)
  * `chars_to_mask`: Number of characters to mask (for "mask" type)
  * `from_end`: Mask from end of string (for "mask" type)
  * `key`: Encryption key (for "encrypt" type)
* `entities` (optional, array): Specific entity types to anonymize
* `allowlist` (optional, array): Words to exclude from anonymization
* `denylist` (optional, array): Words to force anonymization

**Response:**

Returns an array of results, one for each input text:

```json
{
  "results": [
    {
      "text": "Minu nimi on [ISIK] ja ma elan Tallinnas. Projekti nimi on [KONFIDENTSIAALNE].",
      "items": [
        {
          "end": 22,
          "entity_type": "PERSON",
          "operator": "replace",
          "start": 13,
          "text": "[ISIK]"
        },
        {
          "end": 76,
          "entity_type": "DENYLIST_MATCH",
          "operator": "replace",
          "start": 67,
          "text": "[KONFIDENTSIAALNE]"
        }
      ]
    },
    {
      "text": "[ISIK] töötab Tartu Ülikoolis.",
      "items": [
        {
          "end": 10,
          "entity_type": "PERSON",
          "operator": "replace",
          "start": 0,
          "text": "[ISIK]"
        }
      ]
    }
  ]
}
```

**Anonymization Operators:**

Presidio supports multiple anonymization methods. You can specify operators per entity type or use `DEFAULT` to apply the same operator to all entities.

1. **Replace** : Substitute entity with a placeholder

```json
   {
     "anonymizers": {
       "PERSON": {"type": "replace", "new_value": "[ISIK]"},
       "DEFAULT": {"type": "replace", "new_value": "[VARJATUD]"}
     }
   }
```

2. **Redact** : Completely remove the PII text

```json
   {
     "anonymizers": {
       "DEFAULT": {"type": "redact"}
     }
   }
```

* Removes the entity entirely from the text

3. **Mask** : Replace characters with masking character

```json
   {
     "anonymizers": {
       "PERSON": {
         "type": "mask",
         "masking_char": "*",
         "chars_to_mask": 4,
         "from_end": true
       }
     }
   }
```

* `masking_char`: Character to use for masking (default: `*`)
* `chars_to_mask`: Number of characters to mask
* `from_end`: Whether to mask from the end (default: `true`)

4. **Hash** : Replace with cryptographic hash

```json
   {
     "anonymizers": {
       "DEFAULT": {
         "type": "hash",
         "hash_type": "sha256"
       }
     }
   }
```

* `hash_type`: Hash algorithm - `sha256`, `sha512`, or `md5`
* Produces consistent hash for same input

5. **Encrypt** : Replace with AES encrypted value (reversible)

```json
   {
     "anonymizers": {
       "PERSON": {
         "type": "encrypt",
         "key": "WmZq4t7w!z%C&F)J"
       }
     }
   }
```

* `key`: 128-bit, 192-bit, or 256-bit encryption key
* Allows decryption with the same key

6. **Keep** : Retain original value (no anonymization)

```json
   {
     "anonymizers": {
       "LOCATION": {"type": "keep"}
     }
   }
```

* Useful when combined with `DEFAULT` to preserve specific entities

### Get Supported Entities

```http
GET /supportedentities?language=xx
```

Returns list of all entity types that can be detected for a given language.

**Parameters:**

* `language` (optional, string, default: "et"): Language code

**Response:**

```json
{
  "entities": ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", ...],
  "language": "xx",
  "count": 15
}
```

### Get Available Recognizers

```http
GET /recognizers?language=xx
```

Returns list of all active recognizers for a given language.

**Parameters:**

* `language` (optional, string, default: "xx"): Language code

**Response:**

```json
{
  "recognizers": ["EstBERT_NER_Recognizer", "EstonianPersonalCode", "EstonianPhoneNumbers", ...],
  "language": "xx",
  "count": 18
}
```

### Get Configuration

```http
GET /config
```

Returns current API configuration (excluding sensitive data).

**Response:**

```json
{
  "supported_languages": ["xx"],
  "default_score_threshold": 0.83,
  "entities_to_detect": ["PERSON", "ORGANIZATION", ...],
  "estbert_model": "tartuNLP/EstBERT_NER",
  "nlp_engine": "spacy",
  "custom_recognizers": [...]
}
```

## Allowlist and Denylist Usage

### Allowlist (Excluding False Positives)

Use the allowlist to prevent common words or domain-specific terms from being flagged as PII:

**Example: Company and product names**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Microsoft Azure teenust kasutab Eesti Energia oma pilvelahenduste jaoks."],
    "allowlist": ["Microsoft", "Azure", "Eesti Energia"]
  }'
```

**Use cases:**

* Company names that shouldn't be anonymized
* Common place names (e.g., "Tallinn", "Tartu")
* Product names or brands
* Technical terms that trigger false positives

### Denylist (Forcing Detection)

Use the denylist to ensure specific sensitive terms are always detected as PII:

**Example: Project codenames and internal terms**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Projekt Öökull on salastatud. Võta ühendust Kaarel Kasega detailide osas."],
    "denylist": ["Öökull", "salastatud"],
    "anonymizers": {
      "DENYLIST_MATCH": {"type": "replace", "new_value": "[VARJATUD]"}
    }
  }'
```

**Use cases:**

* Internal project codenames
* Classified terminology
* Domain-specific sensitive information
* Custom identifiers not recognized by standard models

### Combined Usage

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Jaan Tamm Microsoft Eesti esindusest töötab projektil Phönix Tallinnas."],
    "allowlist": ["Microsoft Eesti", "Tallinn"],
    "denylist": ["Phönix"],
    "entities": ["PERSON", "ORGANIZATION", "LOCATION"]
  }'
```

**Result:**

* "Jaan Tamm" → anonymized (detected as PERSON)
* "Microsoft Eesti" → kept (in allowlist)
* "Phönix" → anonymized (in denylist)
* "Tallinn" → kept (in allowlist)

## Configuration

The API is configured via `config/presidio-stanza-estbert.yml`. Key sections include:

### NLP Engine Configuration

```yaml
nlp_configuration:
  nlp_engine_name: stanza
  models:
    - lang_code: xx
      model_name: xx_ent_wiki_sm

```

### EstBERT Model Configuration

```yaml
estbert_configuration:
  model_name: "tartuNLP/EstBERT_NER_v2"
  supported_language: xx
  entity_mapping:
    LOC: LOCATION
    ORG: ORGANIZATION
    PER: PERSON
    TIME: DATE_TIME
    DATE: DATE_TIME
    GPE: GPE

```

### Custom Recognizers

```yaml
recognizers:
  - name: EstonianPersonalCode
    supported_language: xx
    supported_entity: EE_PERSONAL_CODE
    patterns:
      - name: isikukood_pattern
        regex: "(?:^|(?<=[.|,|;|:|\\s|!|?]))[1-6][0-9]{2}(01|02|...)[0-9]{3}[0-9](?=[.|,|;|:|\\s|!|?]|$)"
        score: 0.95
```

### Anonymization Rules

```yaml
anonymization_config:
  default_operators:
    PERSON: "[ISIK]"
    LOCATION: "[ASUKOHT]"
    EE_PERSONAL_CODE: "[ISIKUKOOD]"
    DENYLIST_MATCH: "[PII]"
```

## Docker Deployment

### Environment Variables

#### Application Configuration

* **`FLASK_ENV`** : Flask environment mode
* Values: `production`, `development`
* Default: `production`
* Description: Controls Flask's debug mode and error verbosity. Use `development` for debugging.
* **`PORT`** : Port number for the API server
* Type: Integer
* Default: `8000`
* Description: The port on which the Flask application listens for HTTP requests.
* **`HOST`** : Host address to bind to
* Type: String
* Default: `0.0.0.0`
* Description: Network interface to bind to. `0.0.0.0` allows connections from any network interface.

#### Logging Configuration

* **`LOG_LEVEL`** : Application logging verbosity
* Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
* Default: `INFO`
* Description: Controls the level of detail in application logs. Use `DEBUG` for troubleshooting.

#### Resource Limits (Docker Compose)

* **`MEMORY_LIMIT`** : Maximum memory allocation
* Type: String (e.g., "4g", "2048m")
* Default: `4g`
* Description: Limits container memory usage. EstBERT requires ~2-3GB minimum.
* **`CPU_LIMIT`** : CPU cores allocation
* Type: String (e.g., "2", "1.5")
* Default: `2`
* Description: Number of CPU cores the container can use.

### Example docker-compose.yml

```yaml
version: '3.8'

services:
  presidio-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - PORT=8000
      - HOST=0.0.0.0
      - LOG_LEVEL=INFO
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
    volumes:
      - ./config:/app/config
      - transformers_cache:/root/.cache/huggingface
    deploy:
      resources:
        limits:
          memory: 4g
          cpus: '2'
        reservations:
          memory: 2g
          cpus: '1'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

volumes:
  stanza_cache:
  transformers_cache:
```

### Resource Requirements

* **Memory** :
* Minimum: 2GB RAM
* Recommended: 4GB RAM (for EstBERT model loading and processing)
* EstBERT model: ~500MB
* Application runtime: ~1-2GB
* **CPU** :
* Minimum: 1 core
* Recommended: 2 cores for better performance
* Note: No GPU required, but will improve performance if available
* **Storage** :
* Application: ~500MB
* Models cache: ~1.5GB
* Logs and temporary files: ~500MB
* **Total recommended** : ~3GB

### Health Checks

The container includes built-in health checks that verify API availability:

* **Endpoint** : `GET /health`
* **Interval** : 30 seconds (how often to check)
* **Timeout** : 10 seconds (how long to wait for response)
* **Retries** : 5 (consecutive failures before marking unhealthy)
* **Start Period** : 60 seconds (grace period during startup)

## Examples

### Basic Anonymization

**Estonian Personal Code Detection:**

**Simple Anonymization with Replace:**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Jaan Tamm (isikukood: 38001085718) töötab AS Eesti Firma juures Tallinnas."],
    "language": "xx"
  }'
```

**Batch Processing Multiple Texts:**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Jaan Tamm (isikukood: 38001085718) töötab AS Eesti Firma juures Tallinnas.",
      "Mari Mets saadab emaili mari.mets@example.ee",
      "Kontakttelefon: +372 5555 5555"
    ],
    "language": "xx"
  }'
```

### Anonymization Operators Examples

**Using DEFAULT operator (applies to all entities):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Mu nimi on Mart Kask ja email on mart.kask@example.ee"],
    "language": "xx",
    "anonymizers": {
      "DEFAULT": {
        "type": "replace",
        "new_value": "[XXX]"
      }
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Mu nimi on [XXX] ja email on [XXX]",
      "items": [...]
    }
  ]
}
```

**Hash operator (consistent hashing):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Mu nimi on Mart Kask ja elan Tartus"],
    "language": "xx",
    "anonymizers": {
      "DEFAULT": {
        "type": "hash",
        "hash_type": "sha256"
      }
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Mu nimi on 8c9a6b5e3d2f1a4b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b ja elan 3f8a2c9d1e4b7a5c8d0f2e1b3a6c9d7e4f8a2b5c1d3e6f9a0b2c5d8e1f4a7b",
      "items": [...]
    }
  ]
}
```

**Mask operator (partial masking):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Võta ühendust Kalle Kallasega telefonil +372 5123 4567"],
    "language": "xx",
    "anonymizers": {
      "PERSON": {
        "type": "mask",
        "masking_char": "*",
        "chars_to_mask": 6,
        "from_end": true
      },
      "PHONE_NUMBER": {
        "type": "mask",
        "masking_char": "X",
        "chars_to_mask": 4,
        "from_end": true
      }
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Võta ühendust Kalle K****** telefonil +372 5123 XXXX",
      "items": [...]
    }
  ]
}
```

**Redact operator (complete removal):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Võta ühendust Kalle Kask aadressil kalle@example.ee"],
    "language": "xx",
    "anonymizers": {
      "PERSON": {"type": "redact"},
      "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL EEMALDATUD]"}
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Võta ühendust  aadressil [EMAIL EEMALDATUD]",
      "items": [...]
    }
  ]
}
```

**Encrypt operator (reversible encryption):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Töötaja isikukood: 38001085718"],
    "language": "xx",
    "anonymizers": {
      "EE_PERSONAL_CODE": {
        "type": "encrypt",
        "key": "WmZq4t7w!z%C&F)J"
      }
    }
  }'
```

**Mix of operators (entity-specific):**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Võta ühendust Mari Maasikas aadressil mari.maasikas@example.ee või helistades +372 5555 5555"],
    "language": "xx",
    "anonymizers": {
      "PERSON": {"type": "replace", "new_value": "[NIMI]"},
      "EMAIL_ADDRESS": {"type": "hash", "hash_type": "sha256"},
      "PHONE_NUMBER": {"type": "mask", "masking_char": "X", "chars_to_mask": 4, "from_end": true}
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "Võta ühendust [NIMI] aadressil a3b2c1d4e5f6... või helistades +372 5555 XXXX",
      "items": [...]
    }
  ]
}
```

**Using DEFAULT with entity overrides:**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Jaan Tamm elab Tallinnas. Email: jaan.tamm@example.ee, Telefon: +372 5555 5555"],
    "language": "xx",
    "anonymizers": {
      "DEFAULT": {"type": "hash", "hash_type": "sha256"},
      "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL KAITSTUD]"},
      "LOCATION": {"type": "keep"}
    }
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "a1b2c3d4e5f6... elab Tallinnas. Email: [EMAIL KAITSTUD], Telefon: f9e8d7c6b5a4...",
      "items": [...]
    }
  ]
}
```

### Allowlist and Denylist Examples

**Using Allowlist to Preserve Important Terms:**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Mari Maasikas Äriregistri esindusest töötab Tallinnas."],
    "language": "xx",
    "allowlist": ["Äriregister", "Tallinn"]
  }'
```

### Using Denylist for Sensitive Project Names

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Projekt Öökull on salastatud. Võta ühendust Kaarel Kasega detailide osas."],
    "language": "xx",
    "denylist": ["Öökull", "salastatud"]
  }'
```

### Complex Example with Both Lists

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Võta ühendust Kalle Kask Microsoft Eesti esindusest projektiga Phönix seoses. Kohtumine Tallinnas homme.",
      "Mari Mets töötab samuti projektil Phönix meie Tartu kontorist."
    ],
    "language": "xx",
    "allowlist": ["Microsoft Eesti", "Tallinn", "Tartu"],
    "denylist": ["Phönix"],
    "anonymizers": {
      "PERSON": {"type": "replace", "new_value": "[NIMI]"},
      "DENYLIST_MATCH": {"type": "replace", "new_value": "[SALASTATUD]"}
    }
  }'
```

**Result:**

```json
{
  "results": [
    {
      "text": "Võta ühendust [NIMI] Microsoft Eesti esindusest projektiga [SALASTATUD] seoses. Kohtumine Tallinnas homme.",
      "items": [...]
    },
    {
      "text": "[NIMI] töötab samuti projektil [SALASTATUD] meie Tartu kontorist.",
      "items": [...]
    }
  ]
}
```
