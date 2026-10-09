# Estonian Presidio API

A Flask REST API that detects and anonymizes personally identifiable information (PII) in Estonian text.
It plugs a transformer NER model (run through ONNX Runtime) and a set of Estonian regex recognizers into
[Microsoft Presidio](https://microsoft.github.io/presidio/).

Everything is driven by a YAML config file: entities, recognizer patterns, score thresholds and the default
anonymization placeholders all live in `config/`, not in code.

## Features

* **Transformer NER** : `buerokrattRIA/xlm-roberta-NER-syntheticGov`, exported to ONNX on first start and run on CPU,
  for PERSON, ORGANIZATION, LOCATION, GPE and DATE_TIME
* **Estonian patterns** : personal codes (isikukood), document numbers, car plates, phone numbers, Estonian
  date formats, IBANs, coordinates, Ethereum wallets and more, all defined in the YAML config
* **Allowlist / denylist** : exclude false positives or force words to be treated as PII; every entry is
  automatically expanded into all Estonian case forms
* **Batch requests** : one call anonymizes an array of texts
* **Six anonymization operators** : replace, redact, mask, hash, encrypt, keep — per entity or as a `DEFAULT`
* **Swagger UI** : interactive API documentation at `/docs/`
* **Docker** : ready-to-run image and Compose file

## Quick start

### Docker Compose (recommended)

```bash
git clone https://github.com/buerokratt/Presidio-Anonymizer.git
cd Presidio-Anonymizer
docker compose up --build -d
```

The first start downloads the model and exports it to ONNX, which takes a while; the healthcheck allows a
120 s start period. The model is cached in the `models_cache` volume, so later starts are fast. Removing that
volume makes the next start slow again.

`./config` is mounted read-only into the container, so a config change only needs a restart, not a rebuild:

```bash
docker compose restart presidio-api
```

### Local (without Docker)

[uv](https://docs.astral.sh/uv/) is the only supported package manager. Python is pinned to 3.12.10 by
`.python-version`.

```bash
uv sync --frozen
uv run python app.py --config config/presidio-spacy-estbert.yml
```

`--config` is required outside Docker, because the default points at the container path
`/app/config/presidio-spacy-estbert.yml`. Other flags: `--host` (default `0.0.0.0`), `--port` (default `$PORT`
or `8000`) and `--debug`.

### First request

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Minu nimi on Jaan Tamm ja ma elan Tallinnas."],
    "language": "xx"
  }'
```

Swagger UI: <http://localhost:8000/docs/>

## Supported entities

The entities that are actually detected are the ones listed under `entities_to_detect` in the config.
With the default config:

| Entity | Source |
|---|---|
| `PERSON`, `ORGANIZATION`, `LOCATION`, `GPE` | NER model |
| `DATE_TIME` | NER model, Estonian date/time patterns, Presidio built-in |
| `EE_PERSONAL_CODE` | Pattern: Estonian personal code (isikukood) |
| `EST_ID_DOC` | Pattern: Estonian ID card / passport numbers |
| `CAR_NUMBER` | Pattern: Estonian registration plates (`123 ABC`) |
| `PHONE_NUMBER` | Pattern: Estonian and international numbers; Presidio built-in |
| `EMAIL_ADDRESS` | Presidio built-in |
| `URL` | Pattern (file names such as `aruanne.pdf` are excluded); Presidio built-in |
| `IP_ADDRESS` | Pattern: IPv4 and IPv6; Presidio built-in |
| `IBAN_CODE`, `CREDIT_CARD` | Presidio built-in only — validated by mod-97 and Luhn |
| `CRYPTO` | Pattern: Ethereum addresses; Presidio built-in (Bitcoin) |
| `LOCATION` (coordinates) | Pattern: decimal (`59.4370, 24.7536`) and degree/DMS coordinates |
| `DENYLIST_MATCH` | Words from the request's `denylist` |

## API

All endpoints are documented in Swagger at `/docs/`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/anonymize` | Detect and anonymize PII in one or more texts |
| `GET` | `/health` | Liveness check |
| `GET` | `/supportedentities` | Entity types that can be detected |
| `GET` | `/recognizers` | Active recognizers |
| `GET` | `/config` | Active configuration (no secrets) |
| `GET` | `/test` | Returns `{"status": "test route works"}` |

There is no separate `/analyze` endpoint; `/anonymize` does both detection and anonymization.

### `POST /anonymize`

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `texts` | array of strings | yes | Texts to anonymize. Must be a non-empty array. Texts are processed one after another. |
| `language` | string | no, default `xx` | Must match the language code of the loaded config (`xx` for the default config, `et` for the Stanza config). Any other value fails with `No recognizers registered for language`. |
| `anonymizers` | object | no | Operators per entity type, or under `DEFAULT` for all entities. See [Operators](#anonymization-operators). |
| `entities` | array of strings | no | Restrict detection to these entity types. Defaults to `entities_to_detect` from the config. |
| `allowlist` | array of strings | no | Words/phrases that must not be anonymized. |
| `denylist` | array of strings | no | Words/phrases that are always anonymized, as `DENYLIST_MATCH`. |

**Example**

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Kontakt Jaan Tamm email jaan@example.com või telefon +372 5555 5555",
      "Mari Mets elab Tallinnas"
    ],
    "language": "xx",
    "anonymizers": {
      "DEFAULT": {"type": "replace", "new_value": "[VARJATUD]"},
      "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"}
    }
  }'
```

**Response**

One result per input text, in the same order (detections shown are illustrative; what is found depends on
the model). `items` lists every replacement; `start`/`end` are offsets
into the *anonymized* text and `text` is the replacement value.

```json
{
  "results": [
    {
      "text": "Kontakt [VARJATUD] email [EMAIL] või telefon [VARJATUD]",
      "items": [
        {"start": 45, "end": 55, "entity_type": "PHONE_NUMBER", "text": "[VARJATUD]", "operator": "replace"},
        {"start": 25, "end": 32, "entity_type": "EMAIL_ADDRESS", "text": "[EMAIL]", "operator": "replace"},
        {"start": 8, "end": 18, "entity_type": "PERSON", "text": "[VARJATUD]", "operator": "replace"}
      ]
    },
    {
      "text": "[VARJATUD] elab [VARJATUD]",
      "items": ["..."]
    }
  ]
}
```

**Errors**

Errors are returned as `{"error": "..."}`. Malformed requests (not JSON, `texts` missing, not an array or
empty, unknown operator type, unsupported `hash_type`) return `400`; failures during analysis return `500`.

### Anonymization operators

Operators are given per entity type in `anonymizers`. `DEFAULT` applies to every entity without its own entry.

| Type | Parameters | Effect | Reversible |
|---|---|---|---|
| `replace` | `new_value` (default `<ENTITY_TYPE>`) | Substitute a placeholder | no |
| `redact` | — | Remove the text entirely | no |
| `mask` | `masking_char` (default `*`), `chars_to_mask` (default `4`), `from_end` (default `true`) | Mask characters | no |
| `hash` | `hash_type`: `sha256` (default), `sha512` or `md5` | Deterministic hash; the same input gives the same hash | no |
| `encrypt` | `key`: 16, 24 or 32 characters (AES-128/192/256) | AES encryption | yes, with the same key |
| `keep` | — | Leave the text unchanged | — |

Any other `type` is rejected with `400`.

```json
{
  "anonymizers": {
    "DEFAULT": {"type": "hash", "hash_type": "sha256"},
    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
    "PHONE_NUMBER": {"type": "mask", "masking_char": "X", "chars_to_mask": 4, "from_end": true},
    "LOCATION": {"type": "keep"}
  }
}
```

**When `anonymizers` is omitted**, every entity is replaced with the Estonian placeholder configured under
`anonymization_config.default_operators` (`[ISIK]`, `[ASUKOHT]`, `[ISIKUKOOD]`, …), and `DENYLIST_MATCH` with
`[PII]` unless the config sets one.

**When `anonymizers` is given**, the config placeholders are not used at all. Entities with neither their own
entry nor a `DEFAULT` are replaced with `<ENTITY_TYPE>`, e.g. `<PERSON>`.

### Allowlist and denylist

Both lists are case-insensitive, and each entry is expanded with EstNLTK Vabamorf into all 28 Estonian case
forms (14 cases × singular/plural) before matching, so you only need to give the base form:
`"Tallinn"` also covers `Tallinna`, `Tallinnas`, `Tallinnast`, …; multi-word phrases are expanded too.
The expansion runs on every request, so very long lists make requests slower.

* **Allowlist** — a detection whose text is entirely allowlisted is dropped. When an allowlisted term sits at
  the edge of a wider detection, the detection is trimmed rather than kept whole, so
  `"Phoenix Tallinnas"` with `Tallinn` allowlisted still anonymizes `Phoenix`.
* **Denylist** — whole-word matches are reported as `DENYLIST_MATCH` and always survive: they are not subject
  to score thresholds and win any overlap with other detections. An overlapping detection is cut back around
  the denylisted word rather than discarded, so a name next to it stays protected.

```bash
curl -X POST http://localhost:8000/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Microsofti töötaja Jaan Tamm töötab Tallinnas projektiga Phoenix."],
    "language": "xx",
    "allowlist": ["Microsoft", "Tallinn"],
    "denylist": ["Phoenix"],
    "anonymizers": {
      "PERSON": {"type": "replace", "new_value": "[ISIK]"},
      "DENYLIST_MATCH": {"type": "replace", "new_value": "[KONFIDENTSIAALNE]"}
    }
  }'
```

Intended result: `Microsofti töötaja [ISIK] töötab Tallinnas projektiga [KONFIDENTSIAALNE].`

### `GET /health`

```json
{
  "status": "healthy",
  "service": "Estonian Presidio API",
  "version": "1.0.0",
  "supported_languages": ["xx"],
  "model": "buerokrattRIA/xlm-roberta-NER-syntheticGov + spacy"
}
```

`model` is built from the loaded config, so it names the model actually in use.

### `GET /supportedentities?language=xx`

Entity types the analyzer supports for the language, filtered to those in `entities_to_detect`.

```json
{"entities": ["PERSON", "ORGANIZATION", "LOCATION", "EMAIL_ADDRESS", "..."], "language": "xx", "count": 15}
```

### `GET /recognizers?language=xx`

Names of the registered recognizers: `EstBERT_NER_ONNX_Recognizer`, the pattern recognizers from the config
(`EstonianPersonalCode`, `EstonianPhoneNumbers`, …) and Presidio's built-ins (`EmailRecognizer`,
`IbanRecognizer`, …).

```json
{"recognizers": ["EstBERT_NER_ONNX_Recognizer", "EstonianPersonalCode", "..."], "language": "xx", "count": 20}
```

### `GET /config`

```json
{
  "supported_languages": ["xx"],
  "default_score_threshold": 0.83,
  "entities_to_detect": ["PERSON", "ORGANIZATION", "..."],
  "estbert_model": "buerokrattRIA/xlm-roberta-NER-syntheticGov",
  "nlp_engine": "spacy",
  "custom_recognizers": [
    {
      "name": "EstonianPersonalCode",
      "supported_entity": "EE_PERSONAL_CODE",
      "supported_language": "xx",
      "patterns": ["isikukood_pattern"]
    }
  ]
}
```

`custom_recognizers` lists the recognizers defined under `recognizers:` in the config, with their pattern
names. Presidio's own built-ins are not listed here; `/recognizers` names everything that is registered.

## Configuration

Two configs ship in `config/`:

| File | NLP engine | Language code | Notes |
|---|---|---|---|
| `presidio-spacy-estbert.yml` | spaCy `xx_ent_wiki_sm` | `xx` | **Default** — used by Docker and the examples above |
| `presidio-stanza-estbert.yml` | Stanza | `et` | Alternative, with lower pattern scores |

The language code matters: recognizers are registered per language, and requests must send the same
`language` as the loaded config.

### Main sections

```yaml
supported_languages:
  - xx

# Detections from the NER model and Presidio built-ins below this score are dropped.
# Pattern recognizers from this file are never filtered by score (see below).
default_score_threshold: 0.83

# Per-entity overrides of default_score_threshold. Both of these were chosen by
# measurement: agency names score as low as 0.50, and 0.60 is the best GPE cut
# over the chat-NER test set.
entity_score_thresholds:
  ORGANIZATION: 0.45
  GPE: 0.60

# NLP engine used for tokenisation (spaCy's own NER is not used for detection)
nlp_configuration:
  nlp_engine_name: spacy
  models:
    - lang_code: xx
      model_name: xx_ent_wiki_sm

# Transformer NER model, loaded from Hugging Face and exported to ONNX
estbert_configuration:
  model_name: "buerokrattRIA/xlm-roberta-NER-syntheticGov"
  supported_language: xx

# Entities analysed when a request does not pass `entities`
entities_to_detect:
  - PERSON
  - EE_PERSONAL_CODE
  # ...

# Pattern recognizers. `validator:` is optional and names a check from
# VALIDATORS in presidio_flask_estbert.py; a match that fails it is discarded
# rather than reported. Currently only `ee_personal_code` exists, and no
# shipped config switches it on - see Validators below.
recognizers:
  - name: EstonianPersonalCode
    supported_language: xx
    supported_entity: EE_PERSONAL_CODE
    # validator: ee_personal_code
    patterns:
      - name: isikukood_pattern
        regex: "..."
        score: 0.95

# Placeholders used when a request sends no `anonymizers`
anonymization_config:
  default_operators:
    PERSON: "[ISIK]"
    LOCATION: "[ASUKOHT]"
    EE_PERSONAL_CODE: "[ISIKUKOOD]"
    # ...
```

### Score thresholds

The NER model and the Presidio built-in recognizers produce real confidence scores, so their results are
filtered against `entity_score_thresholds`, falling back to `default_score_threshold`. A regex pattern either
matches or it does not, so results from the `recognizers:` patterns are **never** dropped for their score —
if a pattern over-matches, fix the regex.

### Overlapping detections

Three rules decide who owns a piece of text, applied in this order:

* A **pattern beats the NER model**. A pattern either matched or it did not; a model span is a guess, and
  the model scores confidently enough to win on score alone. Where the two overlap and disagree on the
  entity type, the model's span is cut back rather than dropped — a model span over `Auto 123 ABC` keeps
  `Auto` once the plate is carved out of it, and the plate is reported as `CAR_NUMBER`, not
  `ORGANIZATION`. Presidio's own built-ins count as patterns here: they are deterministic too, and the
  validated ones are better evidence than any regex in the config.
* **Model spans separated only by spaces are merged** before the score filter, taking the higher score, so
  a name the model labelled one word at a time is not half dropped. `New Yorki` used to come back as
  `[GPE] Yorki`.
* A **denylist match wins everything** — see [Allowlist and denylist](#allowlist-and-denylist).

Scores still break ties between detections that none of these cover: Presidio keeps the higher-scored one.

### Model label mapping

The NER model's labels are mapped to Presidio entities by `estbert_configuration.entity_mapping` in the
config. When that key is absent the recognizer falls back to its own table: `PER→PERSON`,
`ORG→ORGANIZATION`, `LOC→LOCATION`, `GPE→GPE`, `DATE`/`TIME→DATE_TIME`. Both shipped configs declare a
mapping that matches the fallback.

### Validators

A pattern says what text looks like; a validator says whether it is real. A recognizer can name one with
`validator:`, and Presidio then scores a verified match 1.0 and discards one that fails, so a shape-only
match is not reported at the pattern's nominal confidence.

| Validator | Checks |
|---|---|
| `ee_personal_code` | The isikukood mod-11 check digit, and a first digit in 1-8 |

`IBAN_CODE` and `CREDIT_CARD` are not defined as patterns in the config at all: Presidio's own recognizers
own them and verify mod-97 and Luhn respectively. A card-shaped number that fails Luhn is therefore not
anonymized, which is deliberate.

**`ee_personal_code` is not enabled in either shipped config.** Turning it on rejects personal codes with an
invalid check digit — which is correct, but invented test data usually has one, so enable it together with
test fixtures that use checksum-valid codes.

## Docker deployment

### Environment variables

Read by the application:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the server listens on |
| `ONNX_INTER_OP_THREADS` | `2` | ONNX Runtime inter-op threads |
| `ONNX_INTRA_OP_THREADS` | `2` | ONNX Runtime intra-op threads per inference |
| `HF_HOME`, `TRANSFORMERS_CACHE` | `/app/models` in Compose | Hugging Face / ONNX model cache location |

`docker-compose.yml` and the Dockerfile also set `FLASK_ENV`, `LOG_LEVEL`, `HOST` and `MAX_WORKERS`, but the
application does not read them: logging is fixed at `INFO`, the host is set with `--host`, and texts in a
request are processed sequentially.

### Resources and health

The Compose file limits the container to 4 GB memory and 4 CPUs (reserving 2 GB / 2 CPUs). No GPU is needed.

The Compose healthcheck calls `GET /health` every 30 s (10 s timeout, 5 retries, 120 s start period).

### Image publishing

`ci-build-image.yml` builds and pushes `ghcr.io/<repo>` only on pushes to `dev` that change the repository's
env file, tagging the image from the release/version variables in it. Changing code alone publishes nothing;
bump the version in that file to release.

## Development

```bash
uvx ruff check .                  # lint
uvx ruff format --check .         # format check (drop --check to format)
uv run pyright                    # type check
uv run pre-commit run --all-files # gitleaks + uv-lock hooks
```

CI runs ruff lint, ruff format, pyright, `uv sync --frozen` and gitleaks on every push and pull request.

### Test scripts

The scripts in `tests/` are standalone and run with `uv run python`; there is no pytest suite and CI does
not run them.

| Script | Needs a running API | What it checks |
|---|---|---|
| `tests/test_patterns.py [--config …]` | no | YAML regex patterns against expected matches |
| `tests/test_thresholds.py` | no | Score-threshold filtering: patterns exempt, model and built-ins not |
| `tests/test_spans.py` | no | Merging of model spans the NER split across words |
| `tests/test_precedence.py` | no | Who owns text a pattern and a model span both claim |
| `tests/test_validators.py` | no | The isikukood check digit |
| `tests/test_gov_chats.py [--url …]` | yes | End-to-end cases from government chat texts |
| `tests/eval_conll.py <file.conll>` | yes | Entity-level precision/recall/F1 (PER, ORG, LOC, GPE) against a CoNLL NER test set |

Each exits with `0` when everything passes.

### Code layout

* `app.py` — Flask/Flask-RESTX server, request handling and Swagger models
* `presidio_flask_estbert.py` — builds the Presidio analyzer from the YAML (NER recognizer, pattern
  recognizers, thresholds) and implements allowlist/denylist handling
* `utils.py` — Estonian case-form synthesis for allow/denylists (EstNLTK Vabamorf)
