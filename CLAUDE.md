# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Flask REST API that detects and anonymizes PII in Estonian text by plugging a transformer NER model
(loaded through ONNX Runtime) plus Estonian regex recognizers into Microsoft Presidio.
Everything is driven by a YAML config file — entities, recognizer patterns, thresholds, and default
anonymization placeholders all live in `config/`, not in code.

## Toolchain & commands

`uv` is the only supported package manager (`uv.lock` is authoritative; Python pinned by `.python-version` to 3.12.10).

```bash
uv sync --frozen                  # install exactly what the lock says (what CI runs)
uv run python app.py --config config/presidio-spacy-estbert.yml   # run locally
uvx ruff check .                  # lint (CI: --output-format=github)
uvx ruff format --check .         # format check; drop --check to format
uv run pyright                    # type check
uv run pre-commit run --all-files # gitleaks + uv-lock hooks
```

`--config` is required when running outside Docker: the default path is the container path
`/app/config/presidio-spacy-estbert.yml`.

Docker: `docker compose up --build -d` (port 8000, Swagger UI at `/docs/`). `./config` is bind-mounted
read-only, so config edits only need a container restart, not a rebuild. The `models_cache` volume holds
the HuggingFace/ONNX cache — first start exports the model to ONNX and is slow (healthcheck allows a
120s start period); wiping that volume makes the next start slow again.

There is **no test suite and no test runner in the dependency set** — do not claim tests pass. CI
(`.github/workflows/`) is limited to: ruff lint, ruff format, pyright, `uv sync --frozen`, and gitleaks,
all on every push/PR to any branch. Verify behaviour changes by exercising the running API with `curl`.

Image publishing: `ci-build-image.yml` fires **only** on pushes to `dev` that modify the repo's env file,
and tags `ghcr.io/<repo>:$RELEASE-$VERSION.$BUILD.$FIX` from the variables in it. A code change alone
publishes nothing — the version bump in that file is the release trigger.

## Architecture

Three modules, imported in this direction: `app.py` → `presidio_flask_estbert.py`, `utils.py`.

**`presidio_flask_estbert.py`** — Presidio wiring. `load_presidio_from_config()` reads the YAML,
builds the `NlpEngine` via `NlpEngineProvider`, constructs `AnalyzerEngine`, drops unwanted built-in
recognizers (`MedicalLicenseRecognizer`), then registers, per supported language:

- `EstBERTRecognizerONNX` — an `EntityRecognizer` wrapping a HF token-classification pipeline over
  `ORTModelForTokenClassification` (`export=True`, CPU provider, thread counts from
  `ONNX_INTER_OP_THREADS` / `ONNX_INTRA_OP_THREADS`). Model labels are mapped to Presidio entities by a
  hardcoded `label_mapping` in the class — note this duplicates `estbert_configuration.entity_mapping`
  in the YAML, and the code path actually used is the one in the class.
- `PatternRecognizer`s generated from the YAML `recognizers:` list.

`analyze_with_lists()` is the single analysis entry point: it calls `analyzer.analyze()`, then filters
results whose exact detected span appears in the allowlist (`apply_allowlist`), then appends
`DENYLIST_MATCH` hits from a word-boundary-checked `DenylistRecognizer`. Denylist matches are added
*after* threshold filtering, so they always survive.

**`utils.py`** — `synthesize_all()` expands allowlist/denylist words into all 28 Estonian case/number
forms using EstNLTK Vabamorf (`synthesize`). `app.py` runs every request's allow/denylist through it so
"Microsoft" also matches "Microsofti", "Microsoftis", etc. This is a per-request cost and grows the
denylist scan linearly.

**`app.py`** — `EstonianPresidioFlaskServer` builds the Flask app in its constructor: engines →
Flask-RESTX models → routes → error handlers. Engines are loaded once at construction (model load is
expensive), so `create_app()` / the class must not be instantiated per request. Routes are declared as
nested classes inside `_setup_routes()` and reach the server through the `server_instance = self`
closure — that pattern is load-bearing, since Flask-RESTX `Resource` classes get no reference to the
server otherwise. Endpoints: `POST /anonymize`, `GET /health`, `GET /recognizers`,
`GET /supportedentities`, `GET /config`, `GET /test`. Served by Flask's dev server (`threaded=True`).

### Config files

`config/presidio-spacy-estbert.yml` is the one the app defaults to: spaCy `xx_ent_wiki_sm`, language
code **`xx`**, threshold 0.83. `config/presidio-stanza-estbert.yml` is an alternative Stanza-based
config using language code **`et`** and lower pattern scores; it is not used by the default path. The
language code is not cosmetic — recognizers are registered per language string, and requests must pass a
`language` matching the loaded config or analysis fails with "No recognizers registered".

Despite the `EstBERT*` naming throughout the code, the model named in the active config is
`buerokrattRIA/xlm-roberta-NER-syntheticGov`.

## Gotchas

- **README is out of date.** It documents a `POST /analyze` endpoint and a singular `"text"` field.
  Neither exists: only `/anonymize` is implemented, and it requires `"texts"` as a non-empty array,
  returning `{"results": [{"text", "items"}, ...]}`. Prefer the code and the Swagger docstrings.
- `docker-compose.yml` sets `MAX_WORKERS` and the README mentions a ThreadPoolExecutor, but no parallel
  processing exists — `/anonymize` loops over `texts` sequentially.
- The `anonymizers` request parser in `/anonymize` only forwards params for `replace`, `mask`, `redact`,
  and `encrypt`. `hash` is documented with a `hash_type`, but that param is silently dropped (Presidio
  falls back to its default), so extend that `if/elif` chain when adding or fixing operator support.
- When no `anonymizers` are supplied, operators come from `anonymization_config.default_operators` in the
  YAML and are all forced to `replace`.
- Ruff lint selects `T20` (no `print` — use the `presidio-flask-api` logger), `ANN` (annotate all
  functions, including `-> None`), `ERA` (no commented-out code), plus `E4/E7/E9/F/B/N/PERF`;
  `fix = false`, line length 88. Pyright runs in `standard` mode and excludes `tests/`.
- `.gitleaks.toml` allowlists exactly one historical commit; keep credentials and env files out of diffs.
