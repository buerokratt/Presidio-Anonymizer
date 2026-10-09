"""Unit tests for score-threshold filtering.

The engine filters nothing; apply_score_thresholds() decides what survives, and
it treats the two kinds of result differently -- a YAML pattern match is never
dropped for its score, a model or built-in result is. Runs without a container
or a model:

    uv run python tests/test_thresholds.py

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from presidio_analyzer import RecognizerResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import presidio_flask_estbert as engine  # noqa: E402


def result(entity: str, score: float, source: str | None) -> RecognizerResult:
    """A RecognizerResult tagged with the recognizer that produced it."""
    metadata = (
        {RecognizerResult.RECOGNIZER_NAME_KEY: source} if source is not None else {}
    )
    return RecognizerResult(
        entity_type=entity,
        start=0,
        end=4,
        score=score,
        recognition_metadata=metadata,
    )


def run() -> int:
    engine.default_score_threshold = 0.83
    engine.entity_score_thresholds = {"ORGANIZATION": 0.45}
    engine.pattern_recognizer_names = {"EstonianPhoneNumbers", "DateTime"}

    text = "0000"
    cases: list[tuple[str, list[RecognizerResult], int]] = [
        (
            "a pattern match below the default threshold survives",
            [result("PHONE_NUMBER", 0.80, "EstonianPhoneNumbers")],
            1,
        ),
        (
            "a pattern match far below the threshold still survives",
            [result("DATE_TIME", 0.10, "DateTime")],
            1,
        ),
        (
            "a model span below the default threshold is dropped",
            [result("PERSON", 0.80, "EstBERT_NER_ONNX_Recognizer")],
            0,
        ),
        (
            "a model span above the default threshold is kept",
            [result("PERSON", 0.90, "EstBERT_NER_ONNX_Recognizer")],
            1,
        ),
        (
            "the per-entity override still applies to model spans",
            [result("ORGANIZATION", 0.50, "EstBERT_NER_ONNX_Recognizer")],
            1,
        ),
        (
            "a built-in below the threshold is still dropped",
            [result("URL", 0.60, "UrlRecognizer")],
            0,
        ),
        (
            "a validated built-in at 1.0 is kept",
            [result("EMAIL_ADDRESS", 1.0, "EmailRecognizer")],
            1,
        ),
        (
            "a result with no recognizer metadata is filtered, not exempted",
            [result("PERSON", 0.10, None)],
            0,
        ),
    ]

    failures = 0
    for name, results, expected in cases:
        kept = engine.apply_score_thresholds(results, text)
        if len(kept) == expected:
            print(f"ok    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: expected {expected} kept, got {len(kept)}")

    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
