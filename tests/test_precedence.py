"""Unit tests for pattern-over-model span ownership.

apply_pattern_precedence() decides who owns text that a YAML pattern and a
model span both claim. The pattern wins the overlap and the model span is cut
back rather than dropped. Runs without a container or a model:

    uv run python tests/test_precedence.py

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from presidio_analyzer import RecognizerResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import presidio_flask_estbert as engine  # noqa: E402

PATTERN = "EstonianCarNumber"
MODEL = "EstBERT_NER_ONNX_Recognizer"
BUILTIN = "IbanRecognizer"


def result(
    text: str, surface: str, entity: str, source: str, score: float = 0.9
) -> RecognizerResult:
    start = text.index(surface)
    return RecognizerResult(
        entity_type=entity,
        start=start,
        end=start + len(surface),
        score=score,
        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: source},
    )


def shape(text: str, results: list[RecognizerResult]) -> list[tuple[str, str]]:
    out = engine.apply_pattern_precedence(results, text)
    return sorted((r.entity_type, text[r.start : r.end]) for r in out)


def run() -> int:
    engine.pattern_recognizer_names = {PATTERN, "WebSite", "Coordinates"}

    t1 = "Auto 123 abc sai trahvi."
    t2 = "Nägin autot 123 abc."
    t3 = "Vaata err.ee/uudised?id=12 lehte."
    t4 = "Koht 59.4370, 24.7536 on siin."
    t5 = "Jaan Tamm helistas."
    t6 = "Konto EE38 2200 2210 2014 5685 on avatud."

    cases: list[tuple[str, str, list[RecognizerResult], list[tuple[str, str]]]] = [
        (
            "a model span swallowing a plate is cut back to what is left",
            t1,
            [
                result(t1, "Auto 123 abc", "ORGANIZATION", MODEL, 0.98),
                result(t1, "123 abc", "CAR_NUMBER", PATTERN),
            ],
            [("CAR_NUMBER", "123 abc"), ("ORGANIZATION", "Auto")],
        ),
        (
            "a model span covering only the plate disappears",
            t2,
            [
                result(t2, "123 abc", "ORGANIZATION", MODEL, 0.98),
                result(t2, "123 abc", "CAR_NUMBER", PATTERN),
            ],
            [("CAR_NUMBER", "123 abc")],
        ),
        (
            "a URL is not an organisation",
            t3,
            [
                result(t3, "err.ee/uudised?id=12", "ORGANIZATION", MODEL, 0.98),
                result(t3, "err.ee/uudised?id=12", "URL", "WebSite"),
            ],
            [("URL", "err.ee/uudised?id=12")],
        ),
        (
            "a same-type overlap is a duplicate, not a mislabel, and is left alone",
            t4,
            [
                result(t4, "59.4370, 24.7536", "LOCATION", MODEL, 0.98),
                result(t4, "59.4370, 24.7536", "LOCATION", "Coordinates"),
            ],
            [("LOCATION", "59.4370, 24.7536"), ("LOCATION", "59.4370, 24.7536")],
        ),
        (
            "a validated built-in is never carved by one of our patterns",
            # IbanRecognizer verifies mod-97 and scores 1.0; our credit-card
            # pattern happily matches four digit groups inside that account
            # number. Carving it split one IBAN into
            # "[PANGAKONTO] [PANGAKAART]".
            t6,
            [
                result(t6, "EE38 2200 2210 2014 5685", "IBAN_CODE", BUILTIN, 1.0),
                result(t6, "2200 2210 2014 5685", "CREDIT_CARD", PATTERN),
            ],
            [
                ("CREDIT_CARD", "2200 2210 2014 5685"),
                ("IBAN_CODE", "EE38 2200 2210 2014 5685"),
            ],
        ),
        (
            "a model span half-covering a built-in span is cut back",
            # "Konto EE38" overlapped the IBAN beside it. A partial overlap
            # makes the anonymizer write both spans, so "EE38" came out twice.
            t6,
            [
                result(t6, "Konto EE38", "ORGANIZATION", MODEL, 0.98),
                result(t6, "EE38 2200 2210 2014 5685", "IBAN_CODE", BUILTIN, 1.0),
            ],
            [("IBAN_CODE", "EE38 2200 2210 2014 5685"), ("ORGANIZATION", "Konto")],
        ),
        (
            "a model span that overlaps nothing is untouched",
            t5,
            [result(t5, "Jaan Tamm", "PERSON", MODEL, 0.98)],
            [("PERSON", "Jaan Tamm")],
        ),
        (
            "with no pattern results the model output passes straight through",
            t1,
            [result(t1, "Auto 123 abc", "ORGANIZATION", MODEL, 0.98)],
            [("ORGANIZATION", "Auto 123 abc")],
        ),
    ]

    failures = 0
    for name, text, results, expected in cases:
        actual = shape(text, results)
        if actual == expected:
            print(f"ok    {name}")
        else:
            failures += 1
            print(
                f"FAIL  {name}\n        expected: {expected}\n        actual:   {actual}"
            )

    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
