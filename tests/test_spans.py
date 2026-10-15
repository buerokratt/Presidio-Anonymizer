"""Unit tests for model span shaping.

_merge_adjacent_spans() joins same-type spans separated by spaces alone, so a
name the model labelled one word at a time is filtered as one entity instead of
being half dropped. Runs without a container or a model -- the method is static,
so nothing here loads the NER pipeline:

    uv run python tests/test_spans.py

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from presidio_analyzer import RecognizerResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presidio_flask_estbert import EstBERTRecognizerONNX  # noqa: E402

merge = EstBERTRecognizerONNX._merge_adjacent_spans


def span(
    text: str, surface: str, entity: str, score: float, after: str = ""
) -> RecognizerResult:
    """A result over an occurrence of `surface` in `text`.

    `after` picks which one: the search starts at that substring, which is how
    a single letter can be pinned to the word it belongs to rather than to the
    first time that letter appears.
    """
    start = text.index(surface, text.index(after) if after else 0)
    return RecognizerResult(
        entity_type=entity, start=start, end=start + len(surface), score=score
    )


def shape(text: str, results: list[RecognizerResult]) -> list[tuple[str, str, float]]:
    return [
        (r.entity_type, text[r.start : r.end], round(r.score, 2))
        for r in merge(text, results)
    ]


def run() -> int:
    cases: list[tuple[str, str, list[RecognizerResult], list[tuple[str, str, float]]]]
    t1 = "Ta kolis New Yorki ja elas Tallinnas."
    t2 = "Ta käis Tallinnas, Tartus ja Pärnus."
    t3 = "Maksu- ja Tolliamet\nTöötukassa vastas."
    t4 = "Jaan Tamm helistas."
    t5 = "Tallinn on suur."
    t6 = "Ta kolis New York City piirkonda."
    t7 = "Transpordiamet teatas, et sõiduk 45 XYZ sai trahvi."
    t8 = "TELLISIN LEHE PÕIK 7 LINNA KOHILA AGA KEEGI POLE TULNUD."

    cases = [
        (
            "a weak tail is carried by the strong head",
            t1,
            [span(t1, "New", "GPE", 0.95), span(t1, "Yorki", "GPE", 0.40)],
            [("GPE", "New Yorki", 0.95)],
        ),
        (
            "a weak head is carried by the strong tail",
            t1,
            [span(t1, "New", "GPE", 0.40), span(t1, "Yorki", "GPE", 0.95)],
            [("GPE", "New Yorki", 0.95)],
        ),
        (
            "different types are never merged",
            t4,
            [span(t4, "Jaan", "PERSON", 0.9), span(t4, "Tamm", "ORGANIZATION", 0.9)],
            [("PERSON", "Jaan", 0.9), ("ORGANIZATION", "Tamm", 0.9)],
        ),
        (
            "a comma between spans blocks the merge",
            t2,
            [span(t2, "Tallinnas", "GPE", 0.9), span(t2, "Tartus", "GPE", 0.5)],
            [("GPE", "Tallinnas", 0.9), ("GPE", "Tartus", 0.5)],
        ),
        (
            "a newline between spans blocks the merge",
            t3,
            [
                span(t3, "Maksu- ja Tolliamet", "ORGANIZATION", 0.9),
                span(t3, "Töötukassa", "ORGANIZATION", 0.5),
            ],
            [
                ("ORGANIZATION", "Maksu- ja Tolliamet", 0.9),
                ("ORGANIZATION", "Töötukassa", 0.5),
            ],
        ),
        (
            "a word between spans blocks the merge",
            t1,
            [span(t1, "Yorki", "GPE", 0.9), span(t1, "Tallinnas", "GPE", 0.9)],
            [("GPE", "Yorki", 0.9), ("GPE", "Tallinnas", 0.9)],
        ),
        (
            "three fragments collapse into one span",
            t6,
            [
                span(t6, "New", "GPE", 0.95),
                span(t6, "York", "GPE", 0.30),
                span(t6, "City", "GPE", 0.20),
            ],
            [("GPE", "New York City", 0.95)],
        ),
        (
            "a span that already swallowed a comma merges across it",
            # _normalize_span cuts at commas, so its pieces never end on one --
            # but if a span ever did, the gap to the next span is a bare space
            # and they join. Recorded so the behaviour is not a surprise.
            t2,
            [span(t2, "Tallinnas,", "GPE", 0.9), span(t2, "Tartus", "GPE", 0.3)],
            [("GPE", "Tallinnas, Tartus", 0.9)],
        ),
        (
            "merging must not reassemble a registration plate",
            # _normalize_span drops a plate-shaped piece so the model cannot
            # claim a plate as an organisation. It splits "45 XYZ" into two
            # pieces that are not plate-shaped alone, so the guard has to run
            # again after the merge.
            t7,
            [
                span(t7, "45", "ORGANIZATION", 0.99),
                span(t7, "XYZ", "ORGANIZATION", 0.99),
            ],
            [],
        ),
        (
            "a merged span that is not a plate is kept",
            t7,
            [
                span(t7, "Transpordiamet", "ORGANIZATION", 0.9),
                span(t7, "teatas", "ORGANIZATION", 0.4),
            ],
            [("ORGANIZATION", "Transpordiamet teatas", 0.9)],
        ),
        (
            "a merge that would half-cover another span is abandoned",
            # Taken from a real sentence in test_fresh.conll. The model emits a
            # one-letter LOCATION fragment that sits inside a GPE span.
            # Absorbing it into the LOCATION to its left would leave LOCATION
            # and GPE partly overlapping, which the anonymizer cannot resolve -
            # it writes both and the shared character appears twice.
            t8,
            [
                span(t8, "LEHE PÕIK 7", "LOCATION", 0.9),
                span(t8, "L", "LOCATION", 0.3, after="LINNA"),
                span(t8, "LINNA KOHILA", "GPE", 0.9),
            ],
            [
                ("LOCATION", "LEHE PÕIK 7", 0.9),
                ("LOCATION", "L", 0.3),
                ("GPE", "LINNA KOHILA", 0.9),
            ],
        ),
        (
            "a single span is unchanged",
            t5,
            [span(t5, "Tallinn", "GPE", 0.7)],
            [("GPE", "Tallinn", 0.7)],
        ),
        ("no spans in, no spans out", t5, [], []),
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
