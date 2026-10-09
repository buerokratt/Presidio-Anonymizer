"""Unit tests for recognizer validators.

A pattern describes a shape; a validator decides whether that shape is a real
identifier. Presidio scores a validated match 1.0 and an invalidated one 0, so
a shape-only match disappears rather than being reported with the pattern's
nominal confidence. Runs without a container or a model:

    uv run python tests/test_validators.py

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presidio_flask_estbert import validate_ee_personal_code  # noqa: E402

CASES: list[tuple[str, bool, str]] = [
    # --- real codes -------------------------------------------------------
    ("38001085718", True, "man born 1980, correct check digit"),
    ("37605030299", True, "from the acceptance-test screenshots"),
    ("39001010000", True, "from the acceptance-test list"),
    ("48507240000", True, "woman born 1985"),
    ("49403136515", True, "the number that used to be read as a phone"),
    # --- shape is right, the code is not ----------------------------------
    ("39001010001", False, "check digit changed by one"),
    ("60212310000", False, "invented for the acceptance test; check digit is 7"),
    ("47001010001", False, "invented for the gold suite"),
    ("61203074321", False, "invented for the gold suite"),
    # --- the 2100s, defined by the standard though nobody has one yet ------
    ("79403136518", True, "man born 2100-2199"),
    ("89403136519", True, "woman born 2100-2199"),
    # --- the published method cannot separate these two --------------------
    ("51107121760", True, "documented twin code"),
    ("61107121760", True, "its twin, one digit apart; both verify"),
    # --- impossible dates are not rejected on their own --------------------
    # The standard specifies only 01-31 for the day, so the check digit is the
    # only thing that decides. Both of these fail it anyway.
    ("39002300000", False, "30 February, and the check digit is wrong"),
    ("39002290000", False, "29 February 1900, and the check digit is wrong"),
    # --- wrong shape ------------------------------------------------------
    ("09001010000", False, "first digit 0, outside 1-8"),
    ("99001010000", False, "first digit 9, outside 1-8"),
    ("3900101000", False, "ten digits"),
    ("390010100000", False, "twelve digits"),
    ("3900101000a", False, "not all digits"),
    ("", False, "empty"),
]


def run() -> int:
    failures = 0
    for code, expected, note in CASES:
        actual = validate_ee_personal_code(code)
        if actual == expected:
            print(f"ok    {code or '(empty)':13} {note}")
        else:
            failures += 1
            print(
                f"FAIL  {code or '(empty)':13} expected {expected}, got {actual} - {note}"
            )
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
