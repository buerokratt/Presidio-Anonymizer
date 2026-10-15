"""Unit tests for recognizer validators.

A pattern describes a shape; a validator decides whether that shape is a real
identifier. Presidio scores a validated match 1.0; an invalidated one scores 0
and disappears in strict mode, or keeps the pattern's score in boost mode (see
VALIDATOR_MODES). Runs without a container or a model:

    uv run python tests/test_validators.py

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Callable  # noqa: E402

from presidio_flask_estbert import (  # noqa: E402
    validate_ee_personal_code,
    validate_iban,
    validate_iban_shape,
    validate_luhn,
)

PERSONAL_CODE_CASES: list[tuple[str, bool, str]] = [
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

LUHN_CASES: list[tuple[str, bool, str]] = [
    # --- cards the built-in recognizer never reached ----------------------
    ("2223 0031 2200 3222", True, "Mastercard 2-series"),
    ("5893000000000009", True, "Maestro 58"),
    ("5612 3456 7890 1230", True, "Maestro 56"),
    # --- grouping is not part of the number -------------------------------
    ("4242 4242 4242 4242", True, "spaces"),
    ("4242\u00a04242\u00a04242\u00a04242", True, "non-breaking spaces"),
    ("4111-1111-1111-1111", True, "hyphens"),
    ("378282246310005", True, "Amex, fifteen digits"),
    ("30569309025904", True, "Diners, fourteen digits"),
    # --- shape is right, the number is not --------------------------------
    ("4242 4242 4242 4241", False, "check digit changed by one"),
    ("1234 5678 9012 3456", False, "a run of digits, not a card"),
    # --- wrong shape ------------------------------------------------------
    ("424242424242", False, "twelve digits"),
    ("567890123456", False, "twelve digits that pass Luhn are still too short"),
    ("42424242424242424242", False, "twenty digits"),
    ("4242 4242 4242 424a", False, "not all digits"),
    ("", False, "empty"),
]

IBAN_CASES: list[tuple[str, bool, str]] = [
    ("EE382200221020145685", True, "Estonian, compact"),
    ("EE38 2200 2210 2014 5685", True, "Estonian, grouped"),
    ("ee38 2200 2210 2014 5685", True, "lower case"),
    ("EE38\u00a02200\u00a02210\u00a02014\u00a05685", True, "non-breaking spaces"),
    ("DE89 3704 0044 0532 0130 00", True, "German, short last group"),
    ("LT121000011101001000", True, "Lithuanian"),
    ("GB29 NWBK 6016 1331 9268 19", True, "letters in the account number"),
    ("NO9386011117947", True, "Norwegian, the shortest there is"),
    # --- shape is right, the account is not -------------------------------
    ("EE382200221020145686", False, "one digit off"),
    ("EE99 9999 9999 9999 9999", False, "invented"),
    ("GB29 NWBK 6016 1331 9268 18", False, "one digit off, with letters"),
    # --- wrong shape ------------------------------------------------------
    ("EE38220022", False, "too short"),
    ("EE38 2200 2210 2014 5685 2024", False, "valid account with a year run onto it"),
    ("SN12 3456 7890 1234", False, "not an IBAN country"),
    ("3838 2200 2210 2014 5685", False, "no country code"),
    ("EEAB 2200 2210 2014 5685", False, "letters for check digits"),
    ("", False, "empty"),
]

IBAN_SHAPE_CASES: list[tuple[str, bool, str]] = [
    # A typo leaves the country and the length alone, so these still count.
    ("EE382200221020145686", True, "one digit off, right country and length"),
    ("EE99 9999 9999 9999 9999", True, "invented, but Estonian-shaped"),
    ("PT50 0002 0123 1234 5678 9015 4", True, "Portugal, 25"),
    # --- not an IBAN country -----------------------------------------------
    ("SN12 3456 7890 1234", False, "a serial number"),
    ("US12 3456 7890 1234 5678", False, "the US issues no IBANs"),
    # --- right country, wrong length ---------------------------------------
    ("EE38 2200 2210 2014", False, "four digits short of an Estonian IBAN"),
    ("EE38 2200 2210 2014 5685 2024", False, "a year run onto it"),
    ("", False, "empty"),
]

SUITES: list[tuple[str, Callable[[str], bool], list[tuple[str, bool, str]]]] = [
    ("isikukood", validate_ee_personal_code, PERSONAL_CODE_CASES),
    ("luhn", validate_luhn, LUHN_CASES),
    ("iban", validate_iban, IBAN_CASES),
    ("iban_shape", validate_iban_shape, IBAN_SHAPE_CASES),
]


def run() -> int:
    failures = 0
    total = 0
    for suite, validator, cases in SUITES:
        print(f"--- {suite}")
        for code, expected, note in cases:
            total += 1
            shown = repr(code) if code else "(empty)"
            actual = validator(code)
            if actual == expected:
                print(f"ok    {shown:32} {note}")
            else:
                failures += 1
                print(f"FAIL  {shown:32} expected {expected}, got {actual} - {note}")
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
