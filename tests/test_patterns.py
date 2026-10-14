"""Pattern-level regression tests for the YAML recognizers.

Runs the regexes straight out of the config, with no model and no container, so
a pattern change can be checked in under a second.

The patterns are compiled with PATTERN_FLAGS, which is what Presidio's
PatternRecognizer uses by default. IGNORECASE is the one that matters: it is on
whether or not a pattern was written expecting it, so an uppercase-only
character class does not mean what it looks like it means. Compiling without
those flags here would make this file assert behaviour the running service does
not have.

A recognizer's strict `validator:` (Luhn, mod-97, ...) is applied too, so a
card case asserts what the service reports, not every 16-digit run. The
validators are imported from the service module; the model is never loaded.

    uv run python tests/test_patterns.py
    uv run python tests/test_patterns.py --config config/presidio-stanza-estbert.yml

Each case names a text and, per entity, the exact surfaces that entity's
patterns must match in it -- an empty list means "must not match anything".
Only the entities a case lists are checked, so a case can pin down one
recognizer without asserting the behaviour of every other.

Exit code is 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presidio_flask_estbert import VALIDATORS, find_validated_matches  # noqa: E402

# Presidio's PatternRecognizer default. Our recognizers are built without
# overriding it, so this is what the service actually runs.
PATTERN_FLAGS = re.IGNORECASE | re.MULTILINE | re.DOTALL

# Cases below are grouped by the recognizer they are about, but any case may
# assert several entities at once -- that is how cross-pattern collisions get
# caught (a date pattern claiming part of an IBAN, say).
CASES: list[dict[str, Any]] = [
    # ---------------------------------------------------------------- URL
    {
        "name": "url: comma-separated list keeps its punctuation out of the spans",
        "text": (
            "veebilehtede näited on sellised: google.com, err.ee, "
            "example.com/about, company.ee/kontakt, shop.example.com/products, "
            "portal.example.ee/login, example.com/search?q=test, "
            "shop.example.com/products?category=books&sort=price, "
            "example.com/profile?id=12345, "
            "news.example.org/article/123?utm_source=newsletter, "
            "support.example.com/help/account?lang=et&topic=password."
        ),
        "expect": {
            "URL": [
                "google.com",
                "err.ee",
                "example.com/about",
                "company.ee/kontakt",
                "shop.example.com/products",
                "portal.example.ee/login",
                "example.com/search?q=test",
                "shop.example.com/products?category=books&sort=price",
                "example.com/profile?id=12345",
                "news.example.org/article/123?utm_source=newsletter",
                "support.example.com/help/account?lang=et&topic=password",
            ]
        },
    },
    {
        "name": "url: scheme, bare trailing slash and path all survive",
        "text": "Vaata https://www.example.test/ ja https://www.example.test/kontakt siin.",
        "expect": {
            "URL": ["https://www.example.test/", "https://www.example.test/kontakt"]
        },
    },
    {
        "name": "url: sentence-final and parenthesised URLs stop at the punctuation",
        "text": "Vaata err.ee. Rohkem infot example.com/about. (vt ka company.ee/kontakt)",
        "expect": {"URL": ["err.ee", "example.com/about", "company.ee/kontakt"]},
    },
    {
        "name": "url: prices and abbreviations are not URLs",
        "text": "Arve oli 15.30 eurot, aadress Pärnu mnt. 12 ja trahv 500 EUR.",
        "expect": {"URL": []},
    },
    # ------------------------------------------------------------- CRYPTO
    {
        "name": "crypto: ethereum addresses are detected",
        "text": (
            "krüptorahakoti näited: 0x000000000000000000000000000000000000dEaD "
            "ja 0x742d35Cc6634C0532925a3b844Bc454e4438f44e."
        ),
        "expect": {
            "CRYPTO": [
                "0x000000000000000000000000000000000000dEaD",
                "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            ]
        },
    },
    {
        "name": "crypto: wrong-length hex strings are not addresses",
        "text": "Vale pikkus 0xdEaD ja 0x742d35Cc6634C0532925a3b844Bc454e4438f44e1234.",
        "expect": {"CRYPTO": []},
    },
    # --------------------------------------------------------- EST_ID_DOC
    {
        "name": "doc: every prefix series currently issued by PPA",
        "text": (
            "AB1234567 AC1234567 AD1234567 EB1234567 EC1234567 ED1234567 "
            "NA1234567 UA1234567 UB1234567 PE1234567 PF1234567 BE1234567 "
            "BF1234567 FE1234567 FF1234567 KD1234567 KE1234567 KF1234567 "
            "KG1234567 VD1234567 VE1234567 VF1234567 VG1234567 MF1234567 "
            "MG1234567 SF1234567 SG1234567 RF1234567 CF1234567"
        ),
        "expect": {
            "EST_ID_DOC": [
                f"{p}1234567"
                for p in (
                    "AB AC AD EB EC ED NA UA UB PE PF BE BF FE FF KD KE KF KG "
                    "VD VE VF VG MF MG SF SG RF CF"
                ).split()
            ]
        },
    },
    {
        "name": "doc: a one-letter prefix is not a document number",
        "text": "Vale N1234567 ja liiga lühike KG234567 ja tundmatu XX1234567.",
        "expect": {"EST_ID_DOC": []},
    },
    # ---------------------------------------------------------- DATE_TIME
    {
        "name": "date: dotted, ISO and written-out forms",
        "text": "Kohtume 23.09.2026, 21.09.26, 2026-09-23, 23. september 2026 ja 3. mai 2026.",
        "expect": {
            "DATE_TIME": [
                "23.09.2026",
                "21.09.26",
                "2026-09-23",
                "23. september 2026",
                "3. mai 2026",
            ]
        },
    },
    {
        "name": "date: time with and without seconds",
        "text": "Algus 15:30 ja täpsemalt 15:30:00.",
        "expect": {"DATE_TIME": ["15:30", "15:30:00"]},
    },
    {
        "name": "date: a dotted decimal is not a time",
        "text": "Arve oli 15.30 eurot ja kaal 2.50 kg.",
        "expect": {"DATE_TIME": []},
    },
    # ------------------------------------------------ audit-era negatives
    {
        "name": "plate: currency and units are not registration numbers",
        "text": "Trahv oli 500 EUR ja kiirus 45 km/h, aga auto on 123 ABC.",
        "expect": {"CAR_NUMBER": ["123 ABC"]},
    },
    {
        "name": "phone: an isikukood is not a phone number",
        "text": "Minu isikukood on 39001010000 ja teine 49403136515.",
        "expect": {
            "PHONE_NUMBER": [],
            "EE_PERSONAL_CODE": ["39001010000", "49403136515"],
        },
    },
    {
        "name": "phone: estonian formats",
        "text": "Helista 55501234 või +372 5550 1234 või 003725551234.",
        "expect": {"PHONE_NUMBER": ["55501234", "+372 5550 1234", "003725551234"]},
    },
    {
        "name": "phone: IBAN groups are not phone numbers",
        "text": "Kontod EE38 2200 2210 2014 5685 ja DE89 3704 0044 0532 0130 00.",
        "expect": {"PHONE_NUMBER": []},
    },
    {
        "name": "phone: the card-number overlap is known and shadowed",
        # The first two groups of a 16-digit card also fit the bare
        # local-number branch. The card span contains the phone span, and the
        # anonymizer drops a span contained in another, so this only leaks if
        # the card goes undetected - which is what the card cases below pin.
        "text": "Kaart 4242 4242 4242 4242.",
        "expect": {
            "PHONE_NUMBER": ["4242 4242"],
            "CREDIT_CARD": ["4242 4242 4242 4242"],
        },
    },
    # --------------------------------------------------------- CREDIT_CARD
    # Luhn-checked and strict: a match that fails the check is not reported.
    {
        "name": "card: Mastercard 2-series and Maestro 56-58 are cards",
        # Presidio's built-in regex stopped at a leading 1, 3, 4, 50-55 or 6,
        # so none of these reached its Luhn check.
        "text": "Kaardid 2223 0031 2200 3222, 5893000000000009 ja 5612 3456 7890 1230.",
        "expect": {
            "CREDIT_CARD": [
                "2223 0031 2200 3222",
                "5893000000000009",
                "5612 3456 7890 1230",
            ],
        },
    },
    {
        "name": "card: the phone pattern is shadowed inside a Maestro number too",
        # Before, the card went undetected and this phone match was the only
        # span: the output read "5612 3456 [TELEFON]".
        "text": "Kaart 5612 3456 7890 1230.",
        "expect": {
            "CREDIT_CARD": ["5612 3456 7890 1230"],
            "PHONE_NUMBER": ["7890 1230"],
        },
    },
    {
        "name": "card: non-breaking spaces and hyphens separate groups",
        "text": "Kaart 4242\u00a04242\u00a04242\u00a04242 või 4111-1111-1111-1111.",
        "expect": {
            "CREDIT_CARD": ["4242\u00a04242\u00a04242\u00a04242", "4111-1111-1111-1111"]
        },
    },
    {
        "name": "card: the ways one number is commonly written",
        # The text is not normalised; the pattern tolerates the spacing and
        # Luhn ignores it. Each line of the text is one card.
        "text": (
            "4242424242424242\n"
            "4242 4242 4242 4242\n"
            "4242-4242-4242-4242\n"
            "4242  4242  4242  4242\n"
            "4242   4242 4242 4242\n"
            "4242 - 4242 - 4242 - 4242\n"
            "4242-4242 4242-4242\n"
            "4242\t4242\t4242\t4242\n"
            "3782-822463-10005\n"
            "3782  822463  10005"
        ),
        "expect": {
            "CREDIT_CARD": [
                "4242424242424242",
                "4242 4242 4242 4242",
                "4242-4242-4242-4242",
                "4242  4242  4242  4242",
                "4242   4242 4242 4242",
                "4242 - 4242 - 4242 - 4242",
                "4242-4242 4242-4242",
                "4242\t4242\t4242\t4242",
                "3782-822463-10005",
                "3782  822463  10005",
            ]
        },
    },
    {
        "name": "card: a line break is not a separator",
        # Two numbers on consecutive lines are two numbers, not one card.
        "text": "Kaart 4242 4242\n4242 4242.",
        "expect": {"CREDIT_CARD": []},
    },
    {
        "name": "card: KNOWN LIMITATION - dots and slashes are not separators",
        # They are how dates and amounts are written; nobody writes a card so.
        "text": "Kaart 4242.4242.4242.4242 või 4242/4242/4242/4242.",
        "expect": {"CREDIT_CARD": []},
    },
    {
        "name": "card: Amex and Diners groupings",
        "text": "Amex 3782 822463 10005 ja Diners 3056 930902 5904.",
        "expect": {"CREDIT_CARD": ["3782 822463 10005", "3056 930902 5904"]},
    },
    {
        "name": "card: a number that fails Luhn is not a card",
        "text": "Tellimus 1234 5678 9012 3456 ja 4242 4242 4242 4241.",
        "expect": {"CREDIT_CARD": []},
    },
    {
        "name": "card: a following number does not swallow the card",
        "text": "Kaart 4242 4242 4242 4242 CVC 123.",
        "expect": {"CREDIT_CARD": ["4242 4242 4242 4242"]},
    },
    {
        "name": "card: a rejected match does not hide the card inside it",
        # The year and the first three groups are matched first and fail Luhn.
        # finditer would resume after them and never try the card itself.
        "text": "Aasta 2024 4242 4242 4242 4242.",
        "expect": {"CREDIT_CARD": ["4242 4242 4242 4242"]},
    },
    {
        "name": "card: other groupings and doubled separators",
        "text": (
            "Amex 3782 8224 6310 005, UnionPay 6212 3456 7890 1234 569 "
            "ja 4242  4242 4242 4242."
        ),
        "expect": {
            "CREDIT_CARD": [
                "3782 8224 6310 005",
                "6212 3456 7890 1234 569",
                "4242  4242 4242 4242",
            ]
        },
    },
    {
        "name": "card: an expiry date after the card does not hide it",
        "text": "Kaart 4242 4242 4242 4242 12/26.",
        "expect": {"CREDIT_CARD": ["4242 4242 4242 4242"]},
    },
    {
        "name": "card: every major brand's test number",
        "text": (
            "Visa 4111111111111111, Visa-13 4222222222222, "
            "Mastercard 5555555555554444, Mastercard-2 2221000000000009, "
            "Amex 371449635398431, Discover 6011111111111117, "
            "JCB 3530111333300000, UnionPay 6200000000000005, "
            "Diners 36227206271667, Maestro 6759649826438453."
        ),
        "expect": {
            "CREDIT_CARD": [
                "4111111111111111",
                "4222222222222",
                "5555555555554444",
                "2221000000000009",
                "371449635398431",
                "6011111111111117",
                "3530111333300000",
                "6200000000000005",
                "36227206271667",
                "6759649826438453",
            ]
        },
    },
    {
        "name": "card: surrounding punctuation is not part of the number",
        "text": (
            "4242 4242 4242 4242 on minu kaart (5555 5555 5555 4444), "
            "kaart:4111111111111111; „2223 0031 2200 3222” ja "
            "lõpuks 6011 1111 1111 1117!"
        ),
        "expect": {
            "CREDIT_CARD": [
                "4242 4242 4242 4242",
                "5555 5555 5555 4444",
                "4111111111111111",
                "2223 0031 2200 3222",
                "6011 1111 1111 1117",
            ]
        },
    },
    {
        "name": "card: numbers that are not cards",
        # The look-alikes most likely in Estonian chats: phone numbers,
        # amounts, dates, a list of years, payment reference and registry
        # codes, an ISBN and an IP address.
        "text": (
            "Telefon 5550 1234, 5123 4567 ja +372 5550 1234. "
            "Summa 2 500 000 eurot ja 1 234 567,89 eurot. "
            "Kuupäev 12.05.2024, 2024-05-12 kell 12:30. "
            "Aastatel 2021 2022 2023 2024. "
            "Viitenumber 2900082500, registrikood 10137319. "
            "ISBN 978-3-16-148410-0, server 192.168.1.1."
        ),
        "expect": {"CREDIT_CARD": []},
    },
    {
        "name": "card: an isikukood and a phone number are not cards",
        "text": "Isikukood 39001010000, telefon 003725551234.",
        "expect": {"CREDIT_CARD": []},
    },
    # ----------------------------------------------------------- IBAN_CODE
    {
        "name": "iban: grouping, case and separators do not matter",
        # IbanRecognizer was case-sensitive and knew only "-" and " ".
        "text": (
            "Kontod EE38 2200 2210 2014 5685, EE382200221020145685, "
            "ee382200221020145685, ee38 2200 2210 2014 5685 ja "
            "EE38\u00a02200\u00a02210\u00a02014\u00a05685."
        ),
        "expect": {
            "IBAN_CODE": [
                "EE38 2200 2210 2014 5685",
                "EE382200221020145685",
                "ee382200221020145685",
                "ee38 2200 2210 2014 5685",
                "EE38\u00a02200\u00a02210\u00a02014\u00a05685",
            ]
        },
    },
    {
        "name": "iban: the ways one account is commonly written",
        "text": (
            "EE38-2200-2210-2014-5685\n"
            "EE38  2200  2210  2014  5685\n"
            "EE38 - 2200 - 2210 - 2014 - 5685\n"
            "GB29  NWBK  6016  1331  9268  19\n"
            "GB29-NWBK-6016-1331-9268-19"
        ),
        "expect": {
            "IBAN_CODE": [
                "EE38-2200-2210-2014-5685",
                "EE38  2200  2210  2014  5685",
                "EE38 - 2200 - 2210 - 2014 - 5685",
                "GB29  NWBK  6016  1331  9268  19",
                "GB29-NWBK-6016-1331-9268-19",
            ]
        },
    },
    {
        "name": "iban: foreign all-digit and alphanumeric accounts",
        "text": (
            "Kontod DE89 3704 0044 0532 0130 00, LT121000011101001000, "
            "GB29 NWBK 6016 1331 9268 19 ja gb29nwbk60161331926819."
        ),
        "expect": {
            "IBAN_CODE": [
                "DE89 3704 0044 0532 0130 00",
                "LT121000011101001000",
                "GB29 NWBK 6016 1331 9268 19",
                "gb29nwbk60161331926819",
            ]
        },
    },
    {
        "name": "iban: an all-digit IBAN that fails mod-97 is still anonymised",
        # validator_mode: boost. One digit off is a typo of someone's account
        # number, not a coincidence - two letters, two check digits and sixteen
        # more digits do not turn up by accident.
        "text": "Konto EE382200221020145686 ja EE99 9999 9999 9999 9999.",
        "expect": {"IBAN_CODE": ["EE382200221020145686", "EE99 9999 9999 9999 9999"]},
    },
    {
        "name": "iban: an alphanumeric shape must pass mod-97",
        # validator_mode: strict. Letters in the groups would let words match.
        "text": "Konto GB29 NWBK 6016 1331 9268 18 ja kood AB12 test tere asja.",
        "expect": {"IBAN_CODE": []},
    },
    {
        "name": "iban: a word after an alphanumeric IBAN is not its last group",
        # RO accounts end on a full group, so "ja" fit the optional tail, the
        # whole match failed mod-97 and the account was not reported at all.
        "text": "Konto RO49 AAAA 1B31 0075 9384 0000 ja RO49AAAA1B31007593840000 ka.",
        "expect": {
            "IBAN_CODE": [
                "RO49 AAAA 1B31 0075 9384 0000",
                "RO49AAAA1B31007593840000",
            ]
        },
    },
    {
        "name": "iban: a number right after the IBAN is not part of it",
        # The greedy match is four digits too long for EE and is rejected; the
        # same start is tried shorter and the account itself is reported.
        "text": "Konto EE38 2200 2210 2014 5685 2024 ja EE38 2200 2210 2014 5686 12.",
        "expect": {
            "IBAN_CODE": ["EE38 2200 2210 2014 5685", "EE38 2200 2210 2014 5686"]
        },
    },
    {
        "name": "iban: registry examples from more countries",
        "text": (
            "FI21 1234 5600 0007 85, PL61 1090 1014 0000 0712 1981 2874, "
            "BE68 5390 0754 7034, NO93 8601 1117 947, "
            "SE45 5000 0000 0583 9825 7466, LV80 BANK 0000 4351 9500 1, "
            "FR14 2004 1010 0505 0001 3M02 606, IT60 X054 2811 1010 0000 0123 456, "
            "CH93 0076 2011 6238 5295 7, PT50 0002 0123 1234 5678 9015 4."
        ),
        "expect": {
            "IBAN_CODE": [
                "FI21 1234 5600 0007 85",
                "PL61 1090 1014 0000 0712 1981 2874",
                "BE68 5390 0754 7034",
                "NO93 8601 1117 947",
                "SE45 5000 0000 0583 9825 7466",
                "LV80 BANK 0000 4351 9500 1",
                "FR14 2004 1010 0505 0001 3M02 606",
                "IT60 X054 2811 1010 0000 0123 456",
                "CH93 0076 2011 6238 5295 7",
                "PT50 0002 0123 1234 5678 9015 4",
            ]
        },
    },
    {
        "name": "iban: surrounding punctuation is not part of the account",
        "text": (
            "EE382200221020145685 on konto (EE38 2200 2210 2014 5685), "
            "IBAN:EE382200221020145685; „LT121000011101001000” ja "
            "lõpuks DE89370400440532013000!"
        ),
        "expect": {
            "IBAN_CODE": [
                "EE382200221020145685",
                "EE38 2200 2210 2014 5685",
                "EE382200221020145685",
                "LT121000011101001000",
                "DE89370400440532013000",
            ]
        },
    },
    {
        "name": "iban: serial numbers and codes shaped like an IBAN are not one",
        # The typo allowance needs a real IBAN country and that country's
        # length; a serial number has neither, a US number is not an IBAN, and
        # an EE number four digits short is not an Estonian account.
        "text": (
            "Seerianumber SN12 3456 7890 1234, tootekood AB12 3456 7890 1234 5678, "
            "US12 3456 7890 1234 5678, EE38 2200 2210 2014, "
            "konto 38 2200 2210 2014 5685, dokument AB1234567."
        ),
        "expect": {"IBAN_CODE": []},
    },
    {
        "name": "iban: too short for any country is not an IBAN",
        "text": "Kood XY12 3456 7890 ja AB1234567890.",
        "expect": {"IBAN_CODE": []},
    },
    {
        "name": "phone: international formats",
        "text": "Helista +44 7700 900123 või +1 202 555 0147.",
        "expect": {"PHONE_NUMBER": ["+44 7700 900123", "+1 202 555 0147"]},
    },
    # ------------------------------------------------------------ LOCATION
    {
        "name": "coords: decimal pairs are locations",
        "text": (
            "asukoha näited on sellised: 59.4370, 24.7536. "
            "Sydney on -33.8688, 151.2093 ja tihedalt 59.4370,24.7536."
        ),
        "expect": {
            "LOCATION": [
                "59.4370, 24.7536",
                "-33.8688, 151.2093",
                "59.4370,24.7536",
            ]
        },
    },
    {
        "name": "coords: degree and DMS forms",
        "text": (
            "Punkt 59.4370° N, 24.7536° E ja 59°26'13.2\"N 24°45'13.0\"E "
            "ja 59°26' N, 24°45' E."
        ),
        "expect": {
            "LOCATION": [
                "59.4370° N, 24.7536° E",
                "59°26'13.2\"N 24°45'13.0\"E",
                "59°26' N, 24°45' E",
            ]
        },
    },
    {
        "name": "tricky/coords: two-decimal sums are not coordinates",
        # The whole discriminator: money and measurements carry two decimal
        # places, coordinates four or more.
        "text": "Arve oli 15.30, 24.50 eurot ja kaal 2.50, 3.75 kg.",
        "expect": {"LOCATION": []},
    },
    {
        "name": "tricky/coords: dates and versions are not coordinates",
        "text": "Kuupäevad 23.09.2026, 21.09.26, versioon 1.2.3 ja pi 3.14.",
        "expect": {"LOCATION": []},
    },
    {
        "name": "tricky/coords: out-of-range pairs are not coordinates",
        # Latitude stops at 90 and longitude at 180, so a pair of large
        # four-decimal numbers is not silently accepted.
        "text": "Näidud 91.5000, 24.7536 ja 59.4370, 181.9999.",
        "expect": {"LOCATION": []},
    },
    {
        "name": "tricky/coords: KNOWN LIMITATION - a four-decimal sum pair",
        # Nothing separates "2.5000, 3.7500" from a coordinate off the coast
        # of Ghana. Four decimals on a price is unusual enough to accept.
        "text": "Mõõdud 2.5000, 3.7500 ühikut.",
        "expect": {"LOCATION": ["2.5000, 3.7500"]},
    },
    # ==================================================================
    # Tricky near-misses: text that must NOT be anonymised, and the places
    # where these patterns are known to over-reach. A case marked KNOWN
    # LIMITATION asserts current behaviour rather than desired behaviour --
    # it is here so the cost stays visible and cannot quietly grow.
    # ==================================================================
    {
        "name": "tricky/url: attachments are not hosts",
        "text": "Saatsin aruanne.pdf, pilt.jpg, tabel.xlsx ja esitlus.pptx.",
        "expect": {"URL": []},
    },
    {
        "name": "tricky/url: an e-mail address is not a URL",
        "text": "Kirjuta mari.tamm@example.com või info@company.org.",
        "expect": {"URL": [], "EMAIL_ADDRESS": []},
    },
    {
        "name": "tricky/url: version numbers and decimals are not hosts",
        "text": "Versioon v2.10, pi on 3.14 ja hind 15.30.",
        "expect": {"URL": []},
    },
    {
        "name": "tricky/url: KNOWN LIMITATION - a missing space after a full stop",
        # "Tere.Kuidas" has exactly the shape of host + TLD. Distinguishing it
        # needs a real TLD allowlist; until then a typo costs one [VEEBILEHT].
        "text": "Tere.Kuidas läheb?",
        "expect": {"URL": ["Tere.Kuidas"]},
    },
    {
        "name": "tricky/url: KNOWN LIMITATION - extensions that are also live TLDs",
        # .py is Paraguay, .zip is a real gTLD, so these cannot be excluded the
        # way .pdf and .xlsx are without losing genuine domains.
        "text": "Skript skript.py ja arhiiv pakk.zip.",
        "expect": {"URL": ["skript.py", "pakk.zip"]},
    },
    {
        "name": "tricky/phone: KNOWN LIMITATION - registry codes share the shape",
        # An Estonian MTU registry code starts with 8 and is 8 digits, which is
        # exactly an Estonian mobile number. No pattern can separate them; only
        # context could. Company codes start with 1 and are safe.
        "text": "MTÜ registrikood on 80123456 ja OÜ oma 10060701.",
        "expect": {"PHONE_NUMBER": ["80123456"]},
    },
    {
        "name": "tricky/phone: KNOWN LIMITATION - order numbers and grouped money",
        "text": "Tellimus nr 5551234, summa 5 500 000 eurot.",
        "expect": {"PHONE_NUMBER": ["5551234", "5 500 000"]},
    },
    {
        "name": "tricky/phone: the 00 prefix accepts any country code by design",
        # Deliberate: dropping the country-code allowlist behind 00 is what
        # restored +44 and +1. The cost is that a plain 00-prefixed code is
        # read as international.
        "text": "Kood 0012345678 ja number 003725551234.",
        "expect": {"PHONE_NUMBER": ["0012345678", "003725551234"]},
    },
    {
        "name": "tricky/phone: a 5-digit postcode is too short to be a number",
        "text": "Postiindeks on 15060 ja maja nr 7.",
        "expect": {"PHONE_NUMBER": []},
    },
    {
        "name": "tricky/date: invalid dates and year ranges are not dates",
        "text": "Vale 2026-13-45, kellaaeg 25:99, aastad 2024-2026.",
        "expect": {"DATE_TIME": []},
    },
    {
        "name": "tricky/date: KNOWN LIMITATION - a score reads as a clock time",
        "text": "Mängu seis oli 15:30 teisel poolajal.",
        "expect": {"DATE_TIME": ["15:30"]},
    },
    {
        "name": "tricky/date: KNOWN LIMITATION - a date range is missed entirely",
        # "23.-25. september 2026" matches nothing: the day before the month
        # name is preceded by "-", which is not in the pattern's lookbehind, so
        # not even the closing date is caught. Written out in full
        # ("23. september 2026") it is detected -- see the date case above.
        "text": "Periood 23.-25. september 2026.",
        "expect": {"DATE_TIME": []},
    },
    {
        "name": "tricky/doc: near-miss document shapes",
        # Six digits, eight digits, a space after the prefix, and a prefix that
        # PPA does not issue.
        "text": "Koodid AB123456, AB12345678, KG 1234567 ja OV1234567.",
        "expect": {"EST_ID_DOC": []},
    },
    {
        "name": "tricky/crypto: a git SHA is not an address, an uppercase 0X is",
        # 40 hex digits without the 0x prefix is a commit hash and must not
        # match. "0X" does match, because Presidio compiles with IGNORECASE -
        # which is the behaviour we want anyway: an address written with a
        # capital X is still an address.
        "text": (
            "Commit 3f2a1b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a ja "
            "0X742d35Cc6634C0532925a3b844Bc454e4438f44e."
        ),
        "expect": {"CRYPTO": ["0X742d35Cc6634C0532925a3b844Bc454e4438f44e"]},
    },
    {
        "name": "tricky/plate: KNOWN LIMITATION - uppercase unit abbreviations",
        # The exclusion list covers currency codes only. "45 KMH" is rare in
        # Estonian text, which is written "45 km/h" and does not match.
        "text": "Kiirus 45 KMH, kaal 20 KGF, auto 123 ABC, aga 45 km/h mitte.",
        "expect": {"CAR_NUMBER": ["45 KMH", "20 KGF", "123 ABC"]},
    },
    {
        "name": "tricky/isikukood: an 11-digit number with a first digit outside 1-8",
        # The first digit encodes sex and century and runs 1-8: 1-2 for the
        # 1800s through to 7-8 for the 2100s. 9 is not a thing.
        "text": "Number 99001010000 ei ole isikukood, 39001010000 on.",
        "expect": {"EE_PERSONAL_CODE": ["39001010000"]},
    },
    # ==================================================================
    # Case sensitivity. Presidio compiles with IGNORECASE, so a pattern that
    # looks case-sensitive is not one. These cases pin down what that means
    # per recognizer, because the difference is invisible in the YAML.
    # ==================================================================
    {
        "name": "flags/plate: a lower-case plate matches, a sum does not",
        # The class is written [A-ZÕÄÖÜ] but IGNORECASE makes it match either
        # case - deliberately kept, since a plate typed in lower case in a chat
        # is still a plate. The exclusion list is case-insensitive with it.
        "text": "Auto 123 abc ja 777 AAA, trahv 500 eur ja 500 EUR.",
        "expect": {"CAR_NUMBER": ["123 abc", "777 AAA"]},
    },
    {
        "name": "flags/plate: time and count abbreviations are not plates",
        # "ootasin 15 min" has the shape of a plate, and only matched at all
        # because IGNORECASE reaches lower-case words.
        "text": "Ootasin 15 min, kulus 500 mln eurot ja 30 sek.",
        "expect": {"CAR_NUMBER": []},
    },
    {
        "name": "flags/plate: KNOWN LIMITATION - any 3-letter word still matches",
        # The exclusion list can only name the abbreviations we know about.
        "text": "Kaal 250 kil ja pikkus 120 cmx.",
        "expect": {"CAR_NUMBER": ["250 kil", "120 cmx"]},
    },
    {
        "name": "flags/doc: a lower-case document number matches",
        "text": "Pass kg1234567 ja KG1234567.",
        "expect": {"EST_ID_DOC": ["kg1234567", "KG1234567"]},
    },
    {
        "name": "flags/url: host case does not matter, extension case does not either",
        "text": "Vaata EXAMPLE.COM ja Example.Com/About, manus Aruanne.PDF.",
        "expect": {"URL": ["EXAMPLE.COM", "Example.Com/About"]},
    },
    {
        "name": "flags/date: a capitalised month name still matches",
        "text": "Kohtume 23. September 2026 ja 24. september 2026.",
        "expect": {"DATE_TIME": ["23. September 2026", "24. september 2026"]},
    },
]


# A compiled pattern, and the check a match must pass to be reported - None
# when nothing can drop it.
CompiledPattern = tuple[re.Pattern[str], Callable[[str], bool] | None]


def boost_check(
    validator: Callable[[str], bool], plausible: Callable[[str], bool]
) -> Callable[[str], bool]:
    """A boost-mode match survives if it validates or at least looks right."""
    return lambda surface: validator(surface) or plausible(surface)


def load_patterns(config_path: str) -> dict[str, list[CompiledPattern]]:
    """Compile every YAML pattern, grouped by the entity it reports.

    A recognizer with a strict `validator:` drops the matches that fail it, as
    the service does. One in boost mode drops only what fails its `plausible:`
    check as well, and with no such check drops nothing.
    """
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    by_entity: dict[str, list[CompiledPattern]] = {}
    for recognizer in config.get("recognizers", []):
        entity = recognizer["supported_entity"]
        validator = VALIDATORS.get(recognizer.get("validator", ""))
        plausible = VALIDATORS.get(recognizer.get("plausible", ""))
        check: Callable[[str], bool] | None = None
        if validator is not None:
            if recognizer.get("validator_mode", "strict") == "strict":
                check = validator
            elif plausible is not None:
                check = boost_check(validator, plausible)
        for pattern in recognizer.get("patterns", []):
            by_entity.setdefault(entity, []).append(
                (re.compile(pattern["regex"], PATTERN_FLAGS), check)
            )
    return by_entity


def matches_for(patterns: list[CompiledPattern], text: str) -> list[str]:
    """Every surface the entity's patterns report, in order of appearance.

    Matches are found with the service's own find_validated_matches(), so a
    rejected match is retried shorter and then from inside, exactly as there.
    """
    spans: list[tuple[int, int, str]] = []
    for regex, check in patterns:

        def judge(
            surface: str, check: Callable[[str], bool] | None = check
        ) -> bool | None:
            return None if check is None or check(surface) else False

        spans.extend(
            (start, end, text[start:end])
            for start, end, _ in find_validated_matches(regex, text, judge)
        )
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int, str]] = []
    for start, end, surface in sorted(spans):
        if (start, end) not in seen:
            seen.add((start, end))
            unique.append((start, end, surface))
    return [surface for _, _, surface in unique]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/presidio-spacy-estbert.yml")
    parser.add_argument("--only", help="comma-separated substrings of case names")
    args = parser.parse_args()

    by_entity = load_patterns(args.config)
    cases = CASES
    if args.only:
        wanted = args.only.split(",")
        cases = [c for c in cases if any(w in c["name"] for w in wanted)]

    failures = 0
    for case in cases:
        problems: list[str] = []
        for entity, expected in case["expect"].items():
            actual = matches_for(by_entity.get(entity, []), case["text"])
            if actual != expected:
                problems.append(
                    f"    {entity}\n"
                    f"      expected: {expected}\n"
                    f"      actual:   {actual}"
                )
        if problems:
            failures += 1
            print(f"FAIL  {case['name']}")
            print("\n".join(problems))
        else:
            print(f"ok    {case['name']}")

    print(f"\n{len(cases) - failures}/{len(cases)} passed  ({args.config})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
