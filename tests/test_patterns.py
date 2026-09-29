"""Pattern-level regression tests for the YAML recognizers.

Runs the regexes straight out of the config, with no model and no container, so
a pattern change can be checked in under a second:

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
from typing import Any

import yaml

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
        # Pre-existing, and unchanged by the international-format work: the
        # first two groups of a 16-digit card also fit the bare local-number
        # branch. It costs nothing because CREDIT_CARD covers the whole number
        # and outranks it, but the case is here so it cannot quietly get worse.
        "text": "Kaart 4242 4242 4242 4242.",
        "expect": {
            "PHONE_NUMBER": ["4242 4242"],
            "CREDIT_CARD": ["4242 4242 4242 4242"],
        },
    },
    {
        "name": "phone: international formats",
        "text": "Helista +44 7700 900123 või +1 202 555 0147.",
        "expect": {"PHONE_NUMBER": ["+44 7700 900123", "+1 202 555 0147"]},
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
        "name": "tricky/crypto: a git SHA and an uppercase 0X are not addresses",
        # 40 hex digits without the 0x prefix is a commit hash; Ethereum
        # addresses are written with a lowercase x.
        "text": (
            "Commit 3f2a1b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a ja "
            "0X742d35Cc6634C0532925a3b844Bc454e4438f44e."
        ),
        "expect": {"CRYPTO": []},
    },
    {
        "name": "tricky/plate: KNOWN LIMITATION - uppercase unit abbreviations",
        # The exclusion list covers currency codes only. "45 KMH" is rare in
        # Estonian text, which is written "45 km/h" and does not match.
        "text": "Kiirus 45 KMH, kaal 20 KGF, auto 123 ABC, aga 45 km/h mitte.",
        "expect": {"CAR_NUMBER": ["45 KMH", "20 KGF", "123 ABC"]},
    },
    {
        "name": "tricky/isikukood: an 11-digit number with a wrong century digit",
        # The first digit encodes century and sex and only runs 1-6.
        "text": "Number 79001010000 ei ole isikukood, 39001010000 on.",
        "expect": {"EE_PERSONAL_CODE": ["39001010000"]},
    },
]


def load_patterns(config_path: str) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    """Compile every YAML pattern, grouped by the entity it reports."""
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    by_entity: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for recognizer in config.get("recognizers", []):
        entity = recognizer["supported_entity"]
        for pattern in recognizer.get("patterns", []):
            by_entity.setdefault(entity, []).append(
                (pattern["name"], re.compile(pattern["regex"]))
            )
    return by_entity


def matches_for(patterns: list[tuple[str, re.Pattern[str]]], text: str) -> list[str]:
    """Every surface the entity's patterns match, in order of appearance."""
    spans: list[tuple[int, int, str]] = []
    for _, regex in patterns:
        spans.extend(
            (match.start(), match.end(), match.group())
            for match in regex.finditer(text)
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
