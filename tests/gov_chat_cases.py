"""Gold-annotated Estonian government-agency chat cases for the anonymizer API.

Each detection case is a dict:
    id      - stable identifier
    group   - category used for the report
    desc    - what the case probes
    text    - the input sent as texts[0]
    gold    - list of (surface, accepted_entity_types) that MUST be detected.
              `surface` must occur verbatim in `text`; the first occurrence is
              used unless the tuple carries a third element (occurrence index).
              accepted_entity_types is a set: several labels are defensible for
              e.g. city names (LOCATION vs GPE), and the report records which
              one actually came back.
    traps   - list of surfaces that must NOT be flagged as PII at all
              (precision probes: money sums, statute references, case numbers).

Behaviour cases (allow/denylist, operators, HTTP contract) live in
BEHAVIOUR_CASES and are asserted directly by the runner.
"""

LOC = {"LOCATION", "GPE"}
ORG = {"ORGANIZATION"}
PER = {"PERSON"}

DETECTION_CASES: list[dict] = [
    # ------------------------------------------------------------------
    # A. Core PII inside realistic agency conversations
    # ------------------------------------------------------------------
    {
        "id": "A01",
        "group": "A. Agency conversations",
        "desc": "Tax board: name + personal code + agency",
        "text": (
            "Tere! Minu nimi on Jaan Tamm, isikukood 38001085718. "
            "Soovin teada, kas mul on Maksu- ja Tolliametile maksuvõlga."
        ),
        "gold": [
            ("Jaan Tamm", PER),
            ("38001085718", {"EE_PERSONAL_CODE"}),
            ("Maksu- ja Tolliametile", ORG),
        ],
        "traps": [],
    },
    {
        "id": "A02",
        "group": "A. Agency conversations",
        "desc": "Unemployment fund: name, phone, email",
        "text": (
            "Klient: Mari Mets soovib end Töötukassas töötuna arvele võtta. "
            "Telefon +372 5555 5555, e-post mari.mets@gmail.com."
        ),
        "gold": [
            ("Mari Mets", PER),
            ("Töötukassas", ORG),
            ("+372 5555 5555", {"PHONE_NUMBER"}),
            ("mari.mets@gmail.com", {"EMAIL_ADDRESS"}),
        ],
        "traps": [],
    },
    {
        "id": "A03",
        "group": "A. Agency conversations",
        "desc": "Social insurance board: pension payment to IBAN",
        "text": (
            "Sotsiaalkindlustusamet kannab Kalle Kase pensioni kontole "
            "EE382200221020145685 alates 01.09.2026."
        ),
        "gold": [
            ("Sotsiaalkindlustusamet", ORG),
            ("Kalle Kase", PER),
            ("EE382200221020145685", {"IBAN_CODE"}),
            ("01.09.2026", {"DATE_TIME"}),
        ],
        "traps": [],
    },
    {
        "id": "A04",
        "group": "A. Agency conversations",
        "desc": "Police and border guard: ID document number",
        "text": (
            "Politsei- ja Piirivalveamet väljastas Peeter Kuusele "
            "isikutunnistuse AA1234567, mis kehtib kuni 12.03.2031."
        ),
        "gold": [
            ("Politsei- ja Piirivalveamet", ORG),
            ("Peeter Kuusele", PER),
            ("AA1234567", {"EST_ID_DOC"}),
            ("12.03.2031", {"DATE_TIME"}),
        ],
        "traps": [],
    },
    {
        "id": "A05",
        "group": "A. Agency conversations",
        "desc": "Transport administration: car registration number",
        "text": (
            "Transpordiamet kinnitas, et sõiduk registreerimismärgiga 123 ABC "
            "kuulub Tiit Saarele."
        ),
        "gold": [
            ("Transpordiamet", ORG),
            ("123 ABC", {"CAR_NUMBER"}),
            ("Tiit Saarele", PER),
        ],
        "traps": [],
    },
    {
        "id": "A06",
        "group": "A. Agency conversations",
        "desc": "Health insurance fund: patient with personal code",
        "text": (
            "Tervisekassa andmetel on Liis Oja (isikukood 49403136515) "
            "ravikindlustus kehtiv."
        ),
        "gold": [
            ("Tervisekassa", ORG),
            ("Liis Oja", PER),
            ("49403136515", {"EE_PERSONAL_CODE"}),
        ],
        "traps": [],
    },
    {
        "id": "A07",
        "group": "A. Agency conversations",
        "desc": "City government: street address change",
        "text": (
            "Palun teatage Tallinna Linnavalitsusele, et Anne Lepik kolis "
            "aadressilt Liivalaia 2 aadressile Narva maantee 15, Tallinn."
        ),
        "gold": [
            ("Tallinna Linnavalitsusele", ORG),
            ("Anne Lepik", PER),
            ("Liivalaia 2", LOC),
            ("Narva maantee 15", LOC),
            ("Tallinn", LOC, 1),
        ],
        "traps": [],
    },
    {
        "id": "A08",
        "group": "A. Agency conversations",
        "desc": "State information system authority: email, URL, IP",
        "text": (
            "Riigi Infosüsteemi Amet teatas, et kasutaja toomas.raud@ria.ee "
            "logis sisse aadressilt 192.168.10.24 portaalis www.eesti.ee."
        ),
        "gold": [
            ("Riigi Infosüsteemi Amet", ORG),
            ("toomas.raud@ria.ee", {"EMAIL_ADDRESS"}),
            ("192.168.10.24", {"IP_ADDRESS"}),
            ("www.eesti.ee", {"URL"}),
        ],
        "traps": [],
    },
    {
        "id": "A09",
        "group": "A. Agency conversations",
        "desc": "Rescue board: caller, phone, address, clock time",
        "text": (
            "Päästeamet sai kell 14:30 teate Katrin Ilvese telefonilt 5551234, "
            "et Pärnu mnt 42 korteris on suitsulõhn."
        ),
        "gold": [
            ("Päästeamet", ORG),
            ("14:30", {"DATE_TIME"}),
            ("Katrin Ilvese", PER),
            ("5551234", {"PHONE_NUMBER"}),
            ("Pärnu mnt 42", LOC),
        ],
        "traps": [],
    },
    {
        "id": "A10",
        "group": "A. Agency conversations",
        "desc": "Parliament: petition signed on a spelled-out date",
        "text": (
            "Riigikogu menetles kollektiivset pöördumist, mille esitas "
            "Urmas Vaher 15 märts 2026 e-posti aadressilt urmas@petitsioon.ee."
        ),
        "gold": [
            ("Riigikogu", ORG),
            ("Urmas Vaher", PER),
            ("15 märts 2026", {"DATE_TIME"}),
            ("urmas@petitsioon.ee", {"EMAIL_ADDRESS"}),
        ],
        "traps": [],
    },
    # ------------------------------------------------------------------
    # B. Estonian morphology
    # ------------------------------------------------------------------
    {
        "id": "B01",
        "group": "B. Morphology",
        "desc": "Person name inflected (allative, genitive) across a dialogue",
        "text": (
            "Nõustaja: Kas te rääkisite Jaan Tammele? "
            "Klient: Jah, Jaan Tamme avaldus on juba esitatud."
        ),
        # "Jaan Tamme" is a prefix of "Jaan Tammele", so the second mention is
        # occurrence index 1.
        "gold": [
            ("Jaan Tammele", PER),
            ("Jaan Tamme", PER, 1),
        ],
        "traps": [],
    },
    {
        "id": "B02",
        "group": "B. Morphology",
        "desc": "Agency names in illative and ablative case",
        "text": (
            "Avaldus tuleb esitada Maksu- ja Tolliametisse ning koopia "
            "saata Töötukassalt saadud kirjaga Sotsiaalkindlustusametile."
        ),
        "gold": [
            ("Maksu- ja Tolliametisse", ORG),
            ("Töötukassalt", ORG),
            ("Sotsiaalkindlustusametile", ORG),
        ],
        "traps": [],
    },
    {
        "id": "B03",
        "group": "B. Morphology",
        "desc": "City names in inessive/elative case",
        "text": (
            "Kadri Sepp elab Tallinnas, kolis sinna Tartust ja töötas varem "
            "Narvas Politsei- ja Piirivalveametis."
        ),
        "gold": [
            ("Kadri Sepp", PER),
            ("Tallinnas", LOC),
            ("Tartust", LOC),
            ("Narvas", LOC),
            ("Politsei- ja Piirivalveametis", ORG),
        ],
        "traps": [],
    },
    {
        "id": "B04",
        "group": "B. Morphology",
        "desc": "Names with Estonian diacritics and foreign letters",
        "text": (
            "Kärt Õunapuu ja Jüri Müür esitasid Keskkonnaametile kaebuse, "
            "millele vastas Žanna Šišova."
        ),
        "gold": [
            ("Kärt Õunapuu", PER),
            ("Jüri Müür", PER),
            ("Keskkonnaametile", ORG),
            ("Žanna Šišova", PER),
        ],
        "traps": [],
    },
    {
        "id": "B05",
        "group": "B. Morphology",
        "desc": "Hyphenated given name and double surname",
        "text": (
            "Mari-Liis Kask-Tamm pöördus Justiitsministeeriumi poole "
            "isikukoodiga 47001010001."
        ),
        "gold": [
            ("Mari-Liis Kask-Tamm", PER),
            ("Justiitsministeeriumi", ORG),
            ("47001010001", {"EE_PERSONAL_CODE"}),
        ],
        "traps": [],
    },
    # ------------------------------------------------------------------
    # C. Pattern recognizers
    # ------------------------------------------------------------------
    {
        "id": "C01",
        "group": "C. Pattern recognizers",
        "desc": "Several valid personal codes in one message",
        "text": (
            "Perekonna isikukoodid: 38001085718, 49403136515 ja 61203074321 "
            "on Rahvastikuregistris kontrollitud."
        ),
        "gold": [
            ("38001085718", {"EE_PERSONAL_CODE"}),
            ("49403136515", {"EE_PERSONAL_CODE"}),
            ("61203074321", {"EE_PERSONAL_CODE"}),
        ],
        "traps": [],
    },
    {
        "id": "C02",
        "group": "C. Pattern recognizers",
        "desc": "Four phone number formats",
        "text": (
            "Helistage Maksu- ja Tolliameti infotelefonile 8800811 või "
            "+372 5555 5555, vajadusel 372 5551234 ja (+372) 55512345."
        ),
        "gold": [
            ("8800811", {"PHONE_NUMBER"}),
            ("+372 5555 5555", {"PHONE_NUMBER"}),
            ("372 5551234", {"PHONE_NUMBER"}),
            ("55512345", {"PHONE_NUMBER"}),
        ],
        "traps": [],
    },
    {
        "id": "C03",
        "group": "C. Pattern recognizers",
        "desc": "Estonian and foreign IBAN",
        "text": (
            "Riigilõiv kanti kontole EE382200221020145685, tagasimakse "
            "kontole LT121000011101001000."
        ),
        "gold": [
            ("EE382200221020145685", {"IBAN_CODE"}),
            ("LT121000011101001000", {"IBAN_CODE"}),
        ],
        "traps": [],
    },
    {
        "id": "C04",
        "group": "C. Pattern recognizers",
        "desc": "Car number vs. a money amount with the same shape",
        "text": (
            "Transpordiamet: sõiduk 123 ABC sai trahvi summas 500 EUR, "
            "teine sõiduk 45 XYZ jäi hoiatusega."
        ),
        "gold": [
            ("123 ABC", {"CAR_NUMBER"}),
            ("45 XYZ", {"CAR_NUMBER"}),
        ],
        "traps": ["500 EUR"],
    },
    {
        "id": "C05",
        "group": "C. Pattern recognizers",
        "desc": "Estonian document number prefixes",
        "text": (
            "Politsei- ja Piirivalveamet: pass AA1234567 ja "
            "elamisluba PB7654321 on kehtetud."
        ),
        "gold": [
            ("AA1234567", {"EST_ID_DOC"}),
            ("PB7654321", {"EST_ID_DOC"}),
        ],
        "traps": [],
    },
    {
        "id": "C06",
        "group": "C. Pattern recognizers",
        "desc": "Numeric, spelled-out and inflected dates plus a time",
        "text": (
            "Otsus tehti 12.03.2026, teade saadeti 15 märts 2026, "
            "istung toimub 20 aprillil 2026 kell 09:15."
        ),
        "gold": [
            ("12.03.2026", {"DATE_TIME"}),
            ("15 märts 2026", {"DATE_TIME"}),
            ("20 aprillil 2026", {"DATE_TIME"}),
            ("09:15", {"DATE_TIME"}),
        ],
        "traps": [],
    },
    {
        "id": "C07",
        "group": "C. Pattern recognizers",
        "desc": "Credit card and crypto address",
        "text": (
            "Riigilõivu tasumine kaardiga 4111 1111 1111 1111 ebaõnnestus, "
            "krüptomakse aadressile 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 samuti."
        ),
        "gold": [
            ("4111 1111 1111 1111", {"CREDIT_CARD"}),
            ("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", {"CRYPTO"}),
        ],
        "traps": [],
    },
    # ------------------------------------------------------------------
    # D. Precision traps - text that should stay untouched
    # ------------------------------------------------------------------
    {
        "id": "D01",
        "group": "D. Precision traps",
        "desc": "Purely institutional text, no personal data at all",
        "text": (
            "Maksu- ja Tolliamet avaldas teate, et e-maksuamet on hoolduse "
            "tõttu suletud. Küsimustega pöörduge klienditoe poole."
        ),
        "gold": [
            ("Maksu- ja Tolliamet", ORG),
        ],
        "traps": ["hoolduse", "klienditoe"],
    },
    {
        "id": "D02",
        "group": "D. Precision traps",
        "desc": "Money sums and a payment reference number",
        "text": (
            "Maksuteade: summa 1250,50 eurot, viitenumber 2900123456, "
            "tähtaeg möödas. Intress 0,06% päevas."
        ),
        "gold": [],
        "traps": ["1250,50", "0,06%"],
    },
    {
        "id": "D03",
        "group": "D. Precision traps",
        "desc": "Statute and gazette references",
        "text": (
            "Vastavalt maksukorralduse seaduse § 32 lõikele 2 "
            "(RT I 2002, 26, 150) on Maksu- ja Tolliametil õigus nõuda andmeid."
        ),
        "gold": [
            ("Maksu- ja Tolliametil", ORG),
        ],
        "traps": ["§ 32"],
    },
    {
        "id": "D04",
        "group": "D. Precision traps",
        "desc": "Institution names that are not people",
        "text": (
            "Vabariigi Valitsus ja Riigikogu kinnitasid, et Statistikaamet "
            "avaldab andmed järgmisel nädalal."
        ),
        "gold": [
            ("Vabariigi Valitsus", ORG),
            ("Riigikogu", ORG),
            ("Statistikaamet", ORG),
        ],
        "traps": [],
    },
    {
        "id": "D05",
        "group": "D. Precision traps",
        "desc": "Court case number and room number",
        "text": (
            "Haldusasi 3-21-1234 arutatakse ruumis 305. "
            "Kohtunik teatas otsuse edastamisest Justiitsministeeriumile."
        ),
        "gold": [
            ("Justiitsministeeriumile", ORG),
        ],
        "traps": ["3-21-1234", "ruumis 305"],
    },
    {
        "id": "D06",
        "group": "D. Precision traps",
        "desc": "Agency name as the first word of a short sentence",
        "text": "Päästeamet sai teate. Transpordiamet kinnitas otsuse.",
        "gold": [
            ("Päästeamet", ORG),
            ("Transpordiamet", ORG),
        ],
        "traps": [],
    },
    {
        "id": "D07",
        "group": "D. Precision traps",
        "desc": "IPv4 with an explicit context word (threshold reachability)",
        "text": "Riigi Infosüsteemi Amet: serveri IP on 192.168.10.24 ja IPv6 2001:db8::1.",
        "gold": [
            ("Riigi Infosüsteemi Amet", ORG),
            ("192.168.10.24", {"IP_ADDRESS"}),
            ("2001:db8::1", {"IP_ADDRESS"}),
        ],
        "traps": [],
    },
]

# ----------------------------------------------------------------------
# E/F/G. Behaviour: allow/denylist, operators, HTTP contract.
# Each case declares a request payload and a checker name handled by the
# runner (see run_behaviour_case).
# ----------------------------------------------------------------------
BEHAVIOUR_CASES: list[dict] = [
    {
        "id": "E01",
        "group": "E. Allow/denylist",
        "desc": "Allowlisted agency name survives anonymisation",
        "payload": {
            "texts": [
                "Jaan Tamm töötab Maksu- ja Tolliametis Tallinnas.",
            ],
            "allowlist": ["Maksu- ja Tolliamet"],
        },
        "expect_kept": ["Maksu- ja Tolliametis"],
        "expect_removed": ["Jaan Tamm"],
    },
    {
        "id": "E02",
        "group": "E. Allow/denylist",
        "desc": "Allowlist base form must cover an inflected mention (Vabamorf synthesis)",
        "payload": {
            "texts": ["Mari Mets käis Tallinnas Töötukassa vastuvõtul."],
            "allowlist": ["Tallinn"],
        },
        "expect_kept": ["Tallinnas"],
        "expect_removed": ["Mari Mets"],
    },
    {
        "id": "E03",
        "group": "E. Allow/denylist",
        "desc": "Denylisted codename in base form is forced to PII",
        "payload": {
            "texts": ["Projekt Phoenix on Riigi Infosüsteemi Ameti siseprojekt."],
            "denylist": ["Phoenix"],
            "anonymizers": {
                "DEFAULT": {"type": "replace", "new_value": "[PII]"},
                "DENYLIST_MATCH": {"type": "replace", "new_value": "[SALASTATUD]"},
            },
        },
        "expect_kept": [],
        "expect_removed": ["Phoenix"],
        "expect_in_output": ["[SALASTATUD]"],
    },
    {
        "id": "E04",
        "group": "E. Allow/denylist",
        "desc": "Denylist base form must cover an inflected mention",
        "payload": {
            "texts": ["Projektis Phoenixis osales ka Sotsiaalkindlustusamet."],
            "denylist": ["Phoenix"],
            "anonymizers": {
                "DEFAULT": {"type": "replace", "new_value": "[PII]"},
                "DENYLIST_MATCH": {"type": "replace", "new_value": "[SALASTATUD]"},
            },
        },
        "expect_kept": [],
        "expect_removed": ["Phoenixis"],
        "expect_in_output": ["[SALASTATUD]"],
    },
    {
        "id": "E05",
        "group": "E. Allow/denylist",
        "desc": "Allowlist and denylist together in one request",
        "payload": {
            "texts": [
                "Kalle Kask Maksu- ja Tolliametist juhib projekti Phoenix Tallinnas."
            ],
            "allowlist": ["Maksu- ja Tolliamet", "Tallinn"],
            "denylist": ["Phoenix"],
            "anonymizers": {
                "DEFAULT": {"type": "replace", "new_value": "[PII]"},
                "PERSON": {"type": "replace", "new_value": "[ISIK]"},
                "DENYLIST_MATCH": {"type": "replace", "new_value": "[SALASTATUD]"},
            },
        },
        "expect_kept": ["Maksu- ja Tolliametist", "Tallinnas"],
        "expect_removed": ["Kalle Kask", "Phoenix"],
        "expect_in_output": ["[ISIK]", "[SALASTATUD]"],
    },
    {
        "id": "E06",
        "group": "E. Allow/denylist",
        "desc": "Allowlist entry matching the surface form exactly (mechanism check)",
        "payload": {
            "texts": ["Mari Mets käis Tallinnas Töötukassa vastuvõtul."],
            "allowlist": ["Tallinnas"],
        },
        "expect_kept": ["Tallinnas"],
        "expect_removed": ["Mari Mets"],
    },
    {
        "id": "E07",
        "group": "E. Allow/denylist",
        "desc": "Denylist entry matching the surface form exactly (mechanism check)",
        "payload": {
            "texts": ["Projektis Phoenixis osales ka Sotsiaalkindlustusamet."],
            "denylist": ["Phoenixis"],
            "anonymizers": {
                "DEFAULT": {"type": "replace", "new_value": "[PII]"},
                "DENYLIST_MATCH": {"type": "replace", "new_value": "[SALASTATUD]"},
            },
        },
        "expect_removed": ["Phoenixis"],
        "expect_in_output": ["[SALASTATUD]"],
    },
    {
        "id": "F01",
        "group": "F. Operators",
        "desc": "DEFAULT replace applies to every entity type",
        "payload": {
            "texts": ["Jaan Tamm, isikukood 38001085718, helistas numbril 5551234."],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[VARJATUD]"}},
        },
        "expect_removed": ["Jaan Tamm", "38001085718", "5551234"],
        "expect_in_output": ["[VARJATUD]"],
        "expect_operators": {"replace"},
    },
    {
        "id": "F02",
        "group": "F. Operators",
        "desc": "mask operator with masking_char and chars_to_mask",
        "payload": {
            "texts": ["Isikukood 38001085718 kuulub Jaan Tammele."],
            "anonymizers": {
                "EE_PERSONAL_CODE": {
                    "type": "mask",
                    "masking_char": "X",
                    "chars_to_mask": 7,
                    "from_end": True,
                },
                "PERSON": {"type": "replace", "new_value": "[ISIK]"},
            },
        },
        "expect_removed": ["38001085718"],
        "expect_in_output": ["3800", "XXXXXXX"],
        "expect_operators": {"mask", "replace"},
    },
    {
        "id": "F03",
        "group": "F. Operators",
        "desc": "redact removes the span entirely",
        "payload": {
            "texts": ["Kaebuse esitas Anne Lepik Keskkonnaametile."],
            "anonymizers": {"DEFAULT": {"type": "redact"}},
        },
        "expect_removed": ["Anne Lepik"],
        "expect_operators": {"redact"},
    },
    {
        "id": "F04",
        "group": "F. Operators",
        "desc": "hash with explicit hash_type (md5) - is the parameter forwarded?",
        "payload": {
            "texts": ["Isikukood 38001085718 on registreeritud."],
            "anonymizers": {"EE_PERSONAL_CODE": {"type": "hash", "hash_type": "md5"}},
        },
        "expect_removed": ["38001085718"],
        "expect_operators": {"hash"},
        "check_hash_algo": {"value": "38001085718", "declared": "md5"},
    },
    {
        "id": "F05",
        "group": "F. Operators",
        "desc": "encrypt on one entity, keep on another",
        "payload": {
            "texts": ["Jaan Tamm elab Tallinnas."],
            "anonymizers": {
                "PERSON": {"type": "encrypt", "key": "WmZq4t7w!z%C&F)J"},
                "LOCATION": {"type": "keep"},
                "GPE": {"type": "keep"},
            },
        },
        "expect_kept": ["Tallinnas"],
        "expect_removed": ["Jaan Tamm"],
        "expect_operators": {"encrypt", "keep"},
    },
    {
        "id": "F06",
        "group": "F. Operators",
        "desc": "No anonymizers - YAML default_operators supply Estonian placeholders",
        "payload": {
            "texts": ["Jaan Tamm, isikukood 38001085718, e-post jaan@eesti.ee."],
        },
        "expect_removed": ["Jaan Tamm", "38001085718", "jaan@eesti.ee"],
        "expect_in_output": ["[ISIK]", "[ISIKUKOOD]", "[E-POST]"],
    },
    {
        "id": "F07",
        "group": "F. Operators",
        "desc": "entities filter restricts detection to PERSON only",
        "payload": {
            "texts": ["Jaan Tamm, isikukood 38001085718, helistas numbril 5551234."],
            "entities": ["PERSON"],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[X]"}},
        },
        "expect_kept": ["38001085718", "5551234"],
        "expect_removed": ["Jaan Tamm"],
    },
    {
        "id": "H01",
        "group": "H. Robustness",
        "desc": "Transcript longer than the model's 514-position window keeps finding names",
        "payload": {
            "texts": [
                "Klient: Tere, olen Jaan Tamm.\n"
                + (
                    "Nõustaja selgitas kliendile maksuarvestuse korda ja "
                    "tähtaegu ning viitas juhenditele. "
                )
                * 40
                + "Klient: Minu abikaasa on Mari Mets, tema konto on "
                "EE382200221020145685."
            ],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[PII]"}},
        },
        # The IBAN is regex-detected and survives; the names depend on the
        # transformer, which silently returns nothing once the input exceeds
        # ~512 tokens.
        "expect_removed": ["Jaan Tamm", "Mari Mets", "EE382200221020145685"],
    },
    {
        "id": "H02",
        "group": "H. Robustness",
        "desc": "Denylist match overlapped by an NER span still gets its own operator",
        "payload": {
            "texts": ["Projektis Phoenixis osales ka Sotsiaalkindlustusamet."],
            "denylist": ["Phoenixis"],
            "anonymizers": {
                "DEFAULT": {"type": "keep"},
                "DENYLIST_MATCH": {"type": "redact"},
            },
        },
        # With DEFAULT=keep the overlapping NER hit is left in place, so if the
        # DENYLIST_MATCH result is dropped during conflict resolution the
        # denylisted word leaks verbatim.
        "expect_removed": ["Phoenixis"],
    },
    {
        "id": "H04",
        "group": "H. Robustness",
        "desc": "Names must not be split at subword boundaries (aggregation_strategy)",
        "payload": {
            "texts": [
                "Kaebuse esitasid Ksenia Grigorjeva, Hillar Kõrvits ja "
                "Eerik-Niiles Kross."
            ],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[PII]"}},
        },
        # With aggregation_strategy="simple" the transformer emits 'Ee' +
        # 'rik-Ni' + 'iles Kross'; the low-scoring fragments are dropped by the
        # 0.83 threshold and the rest of the name survives in clear text.
        "expect_removed": ["Kross", "Niiles", "Grigorjeva", "Kõrvits"],
    },
    {
        "id": "H03",
        "group": "H. Robustness",
        "desc": "Text of ~2700 chars is windowed, not dropped",
        "payload": {
            "texts": [
                (
                    "Maksu- ja Tolliamet selgitas menetluse korda. "
                    "Klient esitas taotluse ja lisas dokumendid. "
                )
                * 30
                + "Taotluse esitas Kadri Sepp."
            ],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[PII]"}},
        },
        "expect_removed": ["Kadri Sepp"],
    },
    {
        "id": "G01",
        "group": "G. API contract",
        "desc": "Missing 'texts' field is rejected",
        "payload": {"language": "xx"},
        "expect_status": 400,
    },
    {
        "id": "G02",
        "group": "G. API contract",
        "desc": "'texts' as a string instead of an array is rejected",
        "payload": {"texts": "Jaan Tamm"},
        "expect_status": 400,
    },
    {
        "id": "G03",
        "group": "G. API contract",
        "desc": "Empty 'texts' array is rejected",
        "payload": {"texts": []},
        "expect_status": 400,
    },
    {
        "id": "G04",
        "group": "G. API contract",
        "desc": "Language not registered in the loaded config",
        "payload": {"texts": ["Jaan Tamm elab Tallinnas."], "language": "et"},
        "expect_status": 500,
    },
    {
        "id": "G05",
        "group": "G. API contract",
        "desc": "Batch of three texts (incl. empty string) keeps order and arity",
        "payload": {
            "texts": [
                "Jaan Tamm helistas Maksu- ja Tolliametisse.",
                "",
                "Mari Mets saatis kirja aadressile mari@eesti.ee.",
            ],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[X]"}},
        },
        "expect_result_count": 3,
        "expect_per_text": [
            {"removed": ["Jaan Tamm"]},
            {"exact_text": ""},
            {"removed": ["mari@eesti.ee"]},
        ],
    },
    {
        "id": "G06",
        "group": "G. API contract",
        "desc": "Long multi-turn chat transcript stays consistent",
        "payload": {
            "texts": [
                "Klient: Tere, olen Jaan Tamm, isikukood 38001085718.\n"
                "Nõustaja: Tere! Kuidas saan aidata?\n"
                "Klient: Soovin teada oma maksuvõla suurust Maksu- ja Tolliametis.\n"
                "Nõustaja: Teie võlg on 1250,50 eurot, viitenumber 2900123456.\n"
                "Klient: Kas ma saan maksta kontole EE382200221020145685?\n"
                "Nõustaja: Jah. Kinnitus saadetakse aadressile jaan.tamm@gmail.com.\n"
                "Klient: Aitäh! Minu telefon on +372 5555 5555."
            ],
            "anonymizers": {"DEFAULT": {"type": "replace", "new_value": "[PII]"}},
        },
        "expect_removed": [
            "Jaan Tamm",
            "38001085718",
            "EE382200221020145685",
            "jaan.tamm@gmail.com",
            "+372 5555 5555",
        ],
    },
]
