# Detection audit — Estonian Presidio anonymizer

Run 2026-08-19 against a freshly built container at `localhost:8000`, build `0486598`
(branch `model-update`), config `config/presidio-spacy-estbert.yml`
(model `buerokrattRIA/xlm-roberta-NER-syntheticGov`, language `xx`, threshold 0.83).

Suite: `tests/gov_chat_cases.py` + `tests/test_gov_chats.py`, 53 cases
(29 gold-annotated detection cases, 24 behaviour/contract cases) over Estonian
conversations about state agencies. Raw output: `tests/results.json`.

```bash
docker compose up --build -d
uv run python tests/test_gov_chats.py --json tests/results.json
```

## Fix status

Fixes are being applied on branch `audit-fixes`, one finding at a time, each
verified against the rebuilt container before the next is started.

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Long input loses all transformer detections | Leak | ✅ fixed |
| 2 | Names split at subword boundaries | Leak | ✅ fixed |
| 3 | `IP_ADDRESS` never anonymised | Leak | ✅ fixed |
| 4 | Denylist discarded on span overlap | Leak | ✅ fixed |
| 5 | Case-form synthesis is a no-op | Leak | ✅ fixed |
| 6 | All `PERSON` false positives come from spaCy | Precision | ✅ fixed |
| 7 | Global 0.83 threshold cuts agency names | Recall | ✅ fixed |
| 8 | Case endings survive outside the placeholder | Output | ✅ fixed (via 2) |
| 9 | Car-plate regex matches money | Precision | ⏳ in progress |
| 10 | `hash_type` accepted and ignored | Contract | ⬜ not started |
| 11 | In-handler validation returns 500 | Contract | ⬜ not started |

## Headline

| | |
|---|---|
| Gold spans | 86 — 65 exact, 7 partial, 14 missed |
| Strict P / R / F1 | 0.707 / 0.823 / 0.760 |
| Relaxed P / R / F1 | 0.783 / 0.837 / 0.809 |
| False positives | 20 (10 of them `PERSON`) |
| Planted traps fired | 1 of 8 |
| Behaviour cases | 14/24 pass — the 10 failures document the findings below |
| Latency | 110–190 ms single short text; 1.20 s for a batch of 10 |

Structured identifiers score 1.00 across the board. Every failure is in the
transformer path or the glue code around it.

## Method note

Presidio reports anonymisation offsets against the *rewritten* text. To score
detection, every detection case is sent with a single global `keep` operator:
`keep` leaves the text byte-identical, so the returned `start`/`end`/`text`
refer to the original string. Matching is greedy one-to-one, exact-span first
then overlap, so partial hits stay visible as partial hits.

Causes were confirmed by running the two NER sources in isolation inside the
container, so each finding names a line rather than a symptom.

## Findings

### 1. Texts over ~512 tokens lose all transformer detections, silently — LEAK

> **✅ FIXED** — `presidio_flask_estbert.py`. `analyze()` now splits input into
> 400-token windows with 50 tokens of overlap, using the tokenizer's
> `offset_mapping` to cut on real token boundaries, then shifts each window's
> spans back onto the original offsets and drops duplicates from the overlap.
> A whitespace-aligned fallback covers a non-fast tokenizer.
>
> Verified: names are now found at every length tested up to **38,267 chars**
> (9,954 tokens, 29 windows) — previously nothing above ~1,900 chars. Zero
> `Error in EstBERT ONNX analysis` in the log across the whole suite. Case `H01`
> passes; detection metrics unchanged, so no regression on short text.
>
> ```
> filler  chars   start_name  end_name  end_ik  end_iban  NER_ents
> x0        167   True        True      True    True       2
> x40      5247   True        True      True    True      42
> x150    19217   True        True      True    True     152
> x300    38267   True        True      True    True     302
> ```
>
> Not done: the exception is still swallowed rather than surfaced. It can no
> longer fire for length reasons, but any other model failure remains invisible
> to the caller. Worth a follow-up.

XLM-RoBERTa has 514 position embeddings. Longer input makes the ONNX session
throw; the `except Exception` swallows it and `analyze()` returns `[]`. The
request answers **HTTP 200** with no warning, and detection degrades to
spaCy + regex for the *whole* document, not just the tail.

```
ERROR:presidio_flask_estbert.py:140: Error in EstBERT ONNX analysis:
  [ONNXRuntimeError] : INVALID_ARGUMENT : Gather node
  '/roberta/embeddings/position_embeddings/Gather'
  indices element out of data bounds, idx=514 must be within [-514,513]
```

Boundary sweep: 1853 chars → 1 NER entity; 1915 chars → **0**, and nothing above.

The tokenizer is built with `max_length=512` but the pipeline is never called
with `truncation=True`, so nothing truncates (Transformers logs exactly that at
startup). Truncation alone is not the fix — it trades a whole-document failure
for a silently dropped tail.

- **Cause**: `presidio_flask_estbert.py:75-82` (pipeline), error swallowed at `:139-140`
- **Fix**: split into overlapping ≤512-token windows, offset-correct each window's
  spans back to the original string, merge. Let the failure surface instead of
  returning `[]` — a silent empty result is indistinguishable from clean text.
- **Case**: `H01`

### 2. Names split at subword boundaries; fragments fall below threshold — LEAK

> **✅ FIXED** — `presidio_flask_estbert.py`. `aggregation_strategy` is now
> `"first"`, which labels a whole word from its first subword instead of scoring
> fragments separately.
>
> `"first"` alone over-captures, so it needed a companion `_normalize_span()`
> pass. Three regressions it introduced, and what the pass does about them:
>
> | `"first"` alone | after normalization |
> |---|---|
> | `[PII] esitasid [PII] ja [PII]` — two names merged, comma and full stop eaten | `[PII] esitasid [PII], [PII] ja [PII].` |
> | `e-post [ISIK]` — email claimed as PERSON, outranking the email recognizer | `e-post [E-POST]` |
> | `Jaan Tamm,` — span includes the comma | `Jaan Tamm` |
>
> The pass splits each model span on `,;`, strips it back to alphanumeric edges,
> and discards spans containing `@` so the dedicated recognizers keep ownership
> of addresses.
>
> Verified: `[PII] esitasid [PII], [PII] ja [PII].` — every name anonymised, no
> fragment left in clear text. Case `H04` passes.
>
> | | baseline | after |
> |---|---|---|
> | Strict P / R / F1 | 0.707 / 0.823 / 0.760 | **0.802 / 0.849 / 0.825** |
> | Exact / partial spans | 65 / 7 | **73 / 0** |
> | False positives | 20 | 18 |
> | `ORGANIZATION` F1 | 0.708 | 0.735 |
> | `PERSON` F1 | 0.762 | 0.780 |
> | Behaviour cases | 14/24 | **16/24** |
>
> Word-level aggregation also consumes the Estonian case ending, which is what
> finding 8 was about — so that one is fixed here too, and strict now equals
> relaxed because there are no partial spans left.
>
> Known tradeoff: a genuine entity name containing a comma would be split into
> two placeholders. Benign, and no Estonian agency name in the suite is affected.

`aggregation_strategy="simple"` does not merge sentencepiece continuations for
XLM-R. Fragments are scored separately, sub-threshold ones are dropped, and the
rest of the name is published in clear text.

```
in   Kaebuse esitasid Ksenia Grigorjeva, Hillar Kõrvits ja Eerik-Niiles Kross.
out  [PII] esitasid [PII], [PII] ja [PII]rik-Niiles Kross.
```

Same sentence, model in isolation:

```
simple   'Ee'(1.00) 'rik-Ni'(0.65) 'iles Kross'(0.81)   <- two below 0.83
first    'Eerik-Niiles Kross'(1.00)
max      'Eerik-Niiles Kross'(1.00)
```

Switching also lifts borderline organisations: `Transpordiamet` 0.71 under
`simple`, 0.91 under `max` — missed becomes detected.

- **Cause**: `presidio_flask_estbert.py:79`
- **Fix**: `aggregation_strategy="first"` (verified clean on all probe sentences) or `"max"`
- **Case**: `H04`

### 3. `IP_ADDRESS` is never anonymised — LEAK

> **✅ FIXED** — both config files gained an `IpAddress` pattern recognizer at
> 0.9 (0.85 in the stanza config, matching its lower scores), so `IP_ADDRESS` is
> reachable at the configured threshold instead of relying on a 0.6 built-in.
>
> Two details worth keeping:
>
> - The IPv6 alternatives are ordered **longest-first**. Python's `re` returns the
>   first alternative that matches, and the built-in ordering clipped
>   `2001:db8::1` down to `2001:db8::`, leaving the final group in clear text.
> - The patterns are single-quoted YAML so the backslashes stay literal, and use
>   `\b` boundaries rather than the hand-rolled lookarounds used elsewhere in the
>   file — an earlier draft with those lookarounds matched `14:30` and `23:59` as
>   IPv6 addresses, which would have turned every clock time into an IP.
>
> Verified against Estonian text that could collide — clock times, scores,
> money sums, version numbers, dates — with no false positives:
>
> ```
> in   Riigi Infosüsteemi Amet: serveri IP on 192.168.10.24 ja IPv6 2001:db8::1.
> out  [ORGANISATSIOON]: serveri IP on [IP-AADRESS] ja IPv6 [IP-AADRESS].
>
> in   Otsus tehti 12.03.2026, istung toimub 20 aprillil 2026 kell 09:15.
> out  [ISIK] tehti [KUUPÄEV], istung toimub [KUUPÄEV] kell [KUUPÄEV].
> ```
>
> `IP_ADDRESS` F1 0.00 → **1.00** (3/3). Strict F1 0.825 → **0.844**,
> recall 0.849 → **0.884**, missed spans 13 → 10. `DATE_TIME` still 1.00.
>
> Still open, as the original finding recommended: the rest of
> `entities_to_detect` has not been audited entity-by-entity against the 0.83
> cut. `IP_ADDRESS` was the one that showed up in this suite; there may be others
> reachable only through a sub-threshold built-in.

It is in `entities_to_detect`, but its only source is Presidio's built-in
`IpRecognizer`, scoring **0.6** against a **0.83** threshold. Unlike the email,
credit-card and crypto recognizers it has no validator to promote a match to
full score, so it can never clear the bar. All 3 gold IPs missed, including one
directly after the literal word "IP".

- **Fix**: add IPv4/IPv6 patterns to the YAML `recognizers:` list at 0.9 like the
  other Estonian patterns, or pass per-entity thresholds to `analyze()`. Audit
  the rest of `entities_to_detect` the same way — anything relying on a sub-0.83
  built-in without a validator is dead config.
- **Cases**: `A08`, `D07`

### 4. Denylist matches are discarded when another entity overlaps — LEAK

> **✅ FIXED** — `analyze_with_lists()` now resolves denylist spans last and lets
> them win, instead of handing them to the anonymizer to compete on score and
> length.
>
> The obvious implementation — drop any detection that overlaps a denylist span —
> would have introduced a *new* leak: a merged span like `Kalle Kask Phoenixis`
> also protects a real name, and discarding it wholesale would publish that name.
> So `subtract_spans()` cuts the denylist span out and keeps the remainder,
> re-trimming the pieces to alphanumeric edges and dropping only what is left
> empty.
>
> ```
> DEFAULT keep, DENYLIST_MATCH redact, denylist ["Phoenixis"]
> before  Projektis Phoenixis osales ka Sotsiaalkindlustusamet.   <- leaked
> after   Projektis  osales ka Sotsiaalkindlustusamet.
>
> DEFAULT replace, DENYLIST_MATCH [SALASTATUD], denylist ["Phoenix"]
> in      Kalle Kask Phoenixis kirjutas aruande.
> after   [PII] [SALASTATUD] kirjutas aruande.    <- name still protected
> ```
>
> Verified: cases `E03`, `E04`, `E05`, `E07`, `H02` all pass. Behaviour cases
> 19/24 → **22/24**; the only remaining failures are findings 10 and 11.
> Detection metrics unchanged.

`DENYLIST_MATCH` is dropped during Presidio's overlap resolution if any NER span
touches it, despite carrying score 1.0, so the caller's chosen operator for it
never runs. Usually invisible (the overlapping entity is anonymised anyway), but
a leak the moment the two operators differ — the natural "keep everything,
redact secrets" configuration:

```
denylist ["Phoenixis"], DEFAULT keep, DENYLIST_MATCH redact
in     Projektis Phoenixis osales ka Sotsiaalkindlustusamet.
out    Projektis Phoenixis osales ka Sotsiaalkindlustusamet.   <- leaked
spans  [('Sotsiaalkindlustusamet','ORGANIZATION','keep'),
        ('Projektis Phoenixis','PERSON','keep')]               <- no DENYLIST_MATCH

in     Meie plaan Phoenixis on salajane.      (no overlapping NER span)
out    Meie plaan  on salajane.                                <- redacted correctly
```

- **Fix**: resolve denylist spans last and let them win — drop analyzer results
  overlapping a denylist span before calling `anonymize()`. If the promise is
  "always", the code has to enforce it.
- **Cases**: `H02`, `E04`, `E07`

### 5. Case-form synthesis produces nothing for the words it exists to expand — LEAK

> **✅ FIXED** — `utils.py` and `presidio_flask_estbert.py`, in three parts.
>
> **POS tag.** `synthesize_word()` now tries `"H"` (pärisnimi) first for a
> capitalised word and `"S"` first otherwise, falling back to the other tag when
> the first yields nothing. Matching downstream is case-insensitive, so a
> lowercase fallback form is still useful.
>
> **Multi-word entries.** Estonian inflects the head of the phrase, which is the
> last word, so `synthesize_phrase()` inflects only that and keeps the prefix
> fixed. `synthesize_all()` calls it instead of `synthesize_word()`.
>
> ```
> 'Tallinn'                 ->  1 form   ->  14 forms
> 'Phoenix'                 ->  1 form   ->  14 forms
> 'Maksu- ja Tolliamet'     ->  1 form   ->  37 forms
> 'Riigi Infosüsteemi Amet' ->  1 form   ->  37 forms
> ```
>
> **Span containment.** Fixing the above surfaced a second defect in the same
> feature: `apply_allowlist()` compared the *whole* detected span against the
> allowlist, so an allowlisted word inside a wider model span was never spared.
> With `allowlist: ["Tallinn"]`, the span `Phoenix Tallinnas` matched nothing and
> both words were anonymised. It now trims allowlisted terms off either edge of a
> span and drops the span only when nothing is left:
>
> ```
> before  [ISIK] Maksu- ja Tolliametist juhib projekti [PII].
> after   [ISIK] Maksu- ja Tolliametist juhib projekti [PII] Tallinnas.
> ```
>
> Verified: cases `E01`, `E02`, `E05`, `E06` pass. Behaviour cases 16/24 → **19/24**.
> Detection metrics unchanged, as expected — allow/denylists are per-request and
> the detection cases send none.
>
> Known limit: an allowlisted term in the *middle* of a span is not split out,
> only edges are trimmed. No case in the suite hits it.

`synthesize_all()` calls Vabamorf with POS `"S"` (common noun). Capitalised
proper nouns — exactly what allowlists and denylists contain — are not in that
class, so Vabamorf returns `[]` and only the original word comes back.

```
'Tallinn'             -> 1 form  ['Tallinn']
'Phoenix'             -> 1 form  ['Phoenix']
'Maksu- ja Tolliamet' -> 1 form  ['Maksu- ja Tolliamet']

synthesize('Tallinn','sg in','S') -> []
synthesize('Tallinn','sg in','H') -> ['Tallinnas']
'Microsoft' with "H"  -> 14 forms: Microsofti, Microsoftis, Microsoftile, ...
cost: 1.2 ms/word
```

So `allowlist: ["Tallinn"]` does not spare "Tallinnas", and multi-word entries
never match the inflected mentions that actually occur. The Swagger docstring
promises the opposite. The allowlist *mechanism* is fine — the exact-form case
(`E06`) passes.

- **Cause**: `utils.py:41-42` — POS `"S"` should be `"H"`
- **Fix**: pass `"H"`, and synthesise per whitespace-separated token so multi-word
  entries expand. Then re-measure latency: the per-request cost is currently near
  zero only because the feature does nothing.
- **Cases**: `E01`, `E02`, `E04`, `E05`

### 6. Every `PERSON` false positive comes from spaCy, not the Estonian model

> **✅ FIXED** — `SpacyRecognizer` joined `MedicalLicenseRecognizer` in
> `unwanted_recognizers`, so it is dropped from the registry in
> `load_presidio_from_config()`. spaCy remains the `NlpEngine` for tokenisation
> and lemmas; it just no longer contributes entities. `GET /recognizers` confirms
> it is gone.
>
> This was the largest single improvement in the whole set:
>
> | | before | after |
> |---|---|---|
> | Strict P / R / F1 | 0.809 / 0.884 / 0.844 | **0.907 / 0.907 / 0.907** |
> | False positives | 18 | **8** |
> | `PERSON` P / R / F1 | 0.615 / 1.00 / 0.762 | **1.00 / 1.00 / 1.00** |
> | `ORGANIZATION` F1 | 0.735 | 0.784 |
> | Missed spans | 10 | 8 |
>
> Every one of the ten `PERSON` false positives is gone, and the spurious
> `LOCATION` hits disappeared with them. `ORGANIZATION` recall rose too
> (0.72 → 0.80), because spaCy was previously claiming agencies as people and
> winning the overlap.
>
> Ordering mattered: doing this before finding 1 would have made things worse,
> since spaCy was the only thing still finding names once the transformer blanked
> on long input.

All 10 `PERSON` false positives trace to `xx_ent_wiki_sm`:

```
spaCy alone:   'Soovin'->PER 'Palun'->PER 'Jah'->PER 'Otsus'->PER 'Kaebuse'->PER
               'Töötukassas'->PER 'Vabariigi Valitsus'->PER
               'Tallinna Linnavalitsusele'->PER

Estonian model alone, same sentences:
               'Töötukassa'->ORG(0.97)  'Vabariigi Valitsus'->ORG(0.85)
               'Tallinna Linnavalitsus'->ORG(0.89)  'Statistikaamet'->ORG(0.94)
               — no false PERSON hits on any probe sentence
```

Over-anonymisation destroys text, and the agency mislabels mean consumers that
branch on entity type see a person where the record names an institution.

- **Fix**: remove `SpacyRecognizer` from the registry the way
  `MedicalLicenseRecognizer` already is, keeping spaCy as the `NlpEngine` for
  tokenisation only. Do finding 1 first — spaCy is currently the fallback
  masking it.

### 7. One global threshold of 0.83 cuts real agency names

> **✅ FIXED** — the config gained an `entity_score_thresholds:` block, and
> `apply_score_thresholds()` in `analyze_with_lists()` filters per entity type.
> `AnalyzerEngine` only supports one threshold, so it is now built with the
> *lowest* threshold in play and the real filtering happens afterwards.
>
> `ORGANIZATION: 0.45`; everything else keeps 0.83. The number came from
> measuring, not guessing — under the new aggregation the missed agencies score
> `Sotsiaalkindlustusamet` 0.77, `Riigikogu` 0.53, `Transpordiamet` 0.50, while
> the *false* `ORGANIZATION` hits score `Riigilõiv` 0.98 and `RT I` 0.98. The two
> populations do not overlap, so lowering the threshold buys recall without
> admitting those errors.
>
> | | before | after |
> |---|---|---|
> | Strict P / R / F1 | 0.930 / 0.909 / 0.920 | **0.933 / 0.944 / 0.939** |
> | `ORGANIZATION` exact | 20 | **26** |
> | `ORGANIZATION` P / R / F1 | 0.769 / 0.800 / 0.784 | **0.867 / 0.929 / 0.897** |
> | Missed spans | 8 | **5** |
>
> `Päästeamet` is still missed in both cases that use it, at any threshold: the
> model does not predict it as an entity at all, so no amount of tuning recovers
> it. That is a model limitation, not a configuration one — worth reporting
> upstream or adding as a pattern recognizer if these names matter operationally.
>
> Also corrected here: `Rahvastikuregistris` was being counted as a false
> positive when it is a correct detection the gold set had omitted.

The model finds them; they score under the cut, which makes behaviour look
erratic when it is a threshold sitting inside the score distribution.

```
Maksu- ja Tolliameti    0.99 kept     Transpordiamet   0.71 cut
Sotsiaalkindlustusameti 0.98 kept     Riigikogu        0.80 cut
Töötukassa              0.97 kept     Rahvastiku-      0.61 cut
```

- **Fix**: per-entity thresholds — identifiers stay strict, `ORGANIZATION` drops
  to ~0.6. Fixing finding 2 raises these same scores.
- 8 of the 14 misses are `ORGANIZATION`.

### 8. Case endings survive outside the placeholder

> **✅ FIXED** as a side effect of finding 2. Word-level aggregation covers the
> whole word including its case suffix, and `_normalize_span()` keeps the
> surrounding punctuation out of the span.
>
> ```
> before  [ISIK] elab [GPE], kolis sinna [GPE]st ja töötas varem [GPE]s
>         [ORGANISATSIOON]s.
> after   [ISIK] elab [GPE], kolis sinna [GPE] ja töötas varem [GPE]
>         [ORGANISATSIOON].
> ```
>
> All 7 partial span matches became exact; the suite now reports 0 partial spans.

The model tags the stem and leaves the Estonian case suffix behind. All 7 partial
span matches are this shape. No identity leak, but ungrammatical output, and the
trailing morpheme discloses the grammatical case (and with a street name, part of
the address).

```
in   Kadri Sepp elab Tallinnas, kolis sinna Tartust ja töötas varem Narvas
     Politsei- ja Piirivalveametis.
out  [ISIK] elab [GPE], kolis sinna [GPE]st ja töötas varem [GPE]s
     [ORGANISATSIOON]s.
```

- **Fix**: extend spans forward over trailing word characters before anonymising,
  or lemma-align using the EstNLTK analysis already in the dependency set.

### 9. Car-plate regex matches money; plates get labelled as organisations

`[0-9]{2,3}\s?[a-zA-Z]{3}` is also the shape of a currency amount. Meanwhile the
model claims real plates as `ORGANIZATION` and wins the overlap. Worst entity in
the suite: F1 0.40.

```
in     Transpordiamet: sõiduk 123 ABC sai trahvi summas 500 EUR, teine sõiduk 45 XYZ…
spans  ('500 EUR','CAR_NUMBER')  ('123 ABC','ORGANIZATION')  ('45 XYZ','CAR_NUMBER')
```

A personal code is likewise captured as `PHONE_NUMBER` when comma-adjacent —
same over-broad-regex family.

- **Fix**: require an Estonian plate shape; negative lookahead for
  `EUR|USD|EEK|km|kg`.
- **Cases**: `C04`, `A05`, `C01`

### 10. `hash_type` is accepted and ignored

The operator parser forwards params for `replace`, `mask`, `redact`, `encrypt`.
`hash` has no branch, so `hash_type` never reaches `OperatorConfig`. A caller
asking for md5 gets sha256 and no error (64 hex chars in the output).

- **Fix**: add the branch; reject unknown operator types instead of silently
  falling through to `replace`.
- **Case**: `F04`

### 11. In-handler validation errors are reported as 500

An empty `texts` array is rejected, but the `api.abort(400, …)` doing it is
raised inside the route's own `try`, caught by the blanket `except Exception`,
and re-wrapped as a 500 whose body still quotes the original 400. Every
hand-written validation in the handler is affected; the checks that do return
400 do so because Flask-RESTX schema validation runs before the handler.

```
POST /anonymize {"texts": []}  ->  HTTP 500
{"message": "Anonymization failed: 400 Bad Request: The browser (or proxy)
             sent a request that this server could not understand."}
```

- **Fix**: validate before the `try`, or re-raise `HTTPException` ahead of the
  generic handler. A client cannot currently tell a bad request from a server fault.
- **Case**: `G03`

## What holds up

- **Estonian identifiers are exact** — F1 1.00 over 25 spans: personal codes,
  document numbers, IBANs, credit cards, crypto addresses. Correct boundaries,
  zero false positives. These come from YAML patterns at 0.9–0.95.
- **Dates handle real Estonian** — F1 1.00 over 8 spans: `12.03.2026`,
  `15 märts 2026`, `20 aprillil 2026`, clock times.
- **No name missed under the token limit** — `PERSON` recall 1.00 over 16 spans,
  including diacritics (`Kärt Õunapuu`, `Žanna Šišova`), hyphenated names
  (`Mari-Liis Kask-Tamm`) and inflected forms (`Jaan Tammele`).
- **Places and addresses, inflected** — recall 1.00: `Tallinnas`, `Tartust`,
  `Narvas`, `Liivalaia 2`, `Pärnu mnt 42`, split across `LOCATION`/`GPE` as
  configured.
- **Operators behave** — `replace`, `mask`, `redact`, `encrypt`, `keep` all correct,
  including per-entity overrides on `DEFAULT`. With no `anonymizers`, the YAML
  defaults produce proper Estonian placeholders.
- **Batching is honest** — order and arity preserved, empty string passed through,
  a seven-turn transcript fully anonymised.
- **Precision traps mostly held** — money sums, percentages, statute references
  (`§ 32`, `RT I 2002, 26, 150`), court case numbers (`3-21-1234`) and room
  numbers all left alone. Only `500 EUR` fired.

## Per-entity metrics

| Entity | exact | partial | missed | FP | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DATE_TIME | 8 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| EMAIL_ADDRESS | 3 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| EST_ID_DOC | 3 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| IBAN_CODE | 3 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| LOCATION / GPE | 5 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| CREDIT_CARD | 1 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| CRYPTO | 1 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| URL | 1 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| PHONE_NUMBER | 6 | 0 | 0 | 1 | 0.86 | 1.00 | 0.92 |
| EE_PERSONAL_CODE | 5 | 0 | 1 | 0 | 1.00 | 0.83 | 0.91 |
| PERSON | 16 | 0 | 0 | 10 | 0.62 | 1.00 | 0.76 |
| ORGANIZATION | 12 | 5 | 8 | 6 | 0.74 | 0.68 | 0.71 |
| CAR_NUMBER | 1 | 0 | 2 | 1 | 0.50 | 0.33 | 0.40 |
| IP_ADDRESS | 0 | 0 | 3 | 0 | 0.00 | 0.00 | 0.00 |

Latency (3 runs each, median): plain 0.129 s, with allowlist 0.118 s, with
denylist 0.125 s, both+10 words 0.170 s, batch of 10 texts 1.200 s. Allow/denylist
arguments cost nothing today — a symptom of finding 5. A batch of ten costs ten
times one text: `MAX_WORKERS` is set in `docker-compose.yml` but no parallelism
exists in the handler.

## Fix order

1. **Chunk long input** (1). The only failure that voids detection for a whole
   document, and chat transcripts routinely exceed the limit. Until it is fixed,
   no other measurement generalises to production traffic.
2. **`aggregation_strategy` → `"first"`** (2). One word; stops partial names
   reaching the response and lifts borderline org scores.
3. **Vabamorf POS `"S"` → `"H"`** (5). One letter; makes allow/denylist do what
   the docs claim.
4. **Give `IP_ADDRESS` a reachable recognizer** (3), then audit the rest of
   `entities_to_detect` against the 0.83 cut.
5. **Let denylist spans win overlap resolution** (4).
6. **Drop `SpacyRecognizer`** (6) — after 1, since spaCy currently masks it.
   Expect precision to rise sharply.
7. **Re-measure**, then tune per-entity thresholds (7) and span extension (8).

## Repo changes made for this audit

- `tests/gov_chat_cases.py`, `tests/test_gov_chats.py`, `tests/results.json` — new.
- `pyproject.toml` — added `[tool.ruff.lint.per-file-ignores]` allowing `print`
  under `tests/`, so `T201` does not fail CI on a CLI reporter. `ruff check`,
  `ruff format --check` and `pyright` are all clean.

No application code was changed. The 10 failing behaviour cases each document one
finding and become regression tests once fixed.
