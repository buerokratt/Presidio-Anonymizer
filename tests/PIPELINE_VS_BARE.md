# Pipeline vs. bare model on `test_fresh.conll`

Two measurements of the same NER model on the same 70 sentences / 267 spans:
once as the deployed service exposes it, once as raw model output. Everything
downstream of the model is shared between the two runs — the same
detokenization, the same character-span-to-token mapping, the same scorer, the
same error categories — so any difference is attributable to the pipeline and to
nothing else.

Run 2026-09-02. Model `buerokrattRIA/xlm-roberta-NER-syntheticGov`
(XLM-R large, ~560 M, 9-label PER/ORG/LOC/GPE head), branch `audit-fixes`.

| | Test 1 — pipeline | Test 2 — bare model |
|---|---|---|
| harness | `tests/eval_conll.py` | `tests/eval_conll_bare.py` |
| what runs | live service over HTTP | HF token-classification pipeline |
| aggregation | `first` | `first` (same) |
| score threshold | 0.83, `ORGANIZATION` 0.45 | none — nothing dropped |
| span normalization | yes (`_normalize_span`) | no |
| pattern recognizers | yes | no |
| overlap resolution | yes (Presidio) | no |

```bash
uv run python tests/eval_conll.py      test_fresh.conll --json tests/fresh_pipeline.json
uv run python tests/eval_conll_bare.py test_fresh.conll --json tests/fresh_bare.json
```

---

## Test 1 — inside the pipeline

```
              precision    recall  f1-score   support

         GPE     0.8276    0.6667    0.7385        72
         LOC     0.9259    0.9434    0.9346        53
         ORG     0.8873    0.8873    0.8873        71
         PER     0.9444    0.9577    0.9510        71

   micro avg     0.8980    0.8577    0.8774       267
   macro avg     0.8963    0.8638    0.8779       267
```

**42 errors:** 16 missed, 4 spurious, 22 boundary/type.

## Test 2 — bare model

```
              precision    recall  f1-score   support

         GPE     0.4286    0.4167    0.4225        72
         LOC     0.7377    0.8491    0.7895        53
         ORG     0.7639    0.7746    0.7692        71
         PER     0.8219    0.8451    0.8333        71

   micro avg     0.6884    0.7116    0.6998       267
   macro avg     0.6880    0.7214    0.7036       267
```

284 raw spans emitted. **85 errors:** 1 missed, 8 spurious, 76 boundary/type.

## Head to head

| | bare | pipeline | Δ |
|---|---|---|---|
| micro F1 | 0.6998 | **0.8774** | **+0.1776** |
| micro precision | 0.6884 | **0.8980** | +0.2096 |
| micro recall | 0.7116 | **0.8577** | +0.1461 |
| PER F1 | 0.8333 | **0.9510** | +0.118 |
| ORG F1 | 0.7692 | **0.8873** | +0.118 |
| LOC F1 | 0.7895 | **0.9346** | +0.145 |
| GPE F1 | 0.4225 | **0.7385** | +0.316 |
| total errors | 85 | **42** | −43 |

The pipeline is ahead on every class and on both precision and recall.
**But the headline number is not the finding** — the next section is.

---

## The control that reframes it

Almost all 76 bare-model boundary errors are one thing: the span includes the
following punctuation mark.

```
gold GPE 'Sindi'            -> pred GPE 'Sindi .'
gold ORG 'Hint & Pojad OÜ'  -> pred ORG 'Hint & Pojad OÜ ,'
gold PER 'Tarmo Lepik'      -> pred PER 'Tarmo Lepik .'
gold GPE 'Tapa'             -> pred GPE 'Tapa .'
```

Since the API takes raw text, sentences have to be rebuilt from the gold tokens,
and the choice of how to rebuild them decides whether a name and its full stop
are adjacent. So the same two tests were run under both strategies:

| | natural punctuation<br>`Sindi.` | space-separated<br>`Sindi .` | sensitivity |
|---|---|---|---|
| bare model | 0.6998 | 0.8587 | **−15.9 F1** |
| pipeline | 0.8774 | 0.8880 | **−1.1 F1** |
| pipeline advantage | **+17.8** | **+2.9** | |

**The pipeline's advantage is 17.8 F1 on naturally-punctuated text and 2.9 F1
when punctuation is held away from the entity.** The large gap is real but it
is not the pipeline detecting more; it is the bare model losing 15.9 F1 to
punctuation adjacency, which the pipeline is immune to.

Stated the way it should be stated: **the pipeline's contribution is robustness,
not accuracy.** The model finds roughly the same entities either way — what the
pipeline adds is that the answer stops depending on whether a name happens to
sit next to a comma. Production input is naturally punctuated, so +17.8 is the
operationally relevant figure, but +2.9 is the honest measure of added detection
quality.

## Where the 43-error difference comes from

The two error profiles are almost mirror images:

| | bare | pipeline |
|---|---|---|
| missed entirely | **1** | 16 |
| spurious | 8 | **4** |
| boundary/type | 76 | **22** |

- **The bare model misses almost nothing** — 1 of 267 spans. It locates the
  entities; it just gets their edges wrong.
- **The pipeline fixes 47 spans outright.** Counting exact matches gained:
  span normalization converts 47 boundary errors into exact hits
  (`'Sindi .'` → `Sindi`, `'Hint & Pojad OÜ ,'` → `Hint & Pojad OÜ`).
- **The pipeline loses 8 spans the bare model got right**, all place names, all
  to the score threshold:

```
LOC 'LEHE PÕIK 7'   LOC 'TÄHE 14'      GPE 'JÕHVI'        GPE 'STOCKHOLM'
GPE 'räPINA'        GPE 'Sillamäe'     GPE 'karksi - nuia'  GPE 'tapa'
```

Net: +47 gained, −8 lost, plus 4 fewer spurious.

## The recall cost is worth acting on

For an anonymiser, a missed span is a leaked entity, so the eight losses matter
more than the arithmetic suggests. Their model scores turn out to sit just under
the cut:

```
LOC 'TÄHE 14'        0.749      GPE 'räPINA'          0.778
GPE 'STOCKHOLM'      0.630      GPE 'Sillamäe'        0.674
GPE 'karksi - nuia'  0.613      GPE 'tapa'            0.826
```

And the score distributions show the threshold is doing nothing at all for
people, while cutting into places:

| class | n | min | p25 | median | below 0.83 |
|---|---|---|---|---|---|
| PER | 77 | 0.853 | 0.998 | 0.999 | **0** |
| ORG | 73 | 0.694 | 0.992 | 0.998 | 4 |
| LOC | 63 | 0.439 | 0.989 | 0.997 | 9 |
| GPE | 71 | 0.372 | 0.921 | 0.994 | 13 |

`PER` never falls below 0.83 — its minimum is 0.853 — so the threshold is inert
for the highest-risk class and only bites `GPE` and `LOC`. Sweeping that
threshold:

| GPE/LOC threshold | micro F1 | P | R | missed | spurious |
|---|---|---|---|---|---|
| 0.83 (deployed) | 0.8610 | 0.876 | 0.846 | 16 | 4 |
| 0.75 | 0.8604 | 0.867 | 0.854 | 9 | 4 |
| **0.60** | **0.8641** | 0.859 | **0.869** | **3** | 5 |
| 0.45 | 0.8582 | 0.844 | 0.873 | 1 | 8 |
| 0.30 | 0.8603 | 0.845 | 0.876 | 1 | 9 |

**0.60 is the operating point.** Missed spans fall from 16 to 3 for exactly one
additional spurious span, and F1 edges up rather than down. Thirteen fewer
leaked place names in exchange for one extra over-redaction is a trade an
anonymiser should take every time.

This is the same shape as the `ORGANIZATION: 0.45` change from the audit, and it
is a two-line config edit:

```yaml
entity_score_thresholds:
  ORGANIZATION: 0.45
  GPE: 0.60
  LOCATION: 0.60
```

Not applied — this report only measures.

## Method

- **Identical scoring on both sides.** `eval_conll_bare.py` imports the scorer,
  detokenizer, span mapper and error classifier from `eval_conll.py`, so the two
  tests differ only in what produces the predictions.
- **Entity-level exact match** on (type, start token, end token). The scorer was
  cross-checked against `seqeval` (`mode="strict", scheme=IOB2`) on the pipeline
  run — identical to six decimal places (0.877395 both).
- **Four classes only.** The pipeline emits a wider vocabulary
  (`EE_PERSONAL_CODE`, `IBAN_CODE`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, …); those
  are mapped to `O` so both runs are scored on the gold label set. A large part
  of what the service actually anonymises is therefore invisible to this metric.
- **"Bare" means bare**: HF pipeline output with `aggregation_strategy="first"`
  to match the service's decode setting, no threshold, no span normalization, no
  pattern recognizers, no overlap resolution.
- **267 spans is small.** Bootstrap 95 % CI on the pipeline run is
  [0.8330, 0.9167], so differences under about 2 F1 are not meaningful. The
  17.8-point and 15.9-point effects here are far outside that; the 2.9-point one
  is at its edge and should be treated as directional only.

## Conclusions

1. **The pipeline beats the bare model on every class and every metric**, 0.8774
   vs 0.6998 micro F1 on production-like text.
2. **Most of that gap is punctuation robustness, not detection.** Hold
   punctuation away from the entity and the gap falls to 2.9 F1. The bare model
   swings 15.9 F1 with detokenization; the pipeline swings 1.1.
3. **The model's weakness is boundaries, not recall.** It misses 1 span in 267
   but gets 76 edges wrong. Span normalization fixes 47 of them outright, which
   is where nearly all of the pipeline's advantage comes from.
4. **The pipeline's own weakness is the opposite.** Its thresholds discard 8
   correctly-found place names — the only respect in which the bare model is
   better. `PER` is unaffected, since it never scores below the cut.
5. **Lowering `GPE`/`LOC` to 0.60 is measured, not speculative**: 16 missed → 3,
   for one extra false positive, with F1 unchanged-to-slightly-better.
