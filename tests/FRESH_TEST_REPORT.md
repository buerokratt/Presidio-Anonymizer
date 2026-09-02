# Presidio pipeline on the fresh (de-contaminated) chat-NER test

Evaluated the **deployed Presidio anonymiser service** on `test_fresh.conll`, the
same 70-sentence / 267-span de-contaminated test set used in the Estonian
Chat-NER fine-tuning experiments, so the numbers sit directly beside Runs 1–11
and the zero-shot baselines.

Run 2026-09-02 against the live container on branch `audit-fixes`
(after the 11 audit fixes), model `buerokrattRIA/xlm-roberta-NER-syntheticGov`,
config `presidio-spacy-estbert.yml`, threshold 0.83 with `ORGANIZATION: 0.45`.

Harness: `tests/eval_conll.py`. Raw output: `tests/fresh_results.json`.

```bash
uv run python tests/eval_conll.py test_fresh.conll --json tests/fresh_results.json
uv run python tests/eval_conll.py test_fresh.conll --detok space   # sensitivity
```

## Headline

```
              precision    recall  f1-score   support

         GPE     0.8276    0.6667    0.7385        72
         LOC     0.9259    0.9434    0.9346        53
         ORG     0.8873    0.8873    0.8873        71
         PER     0.9444    0.9577    0.9510        71

   micro avg     0.8980    0.8577    0.8774       267
   macro avg     0.8963    0.8638    0.8779       267
```

**micro F1 = 0.8774**, bootstrap 95 % CI over the 70 sentences
= **[0.8330, 0.9167]**.

Two things to hold in mind before reading that against the fine-tuning table:

1. **This model was never fine-tuned on the chat training set.** It is
   `xlm-roberta-NER-syntheticGov` used as it ships. Every Run 1–11 number is a
   model fine-tuned on the 341 chat examples.
2. **It is not the same model as baseline Z2.** Z2 is
   `buerokrattRIA/EstBert_NER_SyntheticGov` (EstBERT, 110 M). This is the
   XLM-R sibling of that family, which does not appear anywhere in the previous
   report.

## Against the fine-tuning results

| | model | trained on chat? | fresh micro F1 |
|---|---|---|---|
| — | **Presidio pipeline (this run)** | **no** | **0.8774** |
| R10 | roberta-large-NER | yes | 0.8662 |
| R4 | est-roberta | yes | 0.8523 |
| R7 | xlm-roberta-large | yes | 0.8466 |
| R8 | est-roberta mixed | yes | 0.8460 |
| R5 | xlm-roberta-base | yes | 0.8333 |
| R6 | mDeBERTa-v3-base | yes | 0.8305 |
| R2 | Bürokratt EstBERT + chat | yes | 0.8075 |
| R9 | est-roberta sequential | yes | 0.8037 |
| R11 | rapido backbone | yes | 0.7759 |
| R1 | EstBERT_NER_v2 + chat | yes | 0.7747 |
| R3 | EstBERT fresh head | yes | 0.7574 |
| Z2 | Bürokratt EstBERT zero-shot | no | 0.7344 |
| Z1 | EstBERT_NER_v2 zero-shot | no | 0.6866 |

**The honest reading is not "it wins."** R10 (0.8662) and R4 (0.8523) both fall
inside the confidence interval, so on 267 spans this pipeline is
**statistically indistinguishable from the best fine-tuned models**. What *is*
outside the interval is Z2 at 0.7344 — the pipeline is clearly ahead of the
zero-shot baselines, by **+14.3 F1** over the closest one, without seeing a
single chat training example.

Per class against the two leaders:

| class | Presidio pipeline | R10 | R4 | support |
|---|---|---|---|---|
| PER | 0.951 | **0.958** | 0.925 | 71 |
| ORG | **0.887** | 0.867 | 0.825 | 71 |
| LOC | **0.935** | 0.887 | 0.917 | 53 |
| GPE | 0.739 | **0.754** | 0.750 | 72 |

It leads on ORG and LOC, trails slightly on PER and GPE. GPE is the weak class
for every system in both reports — nothing here changes that.

## Where the score comes from — model vs. pipeline

This is the part worth taking away, because it is not a modelling result. The
same model was scored bare and then with each layer of the deployment added:

| configuration | micro F1 | P | R |
|---|---|---|---|
| bare model, `aggregation="first"`, thr 0.83 | 0.6923 | 0.712 | 0.674 |
| bare model, `aggregation="simple"`, thr 0.83 | 0.8577 | 0.881 | 0.835 |
| model path + `_normalize_span()`, `first` (deployed) | 0.8610 | 0.876 | 0.846 |
| model path + `_normalize_span()`, `simple` | 0.8707 | 0.884 | 0.858 |
| **full pipeline** (+ regex recognizers, Presidio dedup) | **0.8774** | 0.898 | 0.858 |

**`_normalize_span()` is worth +16.6 F1** on top of raw `first` output
(0.6947 → 0.8610). It was written during the audit to stop word-level
aggregation swallowing punctuation and merging names across commas; on this
benchmark that boundary tidying turns out to be the single largest contributor
to the score — larger than any difference between the models in the previous
report.

That has a direct implication for the fine-tuning work: **the Run 1–11 numbers
were measured on raw model output.** If a comparable span-normalisation pass
were applied to R10 or R4 before scoring, they would very likely gain too. The
pipeline's apparent edge here may be post-processing, not modelling — and the
cheapest available improvement to the fine-tuned models is probably not a better
backbone but a boundary-cleanup pass at decode time.

### The aggregation question this raises

Audit fix #2 changed `aggregation_strategy` from `"simple"` to `"first"`,
because `"simple"` fragmented names and leaked part of one
(`Eerik-Niiles Kross` → `[PII]rik-Niiles Kross`). On *this* benchmark
`"simple"` scores 1.0 F1 *higher* than `"first"` once both go through
`_normalize_span()` (0.8707 vs 0.8610).

Both facts are real and they do not conflict, because they measure different
things:

- Exact-span F1 penalises over-capture and under-capture equally.
- An anonymiser does not. Over-capturing redacts a harmless extra word;
  under-capturing publishes part of a name.

For a privacy component, `"first"` is the correct failure direction, and 1.0 F1
is inside the ±2 F1 noise band of a 267-span set. **Keeping `"first"`.** Worth
revisiting only if span exactness ever becomes the product requirement.

## Error analysis — 42 errors over 267 spans

**16 missed, 4 spurious, 22 boundary/type.**

### Missed (16) — 14 of them GPE

```
GPE  Vilnius       GPE  SILLAMÄE     GPE  JÕHVI  (×2)   GPE  Sillamäe
GPE  Stockholm     GPE  KÄRDLA       GPE  STOCKHOLM     GPE  karksi - nuia
GPE  Karksi - Nuia GPE  KOHILA       GPE  räPINA        GPE  tapa
GPE  sillamäe      ORG  tuul         LOC  TÄHE 14
```

Three are foreign cities (`Vilnius`, `Stockholm` ×2) — outside the Estonian
government domain the model was trained for. The rest are small Estonian
towns (`Sillamäe`, `Kärdla`, `Kohila`, `Jõhvi`, `Tapa`, `Räpina`) whose names
the model apparently does not hold, and seven of the sixteen are in ALL-CAPS
sentences.

### Spurious (4) — all four in ALL-CAPS sentences

| type | text | context |
|---|---|---|
| ORG | `MIS TOIMUB` | "MIS TOIMUB , INBANK EI KINNITA…" |
| ORG | `BUSSI` | "…EI LUBANUD MIND BUSSI…" |
| LOC | `PALUN` | "PALUN VÄLJASTAGE TÕEND…" |
| PER | `VÄLJASTAGE` | "PALUN VÄLJASTAGE TÕEND…" |

Uppercase is the model's strongest entity cue, so a shouting message turns
ordinary verbs into entities. Same shape as the `tüüp` / `hämara` / `jääki`
false positives in Runs 1–3, but triggered by casing rather than by slot.

### Boundary / type (22) — the administrative-suffix problem, again

Nine of the 22 are exactly the pattern the previous report identified in every
single fine-tuned model:

```
gold GPE 'Lihula'        -> pred GPE 'Lihula vallas' / 'Lihula vallasse'
gold GPE 'Suure - Jaani' -> pred GPE 'Suure - Jaani vallas'   (×2)
gold GPE 'Loksa'         -> pred GPE 'Loksa vallas'
gold GPE 'kohila'        -> pred GPE 'kohila linna'
gold GPE 'tõstamaa'      -> pred GPE 'tõstamaa vallas'
gold GPE 'karksi - nuia' -> pred GPE 'karksi - nuia vallas'
```

Plus the same trailing case-marked-word inclusions on ORG and LOC
(`SEB` → `SEB PANGAS`, `MAGNUM APTEEK` → `MAGNUM APTEEK POES`,
`LEHE PÕIK 7` → `LEHE PÕIK 7 LINNA`, `Pärna 11` → `Pärna 11 tore`), and three
genuine type confusions (`seb` ORG→PER, `Tuul` ORG→PER,
`JOOSEP REILE` PER→LOC).

**This confirms the report's hypothesis, and the effect is three times larger
than it estimated.** Re-scoring with administrative suffixes normalised on both
sides:

| | micro P | micro R | micro F1 |
|---|---|---|---|
| strict | 0.8980 | 0.8577 | 0.8774 |
| suffix-tolerant | 0.9529 | 0.9101 | **0.9310** |

**+5.4 F1 from convention alone**, against the report's estimated +1–2. More
than half of this pipeline's apparent error is a labelling-convention
disagreement, not a detection failure. The recommendation in the previous
report — decide once whether `linn / vald / vallas / AS / OÜ` belongs inside the
span — is worth considerably more than it was credited for.

## Register breakdown

Casing buckets, assigned objectively (no lowercase → ALL CAPS; no uppercase →
all lowercase; otherwise mixed):

| register | sentences | spans | micro F1 | P | R | GPE F1 |
|---|---|---|---|---|---|---|
| mixed case | 45 | 170 | **0.9129** | 0.933 | 0.894 | 0.771 |
| all lowercase | 15 | 58 | 0.8772 | 0.893 | 0.862 | 0.690 |
| ALL CAPS | 10 | 39 | **0.7200** | 0.750 | 0.692 | 0.667 |

A **19-point spread**, and ALL CAPS is the worst case on both sides of the
ledger: it loses recall (seven of the sixteen missed spans) *and* precision
(all four spurious spans). For a service that takes real chat input, this is
the most actionable weakness in the whole report — an angry citizen writing in
capitals gets meaningfully worse anonymisation than a polite one.

## Method notes and caveats

- **Scorer validated against seqeval.** `tests/eval_conll.py` computes
  entity-level exact-match metrics; cross-checked against
  `seqeval.metrics` with `mode="strict", scheme=IOB2` in a throwaway venv —
  identical to six decimal places (0.877395 both), and identical to seqeval's
  default mode too. So these numbers are on the same footing as the report's.
- **The API takes raw text, not tokens.** Sentences are de-tokenized with token
  character spans recorded, and predicted spans mapped back. This is the one
  structural difference from the previous setup, and it costs about a point
  either way: `--detok smart` (punctuation attached, production-faithful)
  gives 0.8774; `--detok space` (naive join, closer to feeding tokens) gives
  **0.8880**. The headline uses the lower, more production-like number.
- **Boundary snapping is not inflating the score.** Mapping character spans to
  tokens could turn near-misses into exact matches; measured, only **2 of 255**
  predicted spans needed snapping (0.8 %).
- **Entity types outside the four gold classes are mapped to O**, matching the
  report's treatment of `B-PROD` and friends. So the pipeline's regex
  recognizers (isikukood, IBAN, phone, e-mail, …) are invisible to this metric
  even though they are a large part of what it actually anonymises.
- **267 spans is small.** The 95 % CI is ±4 F1. Differences under ~2 F1 between
  any two systems in these tables should not be read as real.

## What I would do next

1. **Fix ALL-CAPS handling** — the largest concrete win available. Case
   normalisation before inference (lowercase or truecase the input, map spans
   back) would probably recover most of the 19-point register gap without
   touching the model. Cheap to test with the existing harness.
2. **Settle the suffix convention** — worth a measured +5.4 F1 here, more than
   any model swap in either report. This is a data decision, not a modelling
   one.
3. **Apply span normalisation to the fine-tuned models before comparing them.**
   `_normalize_span()` is worth +16.6 F1 on this pipeline; the Run 1–11 numbers
   have no equivalent pass. Until that is equalised, the ranking in the previous
   report may be partly measuring which model happens to emit tidy boundaries.
4. **Add a GPE gazetteer** for Estonian towns and common foreign cities. Eleven
   of the sixteen misses are place names, and a pattern recognizer at 0.9 would
   catch them deterministically — the same fix that made `IP_ADDRESS` reachable
   in the audit.
5. **Consider fine-tuning this model.** It reaches parity with the best
   fine-tuned systems *zero-shot*. Z2's fine-tune gained +7.3 F1 on the original
   test; if a comparable gain holds here, `xlm-roberta-NER-syntheticGov`
   fine-tuned on the 341 chat examples is the most promising candidate not yet
   in the experiment table.
