"""Score the bare NER model on a CoNLL test set, with no pipeline around it.

    uv run python tests/eval_conll_bare.py test_fresh.conll
    uv run python tests/eval_conll_bare.py test_fresh.conll --agg simple
    uv run python tests/eval_conll_bare.py test_fresh.conll --threshold 0.83

The counterpart to `eval_conll.py`, which scores the running service. Everything
downstream of the model is shared between the two - the same detokenization, the
same character-span-to-token mapping, the same scorer, the same error
categories - so a difference between the two reports is attributable to the
pipeline and to nothing else.

"Bare" means the HF token-classification pipeline output and nothing more: no
score threshold, no span normalization, no pattern recognizers, no allow/deny
list, no overlap resolution. The defaults mirror how the service configures the
model (`aggregation_strategy="first"`), so the two runs share a decode strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

from eval_conll import (
    collect_errors,
    detokenize,
    read_conll,
    score,
    spans_to_bio,
)

# The model emits the four chat classes; eval_conll.TYPE_MAP maps the service's
# vocabulary back onto them, so go the other way here to reuse it unchanged.
MODEL_LABEL_TO_SERVICE = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "GPE": "GPE",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}


def build_pipeline(model_name: str, aggregation: str):  # noqa: ANN201
    """Load the ONNX model exactly as the service does."""
    warnings.filterwarnings("ignore")
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
    model = ORTModelForTokenClassification.from_pretrained(
        model_name, export=True, provider="CPUExecutionProvider"
    )
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation,
        device=-1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conll")
    parser.add_argument("--model", default="buerokrattRIA/xlm-roberta-NER-syntheticGov")
    parser.add_argument(
        "--agg", choices=("first", "simple", "average", "max"), default="first"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="drop predictions below this score; 0.0 keeps everything (bare)",
    )
    parser.add_argument("--detok", choices=("smart", "space"), default="smart")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--errors", type=int, default=0)
    args = parser.parse_args()

    sentences = read_conll(args.conll)
    print(f"file        : {args.conll}")
    print(f"sentences   : {len(sentences)}  tokens: {sum(len(s) for s in sentences)}")
    print(f"detok       : {args.detok}")
    print(f"model       : {args.model}")
    print(f"aggregation : {args.agg}")
    print(f"threshold   : {args.threshold}  (0.0 = bare, nothing dropped)")
    print("pipeline    : none - raw model output only")

    ner = build_pipeline(args.model, args.agg)

    gold = [[label for _, label in sentence] for sentence in sentences]
    predicted = []
    kept = dropped = 0

    for sentence in sentences:
        text, token_spans = detokenize([token for token, _ in sentence], args.detok)
        items = []
        for entity in ner(text):
            group = (
                str(entity.get("entity_group", entity.get("entity", "")))
                .replace("B-", "")
                .replace("I-", "")
            )
            if entity["score"] < args.threshold:
                dropped += 1
                continue
            kept += 1
            items.append(
                {
                    "start": entity["start"],
                    "end": entity["end"],
                    "entity_type": MODEL_LABEL_TO_SERVICE.get(group, group),
                }
            )
        predicted.append(spans_to_bio(items, token_spans))

    print(f"raw spans   : {kept} kept, {dropped} below threshold")

    metrics = score(gold, predicted)
    print(f"\n{'':12s}{'precision':>10} {'recall':>8} {'f1-score':>9} {'support':>8}")
    for entity, row in metrics["per_class"].items():
        print(
            f"{entity:>11s} {row['precision']:9.4f} {row['recall']:8.4f} "
            f"{row['f1']:9.4f} {row['support']:8d}"
        )
    for name in ("micro", "macro"):
        row = metrics[name]
        print(
            f"{name + ' avg':>11s} {row['precision']:9.4f} {row['recall']:8.4f} "
            f"{row['f1']:9.4f} {metrics['support']:8d}"
        )

    errors = collect_errors(sentences, gold, predicted)
    print(
        f"\nerrors: {len(errors['fn'])} missed, {len(errors['fp'])} spurious, "
        f"{len(errors['partial'])} boundary/type"
    )
    for kind, title in (
        ("fn", "MISSED"),
        ("fp", "SPURIOUS"),
        ("partial", "BOUNDARY/TYPE"),
    ):
        for item in errors[kind][: args.errors]:
            if kind == "partial":
                print(
                    f"  {title}: gold {item['gold_type']} {item['gold_text']!r} "
                    f"-> pred {item['pred_type']} {item['pred_text']!r}"
                )
            else:
                print(f"  {title}: {item['type']} {item['text']!r}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": args.model,
                    "aggregation": args.agg,
                    "threshold": args.threshold,
                    "detok": args.detok,
                    "metrics": metrics,
                    "errors": errors,
                    "predictions": predicted,
                    "gold": gold,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nraw results -> {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
