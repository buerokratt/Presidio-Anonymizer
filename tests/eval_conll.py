"""Score the running anonymizer API against a CoNLL NER test set.

    uv run python tests/eval_conll.py test_fresh.conll
    uv run python tests/eval_conll.py test_fresh.conll --detok space
    uv run python tests/eval_conll.py test_fresh.conll --json out.json

Written to make the numbers directly comparable with the Estonian Chat-NER
fine-tuning report, which scores token-classification models with entity-level
`seqeval` over the four chat classes (PER, ORG, LOC, GPE).

Two things differ from that setup and both are unavoidable here:

1. The API takes raw text, not pre-tokenized input, so each sentence is
   de-tokenized and every token's character span recorded; predicted character
   spans are then mapped back onto tokens. `--detok` picks the strategy so the
   sensitivity to it can be measured rather than assumed.
2. The service is a pipeline, not a bare model: the NER output passes through
   per-entity score thresholds and the pattern recognizers. Entity types outside
   the four gold classes are mapped to O, exactly as the report does for
   predictions like B-PROD.

Scoring is entity-level exact match on (type, start token, end token), which is
what seqeval's micro/macro averages reduce to for a BIO tagset. `--verify-seqeval`
checks that claim against seqeval itself if it is importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import requests

TIMEOUT = 600

# The API's entity vocabulary is wider than the gold set; everything else is O.
TYPE_MAP = {
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
    "GPE": "GPE",
}

NO_SPACE_BEFORE = set(",.!?:;)]}%") | {"'", "’", "”"}
NO_SPACE_AFTER = set("([{") | {"“"}


def read_conll(path: str) -> list[list[tuple[str, str]]]:
    """Read a CoNLL file into sentences of (token, label)."""
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if stripped.startswith("-DOCSTART-"):
                continue
            parts = stripped.split()
            current.append((parts[0], parts[-1]))
    if current:
        sentences.append(current)
    return sentences


def detokenize(
    tokens: list[str], strategy: str = "smart"
) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild a sentence and return it with each token's character span."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    suppress_space = True

    for token in tokens:
        gap = ""
        if not suppress_space and not (
            strategy == "smart" and token in NO_SPACE_BEFORE
        ):
            gap = " "
        pieces.append(gap + token)
        position += len(gap)
        spans.append((position, position + len(token)))
        position += len(token)
        suppress_space = strategy == "smart" and token in NO_SPACE_AFTER

    return "".join(pieces), spans


def bio_spans(labels: list[str]) -> list[tuple[str, int, int]]:
    """Entity spans as (type, first token, last token + 1)."""
    spans = []
    index = 0
    while index < len(labels):
        label = labels[index]
        if label.startswith("B-"):
            entity = label[2:]
            end = index + 1
            while end < len(labels) and labels[end] == f"I-{entity}":
                end += 1
            spans.append((entity, index, end))
            index = end
        else:
            index += 1
    return spans


def spans_to_bio(
    predictions: list[dict], token_spans: list[tuple[int, int]]
) -> list[str]:
    """Turn predicted character spans into BIO labels over the gold tokens."""
    labels = ["O"] * len(token_spans)

    # Longest first, so a wide span does not overwrite the label of a narrower
    # one that fits inside it.
    for prediction in sorted(
        predictions, key=lambda p: p["end"] - p["start"], reverse=True
    ):
        mapped = TYPE_MAP.get(prediction["entity_type"])
        if mapped is None:
            continue
        covered = [
            i
            for i, (start, end) in enumerate(token_spans)
            if start < prediction["end"] and prediction["start"] < end
        ]
        if not covered:
            continue
        for offset, token_index in enumerate(covered):
            labels[token_index] = f"{'B' if offset == 0 else 'I'}-{mapped}"
    return labels


def score(gold: list[list[str]], predicted: list[list[str]]) -> dict[str, Any]:
    """Entity-level precision, recall and F1, micro and per class."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for gold_labels, predicted_labels in zip(gold, predicted, strict=True):
        gold_set = set(bio_spans(gold_labels))
        predicted_set = set(bio_spans(predicted_labels))
        for entity, _, _ in gold_set:
            support[entity] += 1
        for span in gold_set & predicted_set:
            tp[span[0]] += 1
        for span in predicted_set - gold_set:
            fp[span[0]] += 1
        for span in gold_set - predicted_set:
            fn[span[0]] += 1

    def prf(t: int, f_pos: int, f_neg: int) -> tuple[float, float, float]:
        precision = t / (t + f_pos) if t + f_pos else 0.0
        recall = t / (t + f_neg) if t + f_neg else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return precision, recall, f1

    classes = sorted(set(support) | set(fp))
    per_class = {}
    for entity in classes:
        precision, recall, f1 = prf(tp[entity], fp[entity], fn[entity])
        per_class[entity] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[entity],
            "tp": tp[entity],
            "fp": fp[entity],
            "fn": fn[entity],
        }

    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    gold_classes = [e for e in classes if support[e]]
    macro = (
        sum(per_class[e]["precision"] for e in gold_classes) / len(gold_classes),
        sum(per_class[e]["recall"] for e in gold_classes) / len(gold_classes),
        sum(per_class[e]["f1"] for e in gold_classes) / len(gold_classes),
    )

    return {
        "per_class": per_class,
        "micro": {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
        "macro": {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
        "support": sum(support.values()),
    }


def verify_against_seqeval(
    gold: list[list[str]], predicted: list[list[str]], micro_f1: float
) -> str:
    """Cross-check the scorer against seqeval when it is available."""
    try:
        from seqeval.metrics import f1_score
        from seqeval.scheme import IOB2
    except ImportError:
        return "seqeval not installed - scorer not cross-checked"
    reference = f1_score(gold, predicted, mode="strict", scheme=IOB2)
    delta = abs(reference - micro_f1)
    verdict = "MATCHES" if delta < 1e-9 else f"DIFFERS by {delta:.6f}"
    return f"seqeval strict micro F1 = {reference:.4f} -> {verdict}"


def _overlaps(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    """Whether two token spans intersect."""
    return a[1] < b[2] and b[1] < a[2]


def collect_errors(
    sentences: list[list[tuple[str, str]]],
    gold: list[list[str]],
    predicted: list[list[str]],
) -> dict[str, list[dict]]:
    """Split mistakes into missed, spurious and boundary/type mismatches."""
    errors: dict[str, list[dict]] = {"fn": [], "fp": [], "partial": []}

    for sentence, gold_labels, predicted_labels in zip(
        sentences, gold, predicted, strict=True
    ):
        tokens = [token for token, _ in sentence]
        gold_set = set(bio_spans(gold_labels))
        predicted_set = set(bio_spans(predicted_labels))

        def text(span: tuple[str, int, int], words: list[str] = tokens) -> str:
            return " ".join(words[span[1] : span[2]])

        for span in gold_set - predicted_set:
            hits = [p for p in predicted_set - gold_set if _overlaps(span, p)]
            if hits:
                errors["partial"].append(
                    {
                        "gold_type": span[0],
                        "gold_text": text(span),
                        "pred_type": hits[0][0],
                        "pred_text": text(hits[0]),
                        "sentence": " ".join(tokens),
                    }
                )
            else:
                errors["fn"].append(
                    {
                        "type": span[0],
                        "text": text(span),
                        "sentence": " ".join(tokens),
                    }
                )

        for span in predicted_set - gold_set:
            if not any(_overlaps(span, g) for g in gold_set):
                errors["fp"].append(
                    {
                        "type": span[0],
                        "text": text(span),
                        "sentence": " ".join(tokens),
                    }
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conll")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--language", default="xx")
    parser.add_argument("--detok", choices=("smart", "space"), default="smart")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--verify-seqeval", action="store_true")
    parser.add_argument("--errors", type=int, default=0, help="print N of each kind")
    args = parser.parse_args()

    sentences = read_conll(args.conll)
    texts = []
    token_spans = []
    for sentence in sentences:
        text, spans = detokenize([token for token, _ in sentence], args.detok)
        texts.append(text)
        token_spans.append(spans)

    config = requests.get(f"{args.url}/config", timeout=TIMEOUT).json()
    print(f"file        : {args.conll}")
    print(f"sentences   : {len(sentences)}  tokens: {sum(len(s) for s in sentences)}")
    print(f"detok       : {args.detok}")
    print(f"model       : {config.get('estbert_model')}")
    print(f"threshold   : {config.get('default_score_threshold')}")

    # `keep` leaves the text byte-identical, so item offsets refer to the input.
    response = requests.post(
        f"{args.url}/anonymize",
        json={
            "texts": texts,
            "language": args.language,
            "anonymizers": {"DEFAULT": {"type": "keep"}},
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()["results"]
    if len(results) != len(texts):
        print(f"expected {len(texts)} results, got {len(results)}", file=sys.stderr)
        return 1

    for text, result in zip(texts, results, strict=True):
        if result["text"] != text:
            print("`keep` rewrote the text - offsets unreliable", file=sys.stderr)
            return 1

    gold = [[label for _, label in sentence] for sentence in sentences]
    predicted = [
        spans_to_bio(result["items"], spans)
        for result, spans in zip(results, token_spans, strict=True)
    ]

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

    if args.verify_seqeval:
        print("\n" + verify_against_seqeval(gold, predicted, metrics["micro"]["f1"]))

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": config,
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
