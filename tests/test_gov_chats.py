"""Black-box test harness for the Estonian Presidio anonymizer API.

Run against a live container:

    uv run python tests/test_gov_chats.py                       # default http://localhost:8000
    uv run python tests/test_gov_chats.py --url http://host:8000
    uv run python tests/test_gov_chats.py --only A01,B03
    uv run python tests/test_gov_chats.py --json tests/results.json

Detection cases are scored by sending the text with a global `keep` operator:
`keep` leaves the text byte-identical, so the returned item offsets and surface
strings refer to the ORIGINAL text and can be compared against gold spans.
(With any other operator, Presidio's offsets refer to the rewritten text.)

Exit code is 0 if every behaviour/contract case passes, 1 otherwise. Detection
quality is reported as metrics rather than pass/fail, since a statistical model
is never expected to score 100%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from typing import Any

import requests

from gov_chat_cases import BEHAVIOUR_CASES, DETECTION_CASES

TIMEOUT = 120


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def post_anonymize(url: str, payload: dict) -> tuple[int, Any, float]:
    """POST /anonymize, returning (status, parsed body or raw text, seconds)."""
    started = time.perf_counter()
    resp = requests.post(f"{url}/anonymize", json=payload, timeout=TIMEOUT)
    elapsed = time.perf_counter() - started
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return resp.status_code, body, elapsed


def locate(text: str, surface: str, occurrence: int = 0) -> tuple[int, int]:
    """Char span of the nth occurrence of `surface` in `text`."""
    idx = -1
    for _ in range(occurrence + 1):
        idx = text.find(surface, idx + 1)
        if idx == -1:
            raise AssertionError(f"gold surface {surface!r} not found in text")
    return idx, idx + len(surface)


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ----------------------------------------------------------------------
# detection scoring
# ----------------------------------------------------------------------
def score_detection_case(url: str, case: dict) -> dict:
    """Send one detection case and match detected spans against gold spans."""
    text = case["text"]
    status, body, elapsed = post_anonymize(
        url, {"texts": [text], "anonymizers": {"DEFAULT": {"type": "keep"}}}
    )
    if status != 200:
        return {
            "id": case["id"],
            "group": case["group"],
            "desc": case["desc"],
            "error": f"HTTP {status}: {body}",
            "elapsed": elapsed,
        }

    result = body["results"][0]
    # `keep` must not rewrite the text - if it does, offsets are meaningless.
    text_unchanged = result["text"] == text
    detected = [
        {
            "span": (item["start"], item["end"]),
            "type": item["entity_type"],
            "surface": item["text"],
        }
        for item in result["items"]
    ]

    gold = []
    for entry in case["gold"]:
        surface, types = entry[0], entry[1]
        occurrence = entry[2] if len(entry) > 2 else 0
        gold.append(
            {
                "span": locate(text, surface, occurrence),
                "types": set(types),
                "surface": surface,
            }
        )

    traps = [
        {"span": locate(text, surface), "surface": surface}
        for surface in case.get("traps", [])
    ]

    # Greedy one-to-one matching: exact span+type first, then overlap+type.
    used_det: set[int] = set()
    for g in gold:
        g["match"] = None
        g["mode"] = None
    for mode in ("exact", "overlap"):
        for g in gold:
            if g["match"] is not None:
                continue
            for i, d in enumerate(detected):
                if i in used_det or d["type"] not in g["types"]:
                    continue
                hit = (
                    d["span"] == g["span"]
                    if mode == "exact"
                    else overlaps(d["span"], g["span"])
                )
                if hit:
                    g["match"] = d
                    g["mode"] = mode
                    used_det.add(i)
                    break

    # Type confusion: gold span covered, but with a label outside the accepted set.
    for g in gold:
        if g["match"] is None:
            wrong = [
                d["type"]
                for i, d in enumerate(detected)
                if i not in used_det and overlaps(d["span"], g["span"])
            ]
            g["wrong_labels"] = sorted(set(wrong))

    unmatched = [d for i, d in enumerate(detected) if i not in used_det]
    trap_hits = [
        {"surface": t["surface"], "as": d["type"], "detected": d["surface"]}
        for t in traps
        for d in unmatched
        if overlaps(d["span"], t["span"])
    ]

    return {
        "id": case["id"],
        "group": case["group"],
        "desc": case["desc"],
        "text": text,
        "elapsed": elapsed,
        "text_unchanged": text_unchanged,
        "detected": [
            {"surface": d["surface"], "type": d["type"], "span": list(d["span"])}
            for d in detected
        ],
        "gold": [
            {
                "surface": g["surface"],
                "types": sorted(g["types"]),
                "matched": g["match"] is not None,
                "mode": g["mode"],
                "got_type": g["match"]["type"] if g["match"] else None,
                "got_surface": g["match"]["surface"] if g["match"] else None,
                "wrong_labels": g.get("wrong_labels", []),
            }
            for g in gold
        ],
        "false_positives": [
            {"surface": d["surface"], "type": d["type"]} for d in unmatched
        ],
        "trap_hits": trap_hits,
    }


# ----------------------------------------------------------------------
# behaviour scoring
# ----------------------------------------------------------------------
def run_behaviour_case(url: str, case: dict) -> dict:
    status, body, elapsed = post_anonymize(url, case["payload"])
    failures: list[str] = []
    notes: list[str] = []

    expected_status = case.get("expect_status", 200)
    if status != expected_status:
        failures.append(f"expected HTTP {expected_status}, got {status}: {body!r:.200}")
        return {
            "id": case["id"],
            "group": case["group"],
            "desc": case["desc"],
            "status": status,
            "elapsed": elapsed,
            "passed": False,
            "failures": failures,
            "notes": notes,
            "output": body if isinstance(body, str) else json.dumps(body)[:400],
        }
    if expected_status != 200:
        return {
            "id": case["id"],
            "group": case["group"],
            "desc": case["desc"],
            "status": status,
            "elapsed": elapsed,
            "passed": True,
            "failures": [],
            "notes": [f"rejected as expected: {json.dumps(body)[:160]}"],
            "output": json.dumps(body)[:400],
        }

    results = body["results"]
    if "expect_result_count" in case and len(results) != case["expect_result_count"]:
        failures.append(
            f"expected {case['expect_result_count']} results, got {len(results)}"
        )

    joined = "\n".join(r["text"] for r in results)
    operators = {item["operator"] for r in results for item in r["items"]}

    failures.extend(
        f"expected to KEEP {surface!r} but it is gone"
        for surface in case.get("expect_kept", [])
        if surface not in joined
    )
    failures.extend(
        f"expected to REMOVE {surface!r} but it survived"
        for surface in case.get("expect_removed", [])
        if surface in joined
    )
    failures.extend(
        f"expected output to contain {surface!r}"
        for surface in case.get("expect_in_output", [])
        if surface not in joined
    )
    if "expect_operators" in case:
        missing = case["expect_operators"] - operators
        if missing:
            failures.append(
                f"operators {sorted(missing)} never applied (saw {sorted(operators)})"
            )

    for idx, spec in enumerate(case.get("expect_per_text", [])):
        if idx >= len(results):
            break
        out = results[idx]["text"]
        if "exact_text" in spec and out != spec["exact_text"]:
            failures.append(f"text[{idx}] expected {spec['exact_text']!r}, got {out!r}")
        failures.extend(
            f"text[{idx}] still contains {surface!r}"
            for surface in spec.get("removed", [])
            if surface in out
        )

    # Is the declared hash_type actually used, or silently dropped?
    if "check_hash_algo" in case:
        spec = case["check_hash_algo"]
        raw = spec["value"].encode()
        digests = {
            algo: hashlib.new(algo, raw).hexdigest()
            for algo in ("md5", "sha256", "sha512")
        }
        used = [algo for algo, digest in digests.items() if digest in joined]
        if not used:
            notes.append("no recognisable md5/sha256/sha512 digest found in output")
        elif spec["declared"] in used:
            notes.append(f"declared hash_type={spec['declared']} was honoured")
        else:
            notes.append(
                f"declared hash_type={spec['declared']} IGNORED, output used {used[0]}"
            )
            failures.append(
                f"hash_type={spec['declared']} silently dropped (output is {used[0]})"
            )

    return {
        "id": case["id"],
        "group": case["group"],
        "desc": case["desc"],
        "status": status,
        "elapsed": elapsed,
        "passed": not failures,
        "failures": failures,
        "notes": notes,
        "output": joined[:600],
    }


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def compute_metrics(detection_results: list[dict]) -> dict:
    per_entity: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp_exact": 0, "tp_overlap": 0, "fn": 0, "fp": 0}
    )
    totals = {"tp_exact": 0, "tp_overlap": 0, "fn": 0, "fp": 0, "traps": 0}

    for res in detection_results:
        if "error" in res:
            continue
        for g in res["gold"]:
            key = "/".join(g["types"])
            if g["matched"]:
                bucket = "tp_exact" if g["mode"] == "exact" else "tp_overlap"
                per_entity[key][bucket] += 1
                totals[bucket] += 1
            else:
                per_entity[key]["fn"] += 1
                totals["fn"] += 1
        for fp in res["false_positives"]:
            per_entity[fp["type"]]["fp"] += 1
            totals["fp"] += 1
        totals["traps"] += len(res["trap_hits"])

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    tp_strict = totals["tp_exact"]
    tp_relaxed = totals["tp_exact"] + totals["tp_overlap"]
    return {
        "totals": totals,
        "strict": prf(tp_strict, totals["fp"] + totals["tp_overlap"], totals["fn"]),
        "relaxed": prf(tp_relaxed, totals["fp"], totals["fn"]),
        "per_entity": {
            key: {
                **counts,
                **prf(
                    counts["tp_exact"] + counts["tp_overlap"],
                    counts["fp"],
                    counts["fn"],
                ),
            }
            for key, counts in sorted(per_entity.items())
        },
    }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def print_report(
    detection_results: list[dict], behaviour_results: list[dict], metrics: dict
) -> None:
    print("\n" + "=" * 78)
    print("DETECTION CASES (gold-span scoring via the `keep` operator)")
    print("=" * 78)
    for res in detection_results:
        if "error" in res:
            print(f"\n[{res['id']}] {res['desc']}\n  ERROR {res['error']}")
            continue
        found = sum(1 for g in res["gold"] if g["matched"])
        flag = (
            "OK " if found == len(res["gold"]) and not res["false_positives"] else "-- "
        )
        print(f"\n{flag}[{res['id']}] {res['desc']}")
        print(f"    gold {found}/{len(res['gold'])}  |  {res['elapsed'] * 1000:.0f} ms")
        if not res["text_unchanged"]:
            print("    !! `keep` altered the text - offsets unreliable")
        for g in res["gold"]:
            if g["matched"]:
                mark = "=" if g["mode"] == "exact" else "~"
                extra = "" if g["mode"] == "exact" else f" (got {g['got_surface']!r})"
                print(f"      {mark} {g['surface']!r} -> {g['got_type']}{extra}")
            else:
                why = (
                    f"mislabelled as {'/'.join(g['wrong_labels'])}"
                    if g["wrong_labels"]
                    else "MISSED"
                )
                print(
                    f"      x {g['surface']!r} expected {'/'.join(g['types'])}: {why}"
                )
        for fp in res["false_positives"]:
            print(f"      + FP {fp['surface']!r} as {fp['type']}")
        for trap in res["trap_hits"]:
            print(f"      ! TRAP {trap['surface']!r} flagged as {trap['as']}")

    print("\n" + "=" * 78)
    print("BEHAVIOUR / CONTRACT CASES")
    print("=" * 78)
    for res in behaviour_results:
        print(f"\n{'PASS' if res['passed'] else 'FAIL'} [{res['id']}] {res['desc']}")
        for note in res["notes"]:
            print(f"      note: {note}")
        for failure in res["failures"]:
            print(f"      -> {failure}")
        if not res["passed"]:
            print(f"      output: {res['output'][:300]!r}")

    print("\n" + "=" * 78)
    print("METRICS")
    print("=" * 78)
    t = metrics["totals"]
    print(
        f"\ngold spans: {t['tp_exact'] + t['tp_overlap'] + t['fn']}  "
        f"exact {t['tp_exact']}  partial {t['tp_overlap']}  missed {t['fn']}  "
        f"false positives {t['fp']}  trap hits {t['traps']}"
    )
    for label in ("strict", "relaxed"):
        m = metrics[label]
        print(
            f"{label:8s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )
    print(
        f"\n{'entity':28s} {'exact':>5} {'part':>5} {'miss':>5} {'FP':>4} "
        f"{'P':>6} {'R':>6} {'F1':>6}"
    )
    for key, m in metrics["per_entity"].items():
        print(
            f"{key:28s} {m['tp_exact']:5d} {m['tp_overlap']:5d} {m['fn']:5d} "
            f"{m['fp']:4d} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
        )

    passed = sum(1 for r in behaviour_results if r["passed"])
    print(f"\nbehaviour cases: {passed}/{len(behaviour_results)} passed")


def measure_latency(url: str) -> dict:
    """Latency with and without the Vabamorf allow/denylist synthesis path."""
    text = (
        "Jaan Tamm, isikukood 38001085718, pöördus Maksu- ja Tolliametisse "
        "telefonil +372 5555 5555."
    )
    samples: dict[str, list[float]] = {}
    probes = {
        "plain": {"texts": [text]},
        "allowlist_3": {
            "texts": [text],
            "allowlist": ["Tallinn", "Microsoft", "Tartu"],
        },
        "denylist_3": {
            "texts": [text],
            "denylist": ["Phoenix", "Falcon", "Nightingale"],
        },
        "both_10": {
            "texts": [text],
            "allowlist": ["Tallinn", "Tartu", "Narva", "Pärnu", "Viljandi"],
            "denylist": ["Phoenix", "Falcon", "Nightingale", "Griffin", "Hydra"],
        },
        "batch_10": {"texts": [text] * 10},
    }
    for name, payload in probes.items():
        timings = []
        for _ in range(3):
            _, _, elapsed = post_anonymize(url, payload)
            timings.append(elapsed)
        samples[name] = timings
    return {
        name: {
            "min": min(v),
            "median": sorted(v)[len(v) // 2],
            "max": max(v),
        }
        for name, v in samples.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--json", dest="json_path", help="write raw results here")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    wanted = set(args.only.split(",")) if args.only else None

    health = requests.get(f"{args.url}/health", timeout=TIMEOUT)
    health.raise_for_status()
    config = requests.get(f"{args.url}/config", timeout=TIMEOUT).json()
    print(f"target      : {args.url}")
    print(f"model       : {config.get('estbert_model')}")
    print(f"nlp engine  : {config.get('nlp_engine')}")
    print(f"languages   : {config.get('supported_languages')}")
    print(f"threshold   : {config.get('default_score_threshold')}")

    detection_results = [
        score_detection_case(args.url, case)
        for case in DETECTION_CASES
        if wanted is None or case["id"] in wanted
    ]
    behaviour_results = [
        run_behaviour_case(args.url, case)
        for case in BEHAVIOUR_CASES
        if wanted is None or case["id"] in wanted
    ]
    metrics = compute_metrics(detection_results)
    latency = {} if args.skip_latency else measure_latency(args.url)

    print_report(detection_results, behaviour_results, metrics)
    if latency:
        print("\nLATENCY (3 runs each, seconds)")
        for name, m in latency.items():
            print(
                f"  {name:14s} min={m['min']:.3f} median={m['median']:.3f} max={m['max']:.3f}"
            )

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config": config,
                    "detection": detection_results,
                    "behaviour": behaviour_results,
                    "metrics": metrics,
                    "latency": latency,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nraw results -> {args.json_path}")

    return 0 if all(r["passed"] for r in behaviour_results) else 1


if __name__ == "__main__":
    sys.exit(main())
