"""Measure verification accuracy on samples/labeled_claims.json (extraction is skipped).

    python evaluate.py            # prints a markdown table you can paste into the README
"""
import argparse
import json
import sys

from config import resolve
from llm_client import OllamaError
from pipeline import ClaimCheckPipeline
from schemas import LABELS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="samples/labeled_claims.json")
    ap.add_argument("--model")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    try:
        data = json.loads(resolve(args.labels).read_text(encoding="utf-8"))
        pipe = ClaimCheckPipeline.from_config(model=args.model, use_cache=not args.no_cache, local_only=True)
        pipe.client.check_ready()
        results = pipe.verify_claims([{"claim": d["claim"], "original_text": None} for d in data])
    except (OllamaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    gold = [d["label"] for d in data]
    pred = [r["label"] for r in results]
    correct = sum(g == p for g, p in zip(gold, pred))
    print(f"\nModel: {pipe.client.model} | claims: {len(data)} | accuracy: {correct}/{len(data)} = {correct / len(data):.0%}\n")
    print("| label | precision | recall | support |\n|---|---|---|---|")
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, pred))
        n_pred, n_gold = pred.count(label), gold.count(label)
        print(f"| {label} | {tp / n_pred if n_pred else 0:.2f} | {tp / n_gold if n_gold else 0:.2f} | {n_gold} |")
    print("\nConfusion (rows = gold, columns = predicted):")
    print("| gold \\ pred | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS))
    for g in LABELS:
        print(f"| {g} | " + " | ".join(str(sum(a == g and b == p for a, b in zip(gold, pred))) for p in LABELS) + " |")
    wrong = [(d["claim"], g, p, r["note"]) for d, g, p, r in zip(data, gold, pred, results) if g != p]
    if wrong:
        print("\nMistakes:")
        for claim, g, p, note in wrong:
            print(f"- {claim}\n    gold={g} predicted={p} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
