"""Command line interface:  python cli.py --file samples/article_1_whatsapp_forward.txt"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

from llm_client import OllamaError
from pipeline import ClaimCheckPipeline

TAG = {"supported": "SUPPORTED", "contradicted": "CONTRADICTED", "not_enough_evidence": "NOT ENOUGH EVIDENCE"}


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\nClaimCheck report  (model: {report['model']})\n" + "=" * 60)
    for r in report["claims"]:
        print(f"\n[{r['id']}] {TAG[r['label']]}  (confidence {r['confidence']:.2f})")
        print(textwrap.fill(r["claim"], 78, initial_indent="    ", subsequent_indent="    "))
        if r["evidence_quote"]:
            print(textwrap.fill(f"evidence: \"{r['evidence_quote']}\" ({r['source']})", 78,
                                initial_indent="    ", subsequent_indent="              "))
        if r["reasoning"]:
            print(textwrap.fill(f"why: {r['reasoning']}", 78, initial_indent="    ", subsequent_indent="         "))
        if r["note"]:
            print(f"    note: {r['note']}")
    print("\n" + "=" * 60)
    for w in report.get("warnings", []):
        print(f"warning: {w}")
    score = "n/a" if s["reliability_score"] is None else f"{s['reliability_score']}/100"
    c = s["counts"]
    print(f"Reliability: {score} - {s['rating']}")
    print(f"Claims: {s['total_claims']} | supported {c['supported']} | contradicted {c['contradicted']} "
          f"| not enough evidence {c['not_enough_evidence']}")
    st = report["stats"]
    print(f"LLM calls: {st['llm_calls']} (cache hits {st['cache_hits']}) | {st['elapsed_seconds']}s")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Extract claims from text and verify them against local evidence.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", help="path to a text file to check")
    src.add_argument("--text", help="text to check (otherwise read from stdin)")
    ap.add_argument("--model", help="Ollama model (overrides config.yaml)")
    ap.add_argument("--config", help="path to an alternative config file")
    ap.add_argument("--json", action="store_true", help="print the raw JSON report")
    ap.add_argument("--no-cache", action="store_true", help="ignore the on-disk LLM cache")
    ap.add_argument("--local-only", action="store_true", help="use only the notes in evidence/, not Wikipedia")
    args = ap.parse_args(argv)
    try:
        text = args.text or (Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read())
        if not text.strip():
            ap.error("no input text (use --file, --text, or pipe text in)")
        pipe = ClaimCheckPipeline.from_config(args.config, model=args.model, use_cache=not args.no_cache,
                                             local_only=args.local_only)
        pipe.client.check_ready()
        report = pipe.run(text)
    except (OllamaError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
