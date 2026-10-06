"""Programmatic guardrails around LLM verdicts: quote check, downgrade, report summary."""
from schemas import CONTRADICTED, LABELS, NOT_ENOUGH, SUPPORTED
from textutils import quote_in_text, word_count

DECISIVE = (SUPPORTED, CONTRADICTED)


def find_quote(quote: str, passages: list[dict], preferred_id: str = "", min_words: int = 3):
    """Return the passage that contains `quote` verbatim (tolerant of case/punctuation), else None."""
    if not quote or word_count(quote) < min_words:
        return None
    for passage in sorted(passages, key=lambda p: p["id"] != preferred_id):
        if quote_in_text(quote, passage["text"]):
            return passage
    return None


def finalize_verdict(raw: dict | None, passages: list[dict], min_quote_words: int = 3) -> dict:
    """Turn a raw model verdict into a trusted one. A decisive label needs a verifiable quote."""
    if raw is None:
        return {"label": NOT_ENOUGH, "model_label": None, "confidence": 0.0, "evidence_quote": "",
                "source": "", "passage_id": "", "rejected_quote": "", "reasoning": "The model returned no verdict.",
                "note": "Missing verdict; defaulted to not_enough_evidence."}
    label = raw["label"]
    result = {
        "label": label,
        "model_label": label,
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
        "evidence_quote": "",
        "source": "",
        "passage_id": "",
        "rejected_quote": "",
        "reasoning": (raw.get("reasoning") or "").strip(),
        "note": "",
    }
    if label in DECISIVE:
        quote = (raw.get("evidence_quote") or "").strip()
        hit = find_quote(quote, passages, raw.get("passage_id", ""), min_quote_words)
        if hit is None:
            result.update(label=NOT_ENOUGH, confidence=0.0, rejected_quote=quote,
                          note=f"Downgraded: the model's quote was not found verbatim in the retrieved "
                               f"evidence. It quoted: \"{quote[:120]}\"")
        else:
            result.update(evidence_quote=quote, source=hit["source"], passage_id=hit["id"])
    return result


def summarize(results: list[dict], reliable: int = 80, mixed: int = 40) -> dict:
    counts = {label: 0 for label in LABELS}
    for r in results:
        counts[r["label"]] += 1
    total = len(results)
    decided = counts[SUPPORTED] + counts[CONTRADICTED]
    score = round(100 * counts[SUPPORTED] / decided) if decided else None
    if score is None:
        rating = "Not enough evidence to rate"
    elif score >= reliable:
        rating = "Mostly reliable"
    elif score >= mixed:
        rating = "Mixed reliability"
    else:
        rating = "Unreliable"
    return {"total_claims": total, "counts": counts, "reliability_score": score,
            "rating": rating, "evidence_coverage": round(decided / total, 2) if total else 0.0}
