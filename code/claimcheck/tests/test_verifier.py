from textutils import chunk_text, clean_text, quote_in_text
from verifier import finalize_verdict, summarize

PASSAGES = [{"id": "gw#3", "source": "great_wall.txt", "score": 0.5,
             "text": "The wall is not visible to the naked eye from the Moon. Astronauts agree."}]


def raw(label, quote, pid="gw#3", conf=0.9):
    return {"id": 1, "reasoning": "r", "label": label, "evidence_quote": quote, "passage_id": pid, "confidence": conf}


def test_verbatim_quote_keeps_verdict_and_records_source():
    v = finalize_verdict(raw("contradicted", "not visible to the naked eye"), PASSAGES)
    assert v["label"] == "contradicted" and v["source"] == "great_wall.txt" and not v["note"]


def test_quote_match_ignores_case_and_curly_quotes():
    assert quote_in_text("NOT VISIBLE to the naked eye", PASSAGES[0]["text"])
    assert quote_in_text("it\u2019s fine", "It's fine.")


def test_number_formatting_does_not_break_matching():
    assert quote_in_text("height as 8848.86 metres", "official height as 8,848.86 metres.")
    assert not quote_in_text("height as 8848.87 metres", "official height as 8,848.86 metres.")


def test_two_word_quote_is_accepted_by_default():
    p = [{"id": "c#0", "source": "c.txt", "score": 1, "text": "The mission carried a rover named Pragyan."}]
    assert finalize_verdict(raw("supported", "named Pragyan", pid="c#0"), p, min_quote_words=2)["label"] == "supported"


def test_ellipsis_fragments_must_each_match():
    assert quote_in_text("not visible ... from the Moon", PASSAGES[0]["text"])
    assert not quote_in_text("not visible ... from Mars", PASSAGES[0]["text"])


def test_fabricated_quote_is_downgraded():
    v = finalize_verdict(raw("supported", "The wall is clearly visible from the Moon"), PASSAGES)
    assert v["label"] == "not_enough_evidence" and v["model_label"] == "supported"
    assert v["confidence"] == 0.0 and "Downgraded" in v["note"]


def test_too_short_quote_is_downgraded():
    assert finalize_verdict(raw("supported", "the wall"), PASSAGES, min_quote_words=3)["label"] == "not_enough_evidence"


def test_wrong_passage_id_still_found_elsewhere():
    assert finalize_verdict(raw("contradicted", "not visible to the naked eye", pid="zzz#9"), PASSAGES)["passage_id"] == "gw#3"


def test_not_enough_evidence_needs_no_quote():
    assert finalize_verdict(raw("not_enough_evidence", ""), PASSAGES)["label"] == "not_enough_evidence"


def test_missing_verdict_defaults_safely():
    v = finalize_verdict(None, PASSAGES)
    assert v["label"] == "not_enough_evidence" and v["note"]


def test_summary_scores_only_decided_claims():
    results = [{"label": l} for l in ["supported", "supported", "contradicted", "not_enough_evidence"]]
    s = summarize(results)
    assert s["reliability_score"] == 67 and s["rating"] == "Mixed reliability" and s["evidence_coverage"] == 0.75


def test_summary_with_nothing_decided():
    s = summarize([{"label": "not_enough_evidence"}])
    assert s["reliability_score"] is None


def test_clean_text_strips_urls_and_forward_banner():
    out = clean_text("Forwarded many times\nSee https://x.com/a now")
    assert "http" not in out and "Forwarded" not in out


def test_chunk_text_respects_limit():
    chunks = chunk_text("One fact here. Two facts there. Three facts everywhere.", 30)
    assert len(chunks) == 3 and all(len(c) <= 30 for c in chunks)
