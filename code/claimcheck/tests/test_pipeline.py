import re

from config import load_config, load_prompts, resolve
from llm_client import InvalidModelOutput
from pipeline import ClaimCheckPipeline
from retriever import EvidenceIndex

GREAT_WALL = "The Great Wall of China is visible to the naked eye from the Moon."
EVEREST = "Mount Everest is 8,848.86 metres tall."
PIZZA = "Pizza was invented by aliens on Mars."


class FakeClient:
    model = "fake"

    def __init__(self, responder):
        self.responder, self.stats, self.messages = responder, {"calls": 0, "cache_hits": 0}, []

    def chat_json(self, messages, schema, temperature=0.0, num_predict=0):
        self.stats["calls"] += 1
        self.messages.append(messages)
        return self.responder(messages, schema)


def make_pipeline(responder):
    cfg = load_config()
    index = EvidenceIndex(resolve(cfg["retrieval"]["evidence_dir"]), cfg["retrieval"]["window_sentences"])
    return ClaimCheckPipeline(cfg, load_prompts(cfg), FakeClient(responder), index)


def extraction(claims):
    return {"claims": [{"claim": c, "original_text": ""} for c in claims]}


def verdicts(messages, honest=True):
    """Fake verifier: answers correctly for the Great Wall, fabricates a quote for Everest."""
    user = messages[-1]["content"]
    out = []
    for n, claim in re.findall(r"CLAIM (\d+): (.+)", user):
        if "Great Wall" in claim:
            out.append({"id": int(n), "reasoning": "r", "label": "contradicted", "passage_id": "",
                        "evidence_quote": "not visible to the naked eye from the Moon", "confidence": 0.95})
        else:
            out.append({"id": int(n), "reasoning": "r", "label": "supported", "passage_id": "",
                        "evidence_quote": "Everest stands exactly 8,848.86 metres, officially", "confidence": 0.9})
    return {"verdicts": out}


def responder(messages, schema):
    if "claims" in schema["properties"]:
        return extraction([GREAT_WALL, EVEREST, PIZZA, GREAT_WALL])  # includes a duplicate
    return verdicts(messages)


def test_retrieval_finds_the_right_document_and_ignores_noise():
    idx = EvidenceIndex(resolve("evidence"), 2)
    assert idx.search("Chandrayaan-3 landed near the lunar south pole")[0][0].source == "chandrayaan3.txt"
    assert idx.search("Humans use only 10 percent of their brains")[0][0].source == "brain_myth.txt"
    assert idx.search(PIZZA) == []


def test_retrieved_passages_do_not_overlap():
    hits = EvidenceIndex(resolve("evidence"), 2).search("Chandrayaan-3 launch landing Moon", top_k=5)
    for i, (a, _) in enumerate(hits):
        for b, _ in hits[i + 1:]:
            assert a.source != b.source or a.end <= b.start or b.end <= a.start


def test_end_to_end_with_guardrails():
    pipe = make_pipeline(responder)
    report = pipe.run("some article")
    by_claim = {r["claim"]: r for r in report["claims"]}
    assert len(report["claims"]) == 3                                   # duplicate removed
    wall = by_claim[GREAT_WALL]
    assert wall["label"] == "contradicted" and wall["source"] == "great_wall.txt"
    everest = by_claim[EVEREST]                                         # fabricated quote -> downgraded
    assert everest["label"] == "not_enough_evidence" and everest["model_label"] == "supported"
    assert by_claim[PIZZA]["passages"] == [] and by_claim[PIZZA]["label"] == "not_enough_evidence"
    assert report["stats"]["llm_calls"] == 3                            # extraction + batched verification + 1 quote repair
    assert report["summary"]["counts"]["contradicted"] == 1


def test_claims_without_evidence_cost_no_llm_call():
    pipe = make_pipeline(lambda m, s: extraction([PIZZA]))
    report = pipe.run("x")
    assert report["stats"]["llm_calls"] == 1 and report["claims"][0]["label"] == "not_enough_evidence"


def test_batch_size_controls_number_of_calls():
    pipe = make_pipeline(responder)
    pipe.cfg["verification"]["batch_size"] = 1
    pipe.cfg["verification"]["quote_repair_attempts"] = 0
    pipe.verify_claims([{"claim": GREAT_WALL}, {"claim": EVEREST}])
    assert pipe.client.stats["calls"] == 2


def test_bad_batch_falls_back_to_single_claim_calls():
    def flaky(messages, schema):
        if len(re.findall(r"CLAIM \d+:", messages[-1]["content"])) > 1:
            raise InvalidModelOutput("bad json")
        return verdicts(messages)

    pipe = make_pipeline(flaky)
    results = pipe.verify_claims([{"claim": GREAT_WALL}, {"claim": EVEREST}])
    assert results[0]["label"] == "contradicted" and results[1]["note"].startswith("Downgraded")


def test_extraction_filters_short_claims_and_checks_original_text():
    def r(messages, schema):
        return {"claims": [{"claim": "Too short", "original_text": ""},
                           {"claim": EVEREST, "original_text": "Mount Everest is 8,848.86 metres tall"},
                           {"claim": GREAT_WALL, "original_text": "invented sentence not in the input"}]}

    claims = make_pipeline(r).extract_claims("Mount Everest is 8,848.86 metres tall. Other text here.")
    assert [c["claim"] for c in claims] == [EVEREST, GREAT_WALL]
    assert claims[0]["original_text"] and claims[1]["original_text"] is None


def test_prompts_include_system_few_shot_and_user_turns():
    pipe = make_pipeline(responder)
    msgs = pipe._messages("verification", "USER TEXT")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[-1]["content"] == "USER TEXT"


def repairable(good_quote):
    """Fake verifier that paraphrases the quote first, then copies it exactly once asked to correct itself."""
    def r(messages, schema):
        if "claims" in schema["properties"]:
            return extraction([GREAT_WALL])
        user = messages[-1]["content"]
        quote = good_quote if "CORRECTION NEEDED" in user else "the wall can't be seen by the naked eye"
        return {"verdicts": [{"id": 1, "reasoning": "r", "label": "contradicted", "passage_id": "",
                              "evidence_quote": quote, "confidence": 0.9}]}
    return r


def test_quote_repair_recovers_a_paraphrased_quote():
    pipe = make_pipeline(repairable("not visible to the naked eye from the Moon"))
    result = pipe.run("x")["claims"][0]
    assert result["label"] == "contradicted" and result["model_label"] == "contradicted"
    assert result["source"] == "great_wall.txt" and "corrected on retry" in result["note"]
    assert pipe.client.stats["calls"] == 3


def test_quote_repair_that_still_fails_stays_downgraded_and_shows_the_quote():
    result = make_pipeline(repairable("still not in the passages")).run("x")["claims"][0]
    assert result["label"] == "not_enough_evidence" and result["model_label"] == "contradicted"
    assert result["note"].startswith("Downgraded after retry") and "can't be seen" in result["note"]


def test_model_can_withdraw_verdict_on_retry():
    def r(messages, schema):
        if "claims" in schema["properties"]:
            return extraction([GREAT_WALL])
        retry = "CORRECTION NEEDED" in messages[-1]["content"]
        return {"verdicts": [{"id": 1, "reasoning": "r", "label": "not_enough_evidence" if retry else "contradicted",
                              "passage_id": "", "evidence_quote": "" if retry else "made up words here", "confidence": 0.5}]}
    result = make_pipeline(r).run("x")["claims"][0]
    assert result["label"] == "not_enough_evidence" and result["model_label"] == "contradicted"
    assert "withdrawn" in result["note"]


def test_repair_can_be_switched_off():
    pipe = make_pipeline(repairable("not visible to the naked eye from the Moon"))
    pipe.cfg["verification"]["quote_repair_attempts"] = 0
    assert pipe.run("x")["claims"][0]["label"] == "not_enough_evidence" and pipe.client.stats["calls"] == 2


# ---- Wikipedia integration -------------------------------------------------------------
from retriever import Passage          # noqa: E402
from wiki_source import WikipediaError  # noqa: E402

MODI_CLAIM = "Narendra Modi is the president of the United States."
MODI_PASSAGE = Passage("wiki:Narendra_Modi#0", "Wikipedia: Narendra Modi",
                       "Narendra Modi is an Indian politician who has served as the prime minister of India since 2014.",
                       0, 1, "https://en.wikipedia.org/wiki/Narendra_Modi")


class FakeWiki:
    def __init__(self, passages=(), error=None):
        self.passages, self.error, self.calls = list(passages), error, 0

    def passages_for(self, claim, queries=None):
        self.calls += 1
        self.last_queries = queries
        if self.error:
            raise self.error
        return self.passages


def modi_responder(messages, schema):
    if "claims" in schema["properties"]:
        return extraction([MODI_CLAIM])
    return {"verdicts": [{"id": 1, "reasoning": "He is India's PM.", "label": "contradicted",
                          "passage_id": "wiki:Narendra_Modi#0",
                          "evidence_quote": "has served as the prime minister of India", "confidence": 0.9}]}


def test_wikipedia_passage_is_retrieved_verified_and_cited():
    pipe = make_pipeline(modi_responder)
    pipe.wiki = FakeWiki([MODI_PASSAGE])
    result = pipe.run("x")["claims"][0]
    assert result["label"] == "contradicted" and result["source"] == "Wikipedia: Narendra Modi"
    assert result["passages"][0]["url"].startswith("https://en.wikipedia.org/")


def test_claim_outside_local_notes_has_no_evidence_without_wikipedia():
    result = make_pipeline(modi_responder).run("x")["claims"][0]
    # weakly related local passages may be retrieved, but nothing about Modi exists, so no verdict is possible
    assert result["label"] == "not_enough_evidence"
    assert not any("Modi" in p["text"] for p in result["passages"])


def test_wikipedia_outage_falls_back_to_local_notes_with_one_warning():
    pipe = make_pipeline(lambda m, s: extraction([GREAT_WALL, EVEREST]) if "claims" in s["properties"] else verdicts(m))
    pipe.wiki = FakeWiki(error=WikipediaError("cannot reach Wikipedia: timeout"))
    report = pipe.run("x")
    assert pipe.wiki.calls == 1                                   # no repeated timeouts
    assert len(report["warnings"]) == 1 and "local evidence only" in report["warnings"][0]
    assert {r["claim"]: r["label"] for r in report["claims"]}[GREAT_WALL] == "contradicted"


def test_extra_passages_are_ranked_with_local_notes():
    idx = EvidenceIndex(resolve("evidence"), 2)
    assert idx.search("Narendra Modi is the president", extra=[MODI_PASSAGE])[0][0].source == "Wikipedia: Narendra Modi"
    assert idx.search("Great Wall visible from the Moon", extra=[MODI_PASSAGE])[0][0].source == "great_wall.txt"


def test_extractor_queries_reach_wikipedia_and_are_reported():
    def r(messages, schema):
        if "claims" in schema["properties"]:
            return {"claims": [{"claim": MODI_CLAIM, "original_text": "", "search_queries": ["Narendra Modi", "President of the United States", ""]}]}
        return modi_responder(messages, schema)

    pipe = make_pipeline(r)
    pipe.wiki = FakeWiki([MODI_PASSAGE])
    result = pipe.run("x")["claims"][0]
    assert pipe.wiki.last_queries == ["Narendra Modi", "President of the United States"]
    assert result["search_queries"] == ["Narendra Modi", "President of the United States"]
