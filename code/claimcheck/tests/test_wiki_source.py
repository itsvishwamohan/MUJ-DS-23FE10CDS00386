import pytest
import requests

from wiki_source import WikipediaError, WikipediaSource


class Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class Sess:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def search_payload(*pages):
    return Resp({"query": {"pages": {str(i): {"pageid": i, "title": t, "index": idx, "extract": ex}
                                      for i, (t, idx, ex) in enumerate(pages, 1)}}})


MODI = ("Narendra Modi", 1, "Narendra Modi is an Indian politician. He has served as the prime minister of India since 2014.")
POTUS = ("President of the United States", 2, "The president of the United States is the head of state and head of government.")
FULL = Resp({"query": {"pages": {"1": {"extract": "Intro sentence one. Intro sentence two.\n\n== Early life ==\nHe was born in Gujarat. He grew up in Vadnagar."}}}})
CLAIM = "Narendra Modi became prime minister of India in 2014."


def test_builds_cited_passages_best_page_first_and_drops_headings():
    sess = Sess([search_payload(POTUS, MODI), FULL])  # API order differs from rank order
    passages = WikipediaSource(session=sess).passages_for(CLAIM)
    assert passages[0].source == "Wikipedia: Narendra Modi"          # ranked by the search index
    assert passages[0].id.startswith("wiki:Narendra_Modi#")
    assert passages[0].url == "https://en.wikipedia.org/wiki/Narendra_Modi"
    assert any("born in Gujarat" in p.text for p in passages)       # full text of the top page
    assert not any("==" in p.text for p in passages)
    assert {p.source for p in passages} == {"Wikipedia: Narendra Modi", "Wikipedia: President of the United States"}


def test_responses_are_cached_on_disk(tmp_path):
    sess = Sess([search_payload(MODI), FULL])
    src = WikipediaSource(session=sess, cache_dir=tmp_path)
    first = src.passages_for(CLAIM)
    second = src.passages_for(CLAIM)
    assert len(sess.calls) == 2 and [p.id for p in first] == [p.id for p in second]


def test_falls_back_to_a_shorter_query_when_first_finds_nothing():
    sess = Sess([Resp({"batchcomplete": ""}), search_payload(MODI), FULL])
    assert WikipediaSource(session=sess).passages_for(CLAIM)
    assert len(sess.calls) == 3 and sess.calls[0]["gsrsearch"] != sess.calls[1]["gsrsearch"]


def test_disambiguation_pages_are_skipped():
    page = ("Mercury", 1, "Mercury may refer to: the planet, the element, or the god.")
    assert WikipediaSource(session=Sess([search_payload(page)] * 2)).passages_for(CLAIM) == []


def test_network_failure_raises_wikipedia_error():
    with pytest.raises(WikipediaError, match="cannot reach"):
        WikipediaSource(session=Sess([requests.ConnectionError("down")])).passages_for(CLAIM)


def test_http_error_raises_wikipedia_error():
    with pytest.raises(WikipediaError, match="503"):
        WikipediaSource(session=Sess([Resp({}, 503)])).passages_for(CLAIM)


def test_full_text_failure_falls_back_to_intro():
    sess = Sess([search_payload(MODI), requests.ConnectionError("x")])
    passages = WikipediaSource(session=sess).passages_for(CLAIM)
    assert passages and "prime minister of India" in passages[0].text


def test_extractor_queries_are_searched_separately_and_deduplicated():
    sess = Sess([search_payload(MODI), search_payload(MODI, POTUS), FULL])
    passages = WikipediaSource(session=sess).passages_for(CLAIM, ["Narendra Modi", "President of the United States"])
    assert [c["gsrsearch"] for c in sess.calls[:2]] == ["Narendra Modi", "President of the United States"]
    assert {p.source for p in passages} == {"Wikipedia: Narendra Modi", "Wikipedia: President of the United States"}
    assert len(sess.calls) == 3  # two searches + full text of the best page


def test_falls_back_to_claim_keywords_when_extractor_queries_find_nothing():
    sess = Sess([Resp({}), search_payload(MODI), FULL])
    assert WikipediaSource(session=sess).passages_for(CLAIM, ["Zzzz Qqqq"])
    assert sess.calls[0]["gsrsearch"] == "Zzzz Qqqq" and sess.calls[1]["gsrsearch"] != "Zzzz Qqqq"
