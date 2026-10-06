import pytest
import requests

from llm_client import InvalidModelOutput, OllamaClient, OllamaError
from schemas import EXTRACT_SCHEMA


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def chat(content):
    return FakeResp(payload={"message": {"content": content}})


class FakeSession:
    def __init__(self, posts=(), tags=None):
        self.posts, self.post_calls, self.tags = list(posts), 0, tags

    def post(self, url, json=None, timeout=None):
        self.post_calls += 1
        item = self.posts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, timeout=None):
        if isinstance(self.tags, Exception):
            raise self.tags
        return FakeResp(payload={"models": [{"name": n} for n in self.tags]})


def client(session, **kw):
    return OllamaClient(session=session, backoff=0, **kw)


GOOD = '{"claims": []}'


def test_invalid_json_is_repaired_once():
    s = FakeSession([chat("not json"), chat(GOOD)])
    assert client(s).chat_json([], EXTRACT_SCHEMA) == {"claims": []}
    assert s.post_calls == 2


def test_schema_violation_triggers_repair():
    s = FakeSession([chat('{"wrong": 1}'), chat(GOOD)])
    assert client(s).chat_json([], EXTRACT_SCHEMA) == {"claims": []}


def test_gives_up_after_repair_budget():
    s = FakeSession([chat("nope"), chat("still nope")])
    with pytest.raises(InvalidModelOutput):
        client(s, json_repair_attempts=1).chat_json([], EXTRACT_SCHEMA)


def test_retries_connection_errors_then_succeeds():
    s = FakeSession([requests.ConnectionError("down"), chat(GOOD)])
    assert client(s).chat_json([], EXTRACT_SCHEMA) == {"claims": []}
    assert s.post_calls == 2


def test_server_error_exhausts_retries():
    s = FakeSession([FakeResp(500)] * 3)
    with pytest.raises(OllamaError):
        client(s, max_retries=3).chat_json([], EXTRACT_SCHEMA)
    assert s.post_calls == 3


def test_client_error_is_not_retried():
    s = FakeSession([FakeResp(404, text="model not found")])
    with pytest.raises(OllamaError, match="404"):
        client(s).chat_json([], EXTRACT_SCHEMA)
    assert s.post_calls == 1


def test_cache_avoids_second_call(tmp_path):
    s = FakeSession([chat(GOOD)])
    c = client(s, cache_dir=tmp_path)
    c.chat_json([{"role": "user", "content": "hi"}], EXTRACT_SCHEMA)
    c.chat_json([{"role": "user", "content": "hi"}], EXTRACT_SCHEMA)
    assert s.post_calls == 1 and c.stats["cache_hits"] == 1


def test_check_ready_matches_latest_tag_and_reports_missing():
    client(FakeSession(tags=["llama3.2:latest"]), model="llama3.2").check_ready()
    with pytest.raises(OllamaError, match="ollama pull"):
        client(FakeSession(tags=["mistral:latest"]), model="llama3.2").check_ready()


def test_check_ready_when_server_down():
    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        client(FakeSession(tags=requests.ConnectionError("x"))).check_ready()
