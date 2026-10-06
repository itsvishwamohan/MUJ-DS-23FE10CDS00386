"""Thin Ollama client: structured JSON output, retries with backoff, JSON repair, disk cache."""
import hashlib
import json
import logging
import time
from pathlib import Path

import requests
from jsonschema import ValidationError, validate

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Ollama is unreachable, misconfigured, or kept failing after retries."""


class InvalidModelOutput(OllamaError):
    """The model replied, but never with JSON that matches the schema."""


class OllamaClient:
    def __init__(self, host="http://localhost:11434", model="llama3.2", timeout=300,
                 max_retries=3, backoff=2.0, keep_alive="10m", seed=42,
                 json_repair_attempts=1, cache_dir=None, session=None):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.keep_alive = keep_alive
        self.seed = seed
        self.json_repair_attempts = json_repair_attempts
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()
        self.stats = {"calls": 0, "cache_hits": 0}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- health -----------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            resp = self.session.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                f"(or open the Ollama app) and try again."
            ) from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def check_ready(self) -> None:
        """Raise a helpful error unless the server is up and the model is pulled."""
        names = self.list_models()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            installed = ", ".join(names) or "none"
            raise OllamaError(
                f"Model '{self.model}' is not installed. Run `ollama pull {self.model}` "
                f"or set another model in config.yaml. Installed: {installed}."
            )

    # ---- chat -------------------------------------------------------------
    def chat_json(self, messages, schema, temperature=0.0, num_predict=1024) -> dict:
        key = self._cache_key(messages, schema, temperature, num_predict)
        cached = self._cache_get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        convo = list(messages)
        last_error = "unknown"
        for attempt in range(self.json_repair_attempts + 1):
            raw = self._chat(convo, schema, temperature, num_predict)
            try:
                data = json.loads(raw)
                validate(data, schema)
            except json.JSONDecodeError as exc:
                last_error = f"not valid JSON ({exc.msg})"
            except ValidationError as exc:
                last_error = f"schema violation ({exc.message[:120]})"
            else:
                self._cache_put(key, data)
                return data
            log.warning("Model output rejected (attempt %d): %s", attempt + 1, last_error)
            convo = convo + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That reply was {last_error}. Reply again with ONLY "
                                            f"valid JSON that matches the required schema."},
            ]
        raise InvalidModelOutput(f"Model returned invalid output after repair: {last_error}")

    def _chat(self, messages, schema, temperature, num_predict) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict, "seed": self.seed},
        }
        delay = self.backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                error = exc
            else:
                if resp.status_code == 200:
                    self.stats["calls"] += 1
                    return (resp.json().get("message") or {}).get("content", "")
                if resp.status_code < 500:
                    raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
                error = OllamaError(f"Ollama returned {resp.status_code}")
            if attempt == self.max_retries:
                raise OllamaError(f"Ollama request failed after {attempt} attempts: {error}") from error
            log.warning("Ollama call failed (%s); retrying in %.1fs", error, delay)
            time.sleep(delay)
            delay *= 2

    # ---- cache ------------------------------------------------------------
    def _cache_key(self, messages, schema, temperature, num_predict) -> str:
        blob = json.dumps([self.model, messages, schema, temperature, num_predict, self.seed],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key):
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key, data) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")
