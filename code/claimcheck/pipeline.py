"""ClaimCheck pipeline: clean -> extract claims (LLM) -> retrieve evidence (TF-IDF) -> verify (LLM) -> quote check."""
import logging
import time
from dataclasses import asdict

from config import load_config, load_prompts, resolve
from llm_client import InvalidModelOutput, OllamaClient
from retriever import EvidenceIndex
from wiki_source import WikipediaError, WikipediaSource
from schemas import EXTRACT_SCHEMA, NOT_ENOUGH, VERIFY_SCHEMA
from textutils import chunk_text, clean_text, normalize, quote_in_text
from verifier import finalize_verdict, summarize

log = logging.getLogger(__name__)


class ClaimCheckPipeline:
    def __init__(self, cfg: dict, prompts: dict, client, index: EvidenceIndex, wiki=None):
        self.cfg, self.prompts, self.client, self.index, self.wiki = cfg, prompts, client, index, wiki
        self.warnings: list[str] = []

    @classmethod
    def from_config(cls, config_path=None, model=None, use_cache=True, local_only=False):
        cfg = load_config(config_path)
        if model:
            cfg["ollama"]["model"] = model
        o = cfg["ollama"]
        cache_dir = resolve(cfg["cache"]["directory"]) if cfg["cache"]["enabled"] and use_cache else None
        client = OllamaClient(
            host=o["host"], model=o["model"], timeout=o["timeout_seconds"],
            max_retries=o["max_retries"], backoff=o["retry_backoff_seconds"],
            keep_alive=o["keep_alive"], seed=o["seed"],
            json_repair_attempts=o["json_repair_attempts"], cache_dir=cache_dir,
        )
        index = EvidenceIndex(resolve(cfg["retrieval"]["evidence_dir"]), cfg["retrieval"]["window_sentences"])
        w, wiki = cfg.get("wikipedia", {}), None
        if w.get("enabled") and not local_only:
            wiki = WikipediaSource(
                language=w["language"], pages=w["pages_per_claim"], max_chars=w["max_chars_top_page"],
                results_per_query=w.get("results_per_query", 2),
                window=cfg["retrieval"]["window_sentences"], timeout=w["timeout_seconds"],
                cache_dir=cache_dir / "wikipedia" if cache_dir else None)
        return cls(cfg, load_prompts(cfg), client, index, wiki)

    # ---- prompt assembly ----------------------------------------------------
    def _messages(self, stage: str, user_text: str) -> list[dict]:
        block = self.prompts[stage]
        messages = [{"role": "system", "content": block["system"].strip()}]
        for shot in block.get("few_shot", []):
            messages.append({"role": "user", "content": shot["user"].strip()})
            messages.append({"role": "assistant", "content": shot["assistant"].strip()})
        messages.append({"role": "user", "content": user_text})
        return messages

    # ---- stage 1: claim extraction --------------------------------------------
    def extract_claims(self, text: str) -> list[dict]:
        cfg = self.cfg["extraction"]
        text = clean_text(text)
        claims, seen = [], set()
        for chunk in chunk_text(text, cfg["chunk_chars"]):
            user = self.prompts["extraction"]["user"].replace("{text}", chunk)
            data = self.client.chat_json(self._messages("extraction", user), EXTRACT_SCHEMA,
                                         cfg["temperature"], cfg["num_predict"])
            for item in data["claims"]:
                claim, key = item["claim"].strip(), normalize(item["claim"])
                if len(key.split()) < cfg["min_claim_words"] or key in seen:
                    continue
                seen.add(key)
                original = item["original_text"].strip()
                queries = [q.strip() for q in item.get("search_queries", []) if q.strip()][:3]
                claims.append({"claim": claim, "search_queries": queries,
                               "original_text": original if original and quote_in_text(original, text) else None})
                if len(claims) >= cfg["max_claims"]:
                    return claims
        return claims

    # ---- stage 2+3: retrieval and verification ------------------------------------
    def verify_claims(self, claims: list[dict]) -> list[dict]:
        r_cfg, v_cfg = self.cfg["retrieval"], self.cfg["verification"]
        results, pending = [], []
        self.warnings, wiki_up = [], self.wiki is not None
        for i, claim in enumerate(claims, 1):
            extra = []
            if wiki_up:
                try:
                    extra = self.wiki.passages_for(claim["claim"], claim.get("search_queries"))
                except WikipediaError as exc:
                    wiki_up = False  # do not wait for a timeout on every remaining claim
                    self.warnings.append(f"Wikipedia lookup failed ({exc}); used local evidence only.")
            hits = self.index.search(claim["claim"], r_cfg["top_k"], r_cfg["min_score"], extra=extra)
            passages = [{**asdict(p), "score": round(s, 3)} for p, s in hits]
            result = {"id": i, "claim": claim["claim"], "original_text": claim.get("original_text"),
                      "passages": passages, "rejected_quote": "",
                      "search_queries": claim.get("search_queries", [])}
            if passages:
                pending.append(result)
            else:  # nothing relevant retrieved: no LLM call needed
                result.update(label=NOT_ENOUGH, model_label=None, confidence=0.0, evidence_quote="",
                              source="", passage_id="", note="",
                              reasoning="No relevant passage was found in the evidence collection.")
            results.append(result)

        size = v_cfg["batch_size"]
        for start in range(0, len(pending), size):
            batch = pending[start:start + size]
            raw_by_id = self._verify_batch(batch)
            for n, result in enumerate(batch, 1):
                result.update(finalize_verdict(raw_by_id.get(n), result["passages"], v_cfg["min_quote_words"]))

        for _ in range(v_cfg.get("quote_repair_attempts", 0)):
            failed = [r for r in pending if r["rejected_quote"]]
            if not failed:
                break
            for start in range(0, len(failed), size):
                batch = failed[start:start + size]
                raw_by_id = self._verify_batch(batch, repair=True)
                for n, result in enumerate(batch, 1):
                    self._apply_repair(result, raw_by_id.get(n), v_cfg["min_quote_words"])
        return results

    @staticmethod
    def _apply_repair(result: dict, raw: dict | None, min_words: int) -> None:
        """Accept the retry only if it is verifiable; otherwise keep the (downgraded) first verdict."""
        new = finalize_verdict(raw, result["passages"], min_words) if raw else None
        if new is None or new["rejected_quote"]:
            result["note"] = result["note"].replace("Downgraded:", "Downgraded after retry:", 1)
            return
        original_label = result["model_label"]
        new["note"] = ("Quote corrected on retry; the first attempt was not verbatim."
                       if new["evidence_quote"] else "Verdict withdrawn on retry: no exact quote could be given.")
        result.update(new)
        result["model_label"] = original_label

    def _verify_batch(self, batch: list[dict], repair: bool = False) -> dict:
        """Return {local claim number: raw verdict}. Falls back to one claim per call on bad JSON."""
        v_cfg = self.cfg["verification"]
        blocks = []
        for n, item in enumerate(batch, 1):
            lines = [f"CLAIM {n}: {item['claim']}", "PASSAGES:"]
            lines += [f"[{p['id']}] ({p['source']}) {p['text']}" for p in item["passages"]]
            blocks.append("\n".join(lines))
        user = (self.prompts["verification"]["user"]
                .replace("{count}", str(len(batch))).replace("{claims_block}", "\n\n".join(blocks)))
        if repair:
            user = self.prompts["verification"]["repair_prefix"].strip() + "\n\n" + user
        try:
            data = self.client.chat_json(self._messages("verification", user), VERIFY_SCHEMA,
                                         v_cfg["temperature"], v_cfg["num_predict"])
        except InvalidModelOutput:
            if len(batch) == 1:
                raise
            log.warning("Batch verification failed; retrying claims one at a time")
            return {n: self._verify_batch([item], repair).get(1) for n, item in enumerate(batch, 1)}
        verdicts = data["verdicts"]
        if len(batch) == 1 and verdicts:  # one claim, one verdict: ignore a mis-numbered id
            return {1: verdicts[0]}
        return {int(v["id"]): v for v in verdicts}

    # ---- full run -------------------------------------------------------------
    def run(self, text: str) -> dict:
        started = time.perf_counter()
        calls0, hits0 = self.client.stats["calls"], self.client.stats["cache_hits"]
        results = self.verify_claims(self.extract_claims(text))
        rep = self.cfg["report"]
        return {
            "model": self.client.model,
            "summary": summarize(results, rep["reliable_threshold"], rep["mixed_threshold"]),
            "claims": results,
            "warnings": list(self.warnings),
            "stats": {
                "llm_calls": self.client.stats["calls"] - calls0,
                "cache_hits": self.client.stats["cache_hits"] - hits0,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
        }
