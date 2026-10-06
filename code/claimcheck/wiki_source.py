"""Live evidence from Wikipedia through the MediaWiki Action API (no API key needed).

For each claim: search Wikipedia with the claim's keywords, take the intro of the top pages and the
full text of the best page, and cut everything into the same sentence-window passages as the local notes.
Raw API responses are cached on disk, so repeated runs work offline and give identical results.
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote

import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from retriever import Passage
from textutils import split_sentences

log = logging.getLogger(__name__)
_HEADING = re.compile(r"^\s*=+.*=+\s*$")
_WORD = re.compile(r"\w+")


class WikipediaError(RuntimeError):
    """Wikipedia could not be reached or returned something unusable."""


class WikipediaSource:
    def __init__(self, language="en", pages=4, window=2, max_chars=30000, timeout=10,
                 cache_dir=None, session=None, results_per_query=2):
        self.language, self.pages, self.window = language, pages, window
        self.results_per_query = results_per_query
        self.max_chars, self.timeout = max_chars, timeout
        self.api = f"https://{language}.wikipedia.org/w/api.php"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = "ClaimCheck/1.0 (student NLP project; python-requests)"
        self.session = session

    # ---- public ---------------------------------------------------------------
    def passages_for(self, claim: str, queries=None) -> list[Passage]:
        pages = self._search_pages(claim, queries or [])
        if not pages:
            return []
        top = pages[0]
        try:  # the best page in full, because most facts are not in the intro
            top_text = self._full_text(top["title"]) or top["extract"]
        except WikipediaError:
            top_text = top["extract"]
        sources = [(top["title"], top_text)] + [(p["title"], p["extract"]) for p in pages[1:]]
        passages: list[Passage] = []
        for title, text in sources:
            passages.extend(self._windows(title, text[: self.max_chars]))
        return passages

    # ---- internals ------------------------------------------------------------
    @staticmethod
    def _queries(claim: str) -> list[str]:
        words = [w for w in dict.fromkeys(_WORD.findall(claim)) if w.lower() not in ENGLISH_STOP_WORDS]
        longest = sorted(words, key=len, reverse=True)[:3]
        queries = [" ".join(words[:8]), " ".join(w for w in words if w in longest)]
        return [q for q in dict.fromkeys(queries) if q]

    def _search(self, query: str, limit: int) -> list[dict]:
        data = self._get({
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
            "gsrlimit": limit, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": limit,
        })
        found = sorted(((data.get("query") or {}).get("pages") or {}).values(),
                       key=lambda p: p.get("index", 99))
        return [p for p in found if p.get("extract") and "may refer to" not in p["extract"][:300]]

    def _search_pages(self, claim: str, queries: list[str]) -> list[dict]:
        """Pages for the extractor's queries first; keyword search on the claim as a fallback."""
        pages, seen = [], set()
        for query in queries[:3]:
            for page in self._search(query, self.results_per_query):
                if page["title"] not in seen:
                    seen.add(page["title"])
                    pages.append(page)
        if not pages:
            for query in self._queries(claim):
                pages = self._search(query, self.pages)
                if pages:
                    break
        return pages[: self.pages]

    def _full_text(self, title: str) -> str:
        data = self._get({"action": "query", "format": "json", "prop": "extracts",
                          "explaintext": 1, "exlimit": 1, "titles": title})
        pages = list(((data.get("query") or {}).get("pages") or {}).values())
        return (pages[0].get("extract") or "") if pages else ""

    def _windows(self, title: str, text: str) -> list[Passage]:
        lines = [ln for ln in text.splitlines() if ln.strip() and not _HEADING.match(ln)]
        sentences = split_sentences("\n".join(lines))
        slug = title.replace(" ", "_")
        url = f"https://{self.language}.wikipedia.org/wiki/{quote(slug)}"
        return [
            Passage(f"wiki:{slug}#{i}", f"Wikipedia: {title}", " ".join(sentences[i:i + self.window]),
                    i, min(i + self.window, len(sentences)), url)
            for i in range(max(1, len(sentences) - self.window + 1)) if sentences
        ]

    def _get(self, params: dict) -> dict:
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cached and cached.exists():
            try:
                return json.loads(cached.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        try:
            resp = self.session.get(self.api, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                raise WikipediaError(f"Wikipedia returned HTTP {resp.status_code}")
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise WikipediaError(f"cannot reach Wikipedia: {exc}") from exc
        if cached:
            cached.write_text(json.dumps(data), encoding="utf-8")
        return data
