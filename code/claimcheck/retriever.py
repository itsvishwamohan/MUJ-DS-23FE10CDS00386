"""Classic-NLP retrieval: sentence windows -> stemmed unigram+bigram TF-IDF -> cosine ranking."""
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from textutils import split_sentences

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SUFFIXES = ("ing", "edly", "ed", "es", "s")


def stem(word: str) -> str:
    """Tiny suffix stripper so 'landed' / 'landing' / 'lands' share a term."""
    if word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def analyze(text: str) -> list[str]:
    stems = [stem(t) for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]
    return stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:])]


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    text: str
    start: int  # first sentence index (inclusive)
    end: int    # last sentence index (exclusive)
    url: str = ""


class EvidenceIndex:
    def __init__(self, evidence_dir, window: int = 2):
        self.passages: list[Passage] = []
        for path in sorted(Path(evidence_dir).glob("*")):
            if path.suffix.lower() in {".txt", ".md"}:
                self.passages.extend(self._windows(path, window))
        if not self.passages:
            raise ValueError(f"No .txt/.md evidence files found in {evidence_dir}")
        self.vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([p.text for p in self.passages])

    @staticmethod
    def _windows(path: Path, window: int) -> list[Passage]:
        sentences = split_sentences(path.read_text(encoding="utf-8"))
        last_start = max(1, len(sentences) - window + 1)
        return [
            Passage(f"{path.stem}#{i}", path.name, " ".join(sentences[i:i + window]), i,
                    min(i + window, len(sentences)))
            for i in range(last_start)
        ]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05, extra=()):
        """Return up to top_k (Passage, score) pairs, skipping overlapping windows of one source.

        `extra` holds live passages (e.g. from Wikipedia). They are ranked together with the local
        notes in one TF-IDF space built for this query.
        """
        if extra:
            passages = self.passages + list(extra)
            vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
            matrix = vectorizer.fit_transform([p.text for p in passages])
        else:
            passages, vectorizer, matrix = self.passages, self.vectorizer, self.matrix
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        picked: list[tuple[Passage, float]] = []
        for i in scores.argsort()[::-1]:
            if scores[i] < min_score:
                break
            p = passages[i]
            if any(q.source == p.source and p.start < q.end and q.start < p.end for q, _ in picked):
                continue
            picked.append((p, float(scores[i])))
            if len(picked) == top_k:
                break
        return picked
