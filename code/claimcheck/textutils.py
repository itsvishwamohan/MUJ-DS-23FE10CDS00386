"""Small dependency-free text helpers shared by the pipeline."""
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_FORWARD_LINE = re.compile(r"(?im)^[ \t]*(?:forwarded(?: many times)?|fwd)[ \t]*[:!.]*[ \t]*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_ELLIPSIS = re.compile(r"\.\.\.|\u2026")
_PUNCT_MAP = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def clean_text(text: str) -> str:
    """Strip URLs and 'Forwarded' banners, normalise unicode and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub("", text)
    text = _FORWARD_LINE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Line-aware sentence splitter (WhatsApp forwards often lack punctuation)."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_BREAK.split(line) if s.strip())
    return sentences


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Group sentences into chunks of at most max_chars (a lone long sentence stays whole)."""
    chunks, current = [], ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def normalize(text: str) -> str:
    """Lower-case, unify quotes/dashes, drop punctuation: used for tolerant matching."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 8,848.86 -> 8848.86
    return " ".join(re.findall(r"\d+(?:\.\d+)?|\w+", text))


def word_count(text: str) -> int:
    return len(normalize(_ELLIPSIS.sub(" ", text)).split())


def quote_in_text(quote: str, text: str) -> bool:
    """True if every fragment of `quote` (split on ellipses) occurs in `text` as whole words."""
    haystack = f" {normalize(text)} "
    fragments = [normalize(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f" {f} " in haystack for f in fragments)
