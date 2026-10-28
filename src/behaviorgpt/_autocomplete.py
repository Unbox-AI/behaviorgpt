"""Search suggestions read from a ranking: BehaviorGPT ranks the items the
person is most likely to want given their history and what they have typed so
far, and each item, best first, offers the first of its search terms that
continues the text. So the suggestions follow the person, not only the text."""

import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Iterator, Optional, Sequence

from behaviorgpt.types import Item, Suggestion

# a term matches when at most this share of the typed text has to change,
# so "cofee m" still finds "coffee maker"
MIN_MATCH = 0.7
# a term this alike to an earlier suggestion is left out
MAX_LIKENESS = 0.9
# how many of an item's best terms may become its suggestion
TERMS_PER_ITEM = 5


def normalize(text: str) -> str:
    """Lowercase, without accents, one space between words."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.casefold().split())


def match_score(typed: str, candidate: str) -> float:
    """How well `candidate` continues `typed`: 1.0 when one of its words
    starts with it ("kni" in "pocket knife"), less for each letter that has to
    change, and 0.0 when nothing is close."""
    q, c = normalize(typed), normalize(candidate)
    if not q:
        return 1.0
    best = len(q)
    for start in _word_starts(c):
        best = min(best, _prefix_distance(q, c[start : start + len(q) + best]))
        if best == 0:
            break
    return 1.0 - best / len(q)


def suggest(
    items: Iterable[Item],
    typed: str,
    limit: int,
    earlier: Sequence[Suggestion] = (),
) -> list[Suggestion]:
    """`earlier` and then at most one suggestion per item, in the items' order:
    the first of its terms that matches `typed` and is not alike to an earlier
    one. An item without terms offers its name."""
    found = list(earlier)
    seen = [normalize(s.text) for s in found]
    if len(found) >= limit:
        return found[:limit]
    for item in items:
        for text, source in _candidates(item):
            key = normalize(text)
            if match_score(typed, text) < MIN_MATCH or _alike(key, seen):
                continue
            found.append(Suggestion(text=text, source=source, item=item))
            seen.append(key)
            break
        if len(found) == limit:
            break
    return found


def _candidates(item: Item) -> Iterator[tuple[str, str]]:
    terms = list(_terms(item.data))[:TERMS_PER_ITEM]
    if terms:
        yield from ((term, "keyword") for term in terms)
        return
    name = item.data.get("name")
    if isinstance(name, str) and name.strip():
        yield name.strip(), "name"


def _terms(data: dict) -> Iterator[str]:
    """An item's search terms, best first: `top_terms` as the API returns
    them, `{"term", "score"}` objects or (term, score) pairs."""
    for entry in data.get("top_terms") or ():
        term: Optional[object] = None
        if isinstance(entry, dict):
            term = entry.get("term", entry.get("text"))
        elif isinstance(entry, (list, tuple)) and entry:
            term = entry[0]
        if isinstance(term, str) and term.strip():
            yield term.strip()


def _alike(key: str, seen: list[str]) -> bool:
    return any(
        key == other or SequenceMatcher(None, key, other).ratio() >= MAX_LIKENESS
        for other in seen
    )


def _word_starts(text: str) -> Iterator[int]:
    for i, char in enumerate(text):
        if char.isalnum() and (i == 0 or not text[i - 1].isalnum()):
            yield i


def _prefix_distance(q: str, s: str) -> int:
    """The fewest edits that turn `q` into the start of `s`."""
    previous = list(range(len(s) + 1))
    for i, qc in enumerate(q, 1):
        current = [i]
        for j, sc in enumerate(s, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (qc != sc),
                )
            )
        previous = current
    return min(previous)
