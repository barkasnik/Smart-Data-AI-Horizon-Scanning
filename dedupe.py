from __future__ import annotations

from rapidfuzz.fuzz import ratio

from .models import Article, Candidate
from .utils import canonicalise_url


def dedupe_candidates(items: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for item in items:
        key = canonicalise_url(item.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def dedupe_articles(items: list[Article], title_threshold: int = 92) -> list[Article]:
    out: list[Article] = []
    seen_urls: set[str] = set()
    for item in sorted(items, key=lambda a: a.heuristic_score, reverse=True):
        url_key = canonicalise_url(item.canonical_url or item.url)
        if url_key in seen_urls:
            continue
        duplicate = False
        for existing in out:
            if ratio(item.title.lower(), existing.title.lower()) >= title_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        seen_urls.add(url_key)
        out.append(item)
    return out

