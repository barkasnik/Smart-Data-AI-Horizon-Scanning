from __future__ import annotations

from collections import Counter
from urllib.parse import urlsplit

from .models import Article
from .utils import domain_of

BAD_PATHS = (
    "/government/organisations/",
    "/government/collections/",
    "/about/",
    "/contact",
    "/careers",
    "/search",
    "/sitemap",
)

GENERIC_TITLES = {
    "department for business and trade",
    "government digital service",
    "financial conduct authority",
    "creating a smart data economy",
    "gov.uk",
    "raidiam",
}


def is_hub(article: Article) -> bool:
    url = (article.canonical_url or article.url or "").lower()
    title = (article.title or "").strip().lower()
    path = urlsplit(url).path.lower()
    if title in GENERIC_TITLES:
        return True
    return any(part in path for part in BAD_PATHS)


def evidence_level(article: Article) -> str:
    text_len = len(article.text or "")
    snippet_len = len(article.snippet or "")
    if text_len >= 700:
        return "strong"
    if text_len >= 250:
        return "partial"
    if snippet_len >= 80:
        return "signal"
    return "thin"


def content_quality(article: Article) -> float:
    """Estimate how much usable evidence an item contains.

    Metadata-only news is allowed as a horizon-scanning signal, but full or partial
    source text is deliberately scored higher so the briefing prefers evidence-bearing
    items over headlines when both are available.
    """
    if is_hub(article):
        return 0.0

    score = 8.0
    if article.published_at:
        score += 20
    if 18 <= len(article.title or "") <= 220:
        score += 12

    text_len = len(article.text or "")
    snippet_len = len(article.snippet or "")
    if text_len >= 1200:
        score += 44
    elif text_len >= 700:
        score += 38
    elif text_len >= 250:
        score += 25
    elif text_len >= 120:
        score += 14

    if snippet_len >= 180:
        score += 12
    elif snippet_len >= 100:
        score += 9
    elif snippet_len >= 60:
        score += 6

    if article.discovery_method.startswith("google_news:"):
        score += 3
    if "/insights/" in (article.canonical_url or article.url or ""):
        score += 5

    return max(0.0, min(100.0, score))


def select_diverse(
    articles: list[Article],
    *,
    limit: int,
    max_govuk: int = 3,
    max_per_domain: int = 2,
) -> list[Article]:
    """Choose relevant evidence with source diversity and an evidence-first bias."""
    evidence_rank = {"strong": 3, "partial": 2, "signal": 1, "thin": 0}
    ranked = sorted(
        articles,
        key=lambda a: (
            evidence_rank[evidence_level(a)],
            a.heuristic_score,
            content_quality(a),
        ),
        reverse=True,
    )

    chosen: list[Article] = []
    counts: Counter[str] = Counter()
    gov_count = 0

    for article in ranked:
        if is_hub(article):
            continue
        domain = domain_of(article.canonical_url or article.url)
        if counts[domain] >= max_per_domain:
            continue
        is_gov = domain == "gov.uk" or domain.endswith(".gov.uk")
        if is_gov and gov_count >= max_govuk:
            continue
        chosen.append(article)
        counts[domain] += 1
        if is_gov:
            gov_count += 1
        if len(chosen) >= limit:
            break

    return chosen
