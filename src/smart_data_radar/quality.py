from __future__ import annotations

from collections import Counter
from urllib.parse import urlsplit

from .models import Article
from .utils import domain_of

# Navigation and organisational landing pages are discovery aids, never intelligence items.
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


def content_quality(article: Article) -> float:
    """Estimate how much usable evidence an item contains.

    A recent Google News item with a real title, publisher, date and substantive RSS
    snippet is valid discovery evidence even when the publisher page cannot be scraped.
    Full article text improves confidence but is not a prerequisite for selection.
    """
    if is_hub(article):
        return 0.0

    score = 10.0
    if article.published_at:
        score += 24
    if 18 <= len(article.title or "") <= 220:
        score += 14

    text_len = len(article.text or "")
    snippet_len = len(article.snippet or "")
    if text_len >= 1200:
        score += 34
    elif text_len >= 600:
        score += 28
    elif text_len >= 250:
        score += 18
    elif text_len >= 120:
        score += 10

    if snippet_len >= 180:
        score += 18
    elif snippet_len >= 100:
        score += 14
    elif snippet_len >= 60:
        score += 9

    if article.discovery_method.startswith("google_news:"):
        score += 8
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
    """Choose the strongest evidence while preventing one source from dominating.

    No source is guaranteed a slot. Raidiam, GOV.UK and every other source compete on
    relevance, evidence quality and recency; source caps provide diversity.
    """
    ranked = sorted(
        articles,
        key=lambda a: (a.heuristic_score, content_quality(a)),
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
