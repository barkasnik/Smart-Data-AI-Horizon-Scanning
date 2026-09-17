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
    "/topics/",
)

GENERIC_TITLES = {
    "department for business and trade",
    "government digital service",
    "financial conduct authority",
    "creating a smart data economy",
    "gov.uk",
    "raidiam",
    "open banking",
    "open finance",
    "smart data",
}

GENERIC_TITLE_PREFIXES = (
    "home - ",
    "homepage - ",
    "about us - ",
    "news and insights - ",
)


def is_hub(article: Article) -> bool:
    """Reject navigation, organisation and collection pages before analysis.

    Horizon scanning should analyse a concrete development, announcement, research item
    or argument. A homepage or organisation landing page is not intelligence evidence.
    """
    url = (article.canonical_url or article.url or "").lower()
    title = (article.title or "").strip().lower()
    path = urlsplit(url).path.lower().rstrip("/")

    if title in GENERIC_TITLES:
        return True
    if any(title.startswith(prefix) for prefix in GENERIC_TITLE_PREFIXES):
        return True
    if any(part in path for part in BAD_PATHS):
        return True

    # Bare domain/root pages are never briefing items.
    if path in {"", "/"}:
        return True

    return False


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

    Metadata-only news is allowed as a monitoring signal, but full or partial source
    text scores higher. Publication date and article-like URLs help; generic landing
    pages score zero.
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

    url = (article.canonical_url or article.url or "").lower()
    if article.discovery_method.startswith("google_news:"):
        score += 3
    if any(fragment in url for fragment in ("/news/", "/insights/", "/publications/", "/consultations/", "/speeches/", "/research/", "/blog/")):
        score += 7

    return max(0.0, min(100.0, score))


def editorial_score(article: Article) -> float:
    """Balance policy relevance with evidence quality.

    The old selector sorted almost entirely by evidence length, which could elevate a
    well-scraped but weakly relevant page above a highly relevant current development.
    This score keeps relevance dominant while rewarding evidence and publication quality.
    """
    level_bonus = {"strong": 14.0, "partial": 9.0, "signal": 2.0, "thin": -8.0}[evidence_level(article)]
    return (
        0.62 * float(article.heuristic_score or 0.0)
        + 0.28 * content_quality(article)
        + level_bonus
    )


def select_diverse(
    articles: list[Article],
    *,
    limit: int,
    max_govuk: int = 3,
    max_per_domain: int = 2,
    max_signal_only: int = 3,
) -> list[Article]:
    """Choose the most relevant evidence without allowing one source or thin RSS signals to dominate."""
    ranked = sorted(
        articles,
        key=lambda a: (editorial_score(a), a.heuristic_score, content_quality(a)),
        reverse=True,
    )

    chosen: list[Article] = []
    counts: Counter[str] = Counter()
    gov_count = 0
    signal_count = 0

    for article in ranked:
        if is_hub(article):
            continue

        level = evidence_level(article)
        if level in {"signal", "thin"} and signal_count >= max_signal_only:
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
        if level in {"signal", "thin"}:
            signal_count += 1

        if len(chosen) >= limit:
            break

    return chosen
