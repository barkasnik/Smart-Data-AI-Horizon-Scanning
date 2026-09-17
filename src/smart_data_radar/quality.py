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
    if any(p in path for p in BAD_PATHS):
        return True
    return False

def content_quality(article: Article) -> float:
    if is_hub(article):
        return 0.0
    score = 20.0
    if article.published_at:
        score += 18
    if 20 <= len(article.title or "") <= 190:
        score += 12
    text_len = len(article.text or "")
    snippet_len = len(article.snippet or "")
    if text_len >= 1000:
        score += 28
    elif text_len >= 500:
        score += 20
    elif text_len >= 180:
        score += 10
    if snippet_len >= 80:
        score += 8
    if article.discovery_method.startswith("google_news:"):
        score += 8
    if "/insights/" in (article.canonical_url or article.url or ""):
        score += 8
    return max(0.0, min(100.0, score))

def is_strategic_watch(article: Article) -> bool:
    domain = domain_of(article.canonical_url or article.url)
    text = " ".join([
        article.title or "",
        article.author or "",
        article.snippet or "",
        article.text[:1500] if article.text else "",
    ]).lower()
    if domain == "raidiam.com" and ("smart data" in text or "open finance" in text or "marie walker" in text):
        return True
    return False

def select_diverse(articles: list[Article], *, limit: int, max_govuk: int = 2, max_per_domain: int = 3) -> list[Article]:
    ranked = sorted(articles, key=lambda a: (a.heuristic_score, content_quality(a)), reverse=True)
    chosen: list[Article] = []
    counts = Counter()
    gov_count = 0

    raidiam = [
        a for a in ranked
        if domain_of(a.canonical_url or a.url) == "raidiam.com"
        and content_quality(a) >= 45
    ]
    if raidiam:
        chosen.append(raidiam[0])
        counts["raidiam.com"] += 1

    for a in ranked:
        if a in chosen:
            continue
        domain = domain_of(a.canonical_url or a.url)
        if counts[domain] >= max_per_domain:
            continue
        is_gov = domain == "gov.uk" or domain.endswith(".gov.uk")
        if is_gov and gov_count >= max_govuk:
            continue
        chosen.append(a)
        counts[domain] += 1
        if is_gov:
            gov_count += 1
        if len(chosen) >= limit:
            break
    return chosen[:limit]
