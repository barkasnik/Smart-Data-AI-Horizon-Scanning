from __future__ import annotations

import math
from datetime import datetime, timezone

from .models import Article
from .utils import domain_of


def _has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def heuristic_score(article: Article, profile: dict, sources: dict, now: datetime | None = None) -> tuple[float, list[str]]:
    now = now or datetime.now(timezone.utc)
    haystack = " ".join([article.title, article.snippet, article.author, article.text[:12000]]).lower()
    watch = profile.get("watch_terms", {})
    matched: list[str] = []
    score = 0.0

    ai_terms = watch.get("ai", [])
    sd_terms = watch.get("smart_data", [])
    has_ai = _has_any(haystack, ai_terms)
    has_sd = _has_any(haystack, sd_terms)

    if "smart data" in haystack:
        score += 24
        matched.append("smart data")
    elif has_sd:
        score += 12
        matched.append("adjacent smart/open data")

    if has_ai:
        score += 22
        matched.append("AI")
    if has_ai and has_sd:
        score += 28
        matched.append("AI + Smart Data intersection")

    category_weights = {
        "trust_identity": 12,
        "interoperability": 10,
        "governance": 10,
        "sectors": 8,
        "wider_government": 8,
    }
    for category, weight in category_weights.items():
        terms = watch.get(category, [])
        if terms and _has_any(haystack, terms):
            score += weight
            matched.append(category)

    domain = domain_of(article.canonical_url or article.url)
    for configured_domain, boost in sources.get("source_boosts", {}).items():
        if domain == configured_domain or domain.endswith("." + configured_domain):
            score += float(boost)
            matched.append(f"source:{configured_domain}")
            break

    for person, boost in sources.get("person_boosts", {}).items():
        if person.lower() in haystack:
            score += float(boost)
            matched.append(f"person:{person}")

    if article.published_at:
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - pub).total_seconds() / 86400)
        recency = 12 * math.exp(-days / 30)
        score += recency
        matched.append("recent")

    # Headline signal gets a modest bonus because it is less noisy than body text.
    title = article.title.lower()
    if "smart data" in title:
        score += 8
    if _has_any(title, ai_terms):
        score += 6

    return round(min(score, 100.0), 2), matched

