from __future__ import annotations

from datetime import timezone
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from dateutil import parser as dateparser
from trafilatura.metadata import extract_metadata

from .models import Article, Candidate
from .utils import canonicalise_url, clean_space, utcnow

USER_AGENT = "SmartDataAIRadar/5.4 (+https://github.com/barkasnik/Smart-Data-AI-Horizon-Scanning)"


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        p = urlsplit(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._cache:
            rp = RobotFileParser()
            rp.set_url(root + "/robots.txt")
            try:
                rp.read()
                self._cache[root] = rp
            except Exception:
                self._cache[root] = None
        rp = self._cache[root]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)


def _parse_meta_date(value: str | None):
    if not value:
        return None
    try:
        dt = dateparser.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _metadata_article(candidate: Candidate) -> Article:
    return Article(
        url=candidate.url,
        canonical_url=canonicalise_url(candidate.url),
        title=candidate.title,
        source_name=candidate.source_name,
        author="",
        published_at=candidate.published_at,
        discovered_at=utcnow(),
        discovery_method=candidate.discovery_method,
        snippet=clean_space(candidate.snippet),
        text="",
    )


def extract_candidate(candidate: Candidate, robots: RobotsCache | None = None) -> Article:
    """Extract article text when permitted, otherwise preserve discovery evidence.

    robots.txt or JavaScript-heavy pages must not erase a valid Google News signal.
    Metadata-only items can still be analysed at lower evidence strength.
    """
    robots = robots or RobotsCache()
    if not robots.allowed(candidate.url):
        return _metadata_article(candidate)

    headers = {"User-Agent": USER_AGENT}
    article = _metadata_article(candidate)

    try:
        with httpx.Client(timeout=35, headers=headers, follow_redirects=True) as client:
            response = client.get(candidate.url)
            response.raise_for_status()
            final_url = str(response.url)
            html = response.text

        if "news.google.com" not in urlsplit(final_url).netloc:
            article.canonical_url = canonicalise_url(final_url)

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        ) or ""
        article.text = clean_space(text)

        meta = extract_metadata(html)
        if meta:
            article.title = clean_space(meta.title or article.title)
            article.author = clean_space(meta.author or "")
            article.published_at = _parse_meta_date(meta.date) or article.published_at
            if getattr(meta, "url", None) and "news.google.com" not in urlsplit(str(meta.url)).netloc:
                article.canonical_url = canonicalise_url(meta.url)
    except Exception:
        pass

    return article
