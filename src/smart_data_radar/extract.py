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

USER_AGENT = "SmartDataAIRadar/0.1 (+https://github.com/)"


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


def extract_candidate(candidate: Candidate, robots: RobotsCache | None = None) -> Article:
    robots = robots or RobotsCache()
    if not robots.allowed(candidate.url):
        raise PermissionError(f"robots.txt disallows extraction: {candidate.url}")

    headers = {"User-Agent": USER_AGENT}
    text = ""
    title = candidate.title
    author = ""
    published_at = candidate.published_at
    canonical = canonicalise_url(candidate.url)

    try:
        with httpx.Client(timeout=35, headers=headers, follow_redirects=True) as client:
            response = client.get(candidate.url)
            response.raise_for_status()
            final_url = str(response.url)
            if "news.google.com" not in urlsplit(final_url).netloc:
                canonical = canonicalise_url(final_url)
            html = response.text
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        ) or ""
        meta = extract_metadata(html)
        if meta:
            title = clean_space(meta.title or title)
            author = clean_space(meta.author or "")
            published_at = _parse_meta_date(meta.date) or published_at
            if getattr(meta, "url", None):
                canonical = canonicalise_url(meta.url)
    except Exception:
        # Preserve discovery metadata so the radar can still rank a blocked or
        # JavaScript-heavy page. The LLM is only called when enough text exists.
        text = ""

    return Article(
        url=candidate.url,
        canonical_url=canonical,
        title=title,
        source_name=candidate.source_name,
        author=author,
        published_at=published_at,
        discovered_at=utcnow(),
        discovery_method=candidate.discovery_method,
        snippet=candidate.snippet,
        text=clean_space(text),
    )

