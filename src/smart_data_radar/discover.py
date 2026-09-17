from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote_plus, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from .models import Candidate
from .utils import canonicalise_url, clean_space, domain_of

USER_AGENT = "SmartDataAIRadar/5.4 (+https://github.com/barkasnik/Smart-Data-AI-Horizon-Scanning)"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return dateparser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _plain_html(value: str | None) -> str:
    if not value:
        return ""
    return clean_space(BeautifulSoup(value, "html.parser").get_text(" ", strip=True))


def _publisher_link_from_google_description(value: str | None) -> str | None:
    """Prefer the publisher URL embedded in a Google News RSS description.

    Google News RSS item links are often redirect URLs that are poor inputs for
    article extraction. The description frequently contains a direct publisher
    anchor; use it when it is a normal http(s) URL outside news.google.com.
    """
    if not value:
        return None
    soup = BeautifulSoup(value, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = clean_space(anchor.get("href", ""))
        if not href.startswith(("http://", "https://")):
            continue
        host = urlsplit(href).netloc.lower()
        if host and "news.google.com" not in host:
            return canonicalise_url(href)
    return None


def search_queries(profile: dict) -> list[str]:
    primary = list(profile.get("primary", {}).get("phrases", []))
    secondary = list(profile.get("secondary", {}).get("phrases", []))
    return primary + secondary


def discover_serper(queries: list[str], per_query: int = 10) -> list[Candidate]:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is required when SEARCH_BACKEND=serper")
    out: list[Candidate] = []
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    with httpx.Client(timeout=30, headers=headers) as client:
        for query in queries:
            response = client.post(
                "https://google.serper.dev/search",
                json={"q": query, "gl": "gb", "hl": "en", "num": per_query},
            )
            response.raise_for_status()
            for item in response.json().get("organic", [])[:per_query]:
                link = item.get("link")
                title = clean_space(item.get("title", ""))
                if not link or not title:
                    continue
                out.append(Candidate(
                    url=canonicalise_url(link),
                    title=title,
                    snippet=clean_space(item.get("snippet", "")),
                    source_name=domain_of(link),
                    discovery_method=f"serper:{query}",
                    published_at=_parse_date(item.get("date")),
                ))
    return out


def discover_google_cse(queries: list[str], per_query: int = 10) -> list[Candidate]:
    key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not key or not cx:
        raise RuntimeError("GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID are required for google_cse")
    out: list[Candidate] = []
    with httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for query in queries:
            params = {"key": key, "cx": cx, "q": query, "num": min(10, per_query)}
            response = client.get("https://customsearch.googleapis.com/customsearch/v1", params=params)
            response.raise_for_status()
            for item in response.json().get("items", []):
                link = item.get("link")
                title = clean_space(item.get("title", ""))
                if not link or not title:
                    continue
                out.append(Candidate(
                    url=canonicalise_url(link),
                    title=title,
                    snippet=clean_space(item.get("snippet", "")),
                    source_name=domain_of(link),
                    discovery_method=f"google_cse:{query}",
                ))
    return out


def discover_google_news(queries: list[str], per_query: int = 10, days: int | None = None) -> list[Candidate]:
    """Credential-free discovery using Google News RSS.

    Prefer a publisher URL embedded in the RSS description. Keep the RSS title,
    publisher, date and snippet as evidence even if the publisher later blocks
    server-side extraction.
    """
    out: list[Candidate] = []
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        for query in queries:
            timed_query = f"{query} when:{days}d" if days else query
            url = (
                "https://news.google.com/rss/search?q=" + quote_plus(timed_query)
                + "&hl=en-GB&gl=GB&ceid=GB:en"
            )
            response = client.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            for item in root.findall("./channel/item")[:per_query]:
                title = clean_space(item.findtext("title", default=""))
                google_link = clean_space(item.findtext("link", default=""))
                raw_description = item.findtext("description", default="")
                direct_link = _publisher_link_from_google_description(raw_description)
                link = direct_link or google_link
                description = _plain_html(raw_description)
                pub = _parse_date(item.findtext("pubDate"))
                source = item.find("source")
                source_name = clean_space(source.text if source is not None and source.text else "Google News")

                suffix = f" - {source_name}"
                if source_name and title.endswith(suffix):
                    title = title[:-len(suffix)].strip()

                if title and link:
                    out.append(Candidate(
                        url=canonicalise_url(link),
                        title=title,
                        snippet=description,
                        source_name=source_name,
                        discovery_method=f"google_news:{query}",
                        published_at=pub,
                    ))
    return out


def discover_search(profile: dict, backend: str | None = None, per_query: int = 8, days: int | None = None) -> list[Candidate]:
    backend = (backend or os.getenv("SEARCH_BACKEND", "google_news")).lower()
    queries = search_queries(profile)
    if backend == "serper":
        return discover_serper(queries, per_query)
    if backend == "google_cse":
        return discover_google_cse(queries, per_query)
    if backend == "google_news":
        return discover_google_news(queries, per_query, days=days)
    raise ValueError(f"Unsupported SEARCH_BACKEND={backend!r}")


def discover_index_sources(sources: dict) -> list[Candidate]:
    out: list[Candidate] = []
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        for cfg in sources.get("index_sources", []):
            try:
                response = client.get(cfg["url"])
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            allowed = {d.lower() for d in cfg.get("allowed_domains", [])}
            include_fragments = cfg.get("link_contains", [])
            exclude_fragments = cfg.get("exclude_link_contains", [])
            seen: set[str] = set()
            max_links = int(cfg.get("max_links", 100))

            for anchor in soup.find_all("a", href=True):
                href = canonicalise_url(urljoin(str(response.url), anchor["href"]))
                if href in seen:
                    continue
                if allowed and domain_of(href) not in allowed:
                    continue
                if include_fragments and not any(fragment in href for fragment in include_fragments):
                    continue
                if exclude_fragments and any(fragment in href for fragment in exclude_fragments):
                    continue

                title = clean_space(anchor.get_text(" ", strip=True))
                if len(title) < 8:
                    continue

                seen.add(href)
                out.append(Candidate(
                    url=href,
                    title=title,
                    source_name=cfg.get("name", domain_of(href)),
                    discovery_method=f"index:{cfg.get('kind', 'index')}",
                ))
                if len(seen) >= max_links:
                    break
    return out
