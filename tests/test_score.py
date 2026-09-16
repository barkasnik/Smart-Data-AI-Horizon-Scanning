from datetime import datetime, timezone

from smart_data_radar.models import Article
from smart_data_radar.score import heuristic_score


def article(title: str, text: str, url: str = "https://example.com/x") -> Article:
    return Article(
        url=url,
        canonical_url=url,
        title=title,
        text=text,
        discovered_at=datetime.now(timezone.utc),
        discovery_method="test",
    )


def test_ai_smart_data_intersection_scores_above_generic_smart_data():
    profile = {
        "watch_terms": {
            "ai": ["artificial intelligence", "agentic ai"],
            "smart_data": ["smart data", "open finance"],
            "trust_identity": ["consent", "digital identity"],
            "interoperability": ["api", "interoperability"],
            "governance": ["governance"],
            "sectors": ["finance"],
            "wider_government": ["competition"],
        }
    }
    sources = {"source_boosts": {}, "person_boosts": {}}
    a = article("Smart Data and agentic AI", "Consent and API interoperability for artificial intelligence agents")
    b = article("Smart Data update", "A new finance smart data scheme")
    sa, _ = heuristic_score(a, profile, sources)
    sb, _ = heuristic_score(b, profile, sources)
    assert sa > sb

