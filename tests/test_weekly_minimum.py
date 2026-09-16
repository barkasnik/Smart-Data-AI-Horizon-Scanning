from datetime import timezone

from smart_data_radar.llm import fallback_analysis
from smart_data_radar.models import Article
from smart_data_radar.utils import utcnow


def test_fallback_keeps_article_rankable():
    article = Article(
        url="https://example.com/item",
        canonical_url="https://example.com/item",
        title="Smart Data and AI interoperability update",
        source_name="Example",
        published_at=utcnow(),
        discovered_at=utcnow(),
        discovery_method="test",
        snippet="A new Smart Data interoperability initiative examines AI, consent and trusted APIs for consumer services.",
        text="A new Smart Data interoperability initiative examines AI, consent and trusted APIs for consumer services. It will be monitored for policy implications.",
        heuristic_score=72.0,
        matched_terms=["smart data", "AI", "AI + Smart Data intersection", "interoperability", "trust_identity"],
    )
    analysis = fallback_analysis(article, mode="weekly")
    assert analysis.relevance_score == 72
    assert analysis.primary_category == "ai_smart_data"
    assert analysis.priority.monitoring_need >= 3
    assert analysis.bottom_line
