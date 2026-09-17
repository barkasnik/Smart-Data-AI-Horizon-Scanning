from smart_data_radar.models import Article
from smart_data_radar.quality import content_quality, is_hub
from smart_data_radar.utils import utcnow


def test_recent_google_news_metadata_is_valid_discovery_evidence():
    article = Article(
        url="https://news.google.com/rss/articles/example",
        canonical_url="https://news.google.com/rss/articles/example",
        title="Smart Data and AI interoperability update",
        source_name="Example Publisher",
        published_at=utcnow(),
        discovered_at=utcnow(),
        discovery_method="google_news:smart data AI",
        snippet=(
            "A new Smart Data interoperability initiative examines AI, consent and trusted APIs "
            "for consumer services and sets out new implementation evidence for policy teams."
        ),
        text="",
        heuristic_score=72.0,
        matched_terms=["smart data", "AI", "interoperability"],
    )
    assert not is_hub(article)
    assert content_quality(article) >= 45


def test_generic_government_organisation_page_is_never_an_intelligence_item():
    article = Article(
        url="https://www.gov.uk/government/organisations/government-digital-service",
        canonical_url="https://www.gov.uk/government/organisations/government-digital-service",
        title="Government Digital Service",
        source_name="GOV.UK",
        published_at=utcnow(),
        discovered_at=utcnow(),
        discovery_method="test",
        snippet="Government organisation landing page.",
        text="",
    )
    assert is_hub(article)
    assert content_quality(article) == 0
