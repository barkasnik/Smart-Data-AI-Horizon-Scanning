from datetime import datetime, timezone

from smart_data_radar.digest import render_markdown, write_outputs
from smart_data_radar.models import AnalysedArticle, Article, ArticleAnalysis
from test_llm import sample_analysis


def test_digest_contains_pestle_swot_and_hashtags():
    article = Article(
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        title="Agentic AI and Smart Data",
        discovered_at=datetime.now(timezone.utc),
        discovery_method="test",
    )
    analysis = ArticleAnalysis.model_validate(sample_analysis())
    item = AnalysedArticle(
        article=article,
        analysis=analysis,
        final_score=88.5,
        priority_score=91.0,
        mode="weekly",
    )
    text = render_markdown([item], mode="weekly")
    assert "PESTLE" in text
    assert "SWOT" in text
    assert "#AIxSmartData" in text
    assert "Policy priority" in text


def test_public_json_excludes_scraped_article_text(tmp_path):
    article = Article(
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        title="Agentic AI and Smart Data",
        discovered_at=datetime.now(timezone.utc),
        discovery_method="test",
        snippet="A discovery snippet that should not be published.",
        text="Full scraped article body that must stay out of public JSON.",
    )
    analysis = ArticleAnalysis.model_validate(sample_analysis())
    item = AnalysedArticle(
        article=article, analysis=analysis, final_score=88.5, priority_score=91.0, mode="weekly"
    )
    _, json_path, _ = write_outputs([item], out_dir=tmp_path, mode="weekly")
    payload = json_path.read_text(encoding="utf-8")
    assert "Full scraped article body" not in payload
    assert "A discovery snippet" not in payload
    assert "Agentic AI and Smart Data" in payload
    assert "#AIxSmartData" in payload

