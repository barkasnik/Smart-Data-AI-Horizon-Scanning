from smart_data_radar.models import ArticleAnalysis
from smart_data_radar.prioritise import normalise_hashtags, priority_score
from test_llm import sample_analysis


def test_weekly_and_monthly_weighting_differ():
    data = sample_analysis()
    data["priority"]["urgency"] = 5
    data["priority"]["strategic_significance"] = 2
    analysis = ArticleAnalysis.model_validate(data)
    assert priority_score(analysis, "weekly") > priority_score(analysis, "monthly")


def test_hashtags_add_priority_signals():
    analysis = ArticleAnalysis.model_validate(sample_analysis())
    tags = normalise_hashtags(analysis, ["AI + Smart Data intersection"])
    assert "#AIxSmartData" in tags
    assert "#Opportunity" in tags
    assert "#Monitor" in tags
    assert "#Threat" in tags
    assert "#PolicyGap" in tags

