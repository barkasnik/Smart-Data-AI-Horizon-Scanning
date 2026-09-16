from smart_data_radar.models import ArticleAnalysis, DigestSynthesis


def sample_analysis() -> dict:
    return {
        "relevance_score": 90,
        "relevance_tier": "must_read",
        "primary_category": "AI + Smart Data",
        "bottom_line": "Agentic services could change how consent and delegation work in Smart Data schemes.",
        "summary": "A concise summary.",
        "why_it_matters": "It changes the consent model.",
        "government_smart_data_perspective": "The development may require scheme designers to distinguish a consumer instruction from an agent acting under delegated authority.",
        "smart_data_ai_link": "AI agents act over permissioned data.",
        "pestle": {
            "political": ["Cross-government ownership may need to be clear."],
            "economic": ["Could lower switching friction."],
            "social": ["Trust will depend on understandable delegation."],
            "technological": ["APIs need machine-readable permissions."],
            "legal": ["Consent and liability boundaries may need testing."],
            "environmental": [],
        },
        "swot": {
            "strengths": ["Existing API and trust-framework experience."],
            "weaknesses": ["A policy gap around delegated authority."],
            "opportunities": ["Test agent permissions in a sandbox."],
            "threats": ["Fragmented proprietary agent standards."],
        },
        "priority": {
            "urgency": 4,
            "impact": 5,
            "consequences": 4,
            "policy_advancement": 4,
            "opportunity": 4,
            "monitoring_need": 5,
            "strategic_significance": 5,
            "implementation_risk": 4,
            "novelty": 4,
            "evidence_strength": 3,
            "rationale": "The signal is strategically important but still developing.",
        },
        "policy_or_market_implications": ["Review delegated authority."],
        "key_claims": [{"claim": "A claim", "claim_type": "source_claim", "evidence_note": "In article"}],
        "source_perspective": "Commercial infrastructure provider.",
        "tensions_or_tradeoffs": ["Convenience vs control"],
        "follow_up_questions": ["Who is accountable?"],
        "hashtags": ["#AIxSmartData", "#Monitor"],
        "tags": ["agentic-ai", "consent"],
        "confidence": "medium",
    }


def test_article_analysis_schema_accepts_expected_shape():
    obj = ArticleAnalysis.model_validate(sample_analysis())
    assert obj.relevance_score == 90
    assert obj.priority.monitoring_need == 5
    assert obj.pestle.environmental == []


def test_digest_synthesis_schema():
    x = DigestSynthesis(
        mode="weekly",
        headline="Delegation becomes a Smart Data issue",
        executive_summary="Several items point to this theme.",
        confidence="medium",
    )
    assert x.headline.startswith("Delegation")
    assert x.mode == "weekly"

