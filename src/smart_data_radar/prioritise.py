from __future__ import annotations

from .models import ArticleAnalysis

WEEKLY_WEIGHTS = {
    "urgency": 0.20,
    "impact": 0.16,
    "consequences": 0.14,
    "policy_advancement": 0.11,
    "opportunity": 0.08,
    "monitoring_need": 0.11,
    "strategic_significance": 0.08,
    "implementation_risk": 0.07,
    "novelty": 0.05,
}

MONTHLY_WEIGHTS = {
    "urgency": 0.06,
    "impact": 0.17,
    "consequences": 0.14,
    "policy_advancement": 0.14,
    "opportunity": 0.11,
    "monitoring_need": 0.08,
    "strategic_significance": 0.17,
    "implementation_risk": 0.06,
    "novelty": 0.07,
}

CONTROLLED_HASHTAGS = {
    "#AIxSmartData",
    "#PolicyGap",
    "#Opportunity",
    "#Threat",
    "#Monitor",
    "#Urgent",
    "#Strategic",
    "#ImplementationRisk",
    "#RegulatoryChange",
    "#ConsumerProtection",
    "#Interoperability",
    "#DigitalIdentity",
    "#Consent",
    "#Competition",
    "#EconomicSecurity",
    "#OpenFinance",
    "#OpenProperty",
    "#Energy",
    "#Transport",
    "#Trade",
    "#Fraud",
    "#International",
    "#TrustFramework",
    "#DataPortability",
}


def priority_score(analysis: ArticleAnalysis, mode: str) -> float:
    weights = MONTHLY_WEIGHTS if mode == "monthly" else WEEKLY_WEIGHTS
    signals = analysis.priority
    raw = sum((getattr(signals, key) / 5.0) * weight for key, weight in weights.items()) * 100
    # Evidence strength tempers, but does not dominate, the priority calculation.
    evidence_factor = 0.90 + (signals.evidence_strength / 5.0) * 0.10
    return round(raw * evidence_factor, 2)


def final_rank_score(*, heuristic: float, relevance: int, priority: float) -> float:
    return round((0.15 * heuristic) + (0.20 * relevance) + (0.65 * priority), 2)


def normalise_hashtags(analysis: ArticleAnalysis, matched_terms: list[str]) -> list[str]:
    tags: list[str] = []

    for raw in analysis.hashtags:
        tag = raw.strip().replace(" ", "")
        if not tag.startswith("#"):
            tag = "#" + tag
        # case-insensitive matching onto canonical forms
        canonical = next((x for x in CONTROLLED_HASHTAGS if x.lower() == tag.lower()), None)
        if canonical and canonical not in tags:
            tags.append(canonical)

    if "AI + Smart Data intersection" in matched_terms and "#AIxSmartData" not in tags:
        tags.append("#AIxSmartData")
    if analysis.priority.opportunity >= 4 and "#Opportunity" not in tags:
        tags.append("#Opportunity")
    if analysis.priority.monitoring_need >= 4 and "#Monitor" not in tags:
        tags.append("#Monitor")
    if analysis.priority.urgency >= 4 and "#Urgent" not in tags:
        tags.append("#Urgent")
    if analysis.priority.strategic_significance >= 4 and "#Strategic" not in tags:
        tags.append("#Strategic")
    if analysis.priority.implementation_risk >= 4 and "#ImplementationRisk" not in tags:
        tags.append("#ImplementationRisk")
    if analysis.swot.threats and "#Threat" not in tags:
        tags.append("#Threat")

    gap_text = " ".join(analysis.swot.weaknesses + analysis.policy_or_market_implications).lower()
    if any(term in gap_text for term in ("policy gap", "governance gap", "regulatory gap", "lack of", "missing")):
        if "#PolicyGap" not in tags:
            tags.append("#PolicyGap")

    return tags[:8]

