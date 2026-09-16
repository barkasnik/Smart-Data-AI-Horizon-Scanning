from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    url: str
    title: str
    snippet: str = ""
    source_name: str = ""
    discovery_method: str = ""
    published_at: datetime | None = None


class Article(BaseModel):
    url: str
    canonical_url: str
    title: str
    source_name: str = ""
    author: str = ""
    published_at: datetime | None = None
    discovered_at: datetime
    discovery_method: str
    snippet: str = ""
    text: str = ""
    heuristic_score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)


class KeyClaim(BaseModel):
    claim: str
    claim_type: Literal["reported_fact", "source_claim", "source_opinion", "analysis"]
    evidence_note: str = ""


class PestleAnalysis(BaseModel):
    political: list[str] = Field(default_factory=list)
    economic: list[str] = Field(default_factory=list)
    social: list[str] = Field(default_factory=list)
    technological: list[str] = Field(default_factory=list)
    legal: list[str] = Field(default_factory=list)
    environmental: list[str] = Field(default_factory=list)


class SwotAnalysis(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)


class PrioritySignals(BaseModel):
    urgency: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    consequences: int = Field(ge=1, le=5)
    policy_advancement: int = Field(ge=1, le=5)
    opportunity: int = Field(ge=1, le=5)
    monitoring_need: int = Field(ge=1, le=5)
    strategic_significance: int = Field(ge=1, le=5)
    implementation_risk: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    evidence_strength: int = Field(ge=1, le=5)
    rationale: str


class ArticleAnalysis(BaseModel):
    relevance_score: int = Field(ge=0, le=100)
    relevance_tier: Literal["must_read", "high", "medium", "low"]
    primary_category: str
    bottom_line: str
    summary: str
    why_it_matters: str
    government_smart_data_perspective: str
    smart_data_ai_link: str
    pestle: PestleAnalysis
    swot: SwotAnalysis
    priority: PrioritySignals
    policy_or_market_implications: list[str]
    key_claims: list[KeyClaim]
    source_perspective: str
    tensions_or_tradeoffs: list[str]
    follow_up_questions: list[str]
    hashtags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


class DigestSynthesis(BaseModel):
    mode: Literal["weekly", "monthly"]
    headline: str
    executive_summary: str
    ranked_priorities: list[str] = Field(default_factory=list)
    emerging_patterns: list[str] = Field(default_factory=list)
    ai_smart_data_developments: list[str] = Field(default_factory=list)
    wider_smart_data_developments: list[str] = Field(default_factory=list)
    policy_gaps: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    tensions_to_watch: list[str] = Field(default_factory=list)
    monitoring_list: list[str] = Field(default_factory=list)
    questions_for_policy_teams: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


class AnalysedArticle(BaseModel):
    article: Article
    analysis: ArticleAnalysis
    final_score: float
    priority_score: float = 0.0
    mode: Literal["weekly", "monthly"] = "weekly"

