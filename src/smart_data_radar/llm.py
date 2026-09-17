from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

import httpx

from .models import (
    Article,
    ArticleAnalysis,
    DigestSynthesis,
    PestleAnalysis,
    PrioritySignals,
    SwotAnalysis,
)
from .quality import evidence_level

SYSTEM_INSTRUCTIONS = dedent("""
You are the analytical editor for a UK Smart Data and AI policy-intelligence radar.
Assess developments from a UK Government / Smart Data programme perspective. Do not
promote or oppose any source or organisation.

EVIDENCE DISCIPLINE — NON-NEGOTIABLE
- Analyse only the specific event, announcement, research finding, regulatory move or
  argument supported by the supplied evidence.
- Never turn an organisation homepage, collection page or navigation page into a policy
  development.
- Never describe a company or organisation as UK-based, UK-regulated, government-backed,
  authoritative, adopted by government, or relevant to a specific UK regime unless the
  supplied evidence says so.
- Never recommend that UK Government adopts or integrates a named supplier's product.
- A commercial product announcement may be an interoperability or market signal; that is
  not evidence of a UK policy gap.
- If EVIDENCE TYPE is RSS/discovery metadata only, treat the item as a MONITORING SIGNAL:
  no claimed UK policy gap, no claimed UK weakness/threat, no implementation recommendation,
  no asserted regulatory consequence, and confidence must be low. Use empty PESTLE/SWOT
  fields where evidence is insufficient.
- Distinguish reported fact, source claim/opinion and analysis. Do not invent corroboration.
- Raidiam is one monitored commercial source, neither endorsed nor opposed.

EDITORIAL PRIORITIES
Primary: Smart Data + AI, including agentic delegation, consent, digital identity,
APIs/trust frameworks, AI governance, interoperability, data portability and AI-enabled
services. Secondary: Smart Data implementation across finance, property, energy,
transport, retail, trade, agri-food and international data sharing.

PESTLE
Apply PESTLE only where evidenced. Political = ownership/priorities/coordination;
Economic = growth, competition, productivity, market structure, consumers/SMEs;
Social = trust, inclusion/accessibility; Technological = APIs, standards, identity, AI,
interoperability, security; Legal = DUAA/data protection/consumer or sector rules,
liability/consent; Environmental only if materially relevant. Prefer [] to speculation.

SWOT
Use the position of the UK Smart Data programme. Strength = existing UK capability that
specifically helps with this development; Weakness = a programme/design/governance gap
actually exposed by evidence; Opportunity = an external development HMG could plausibly
exploit; Threat = an external development that could obstruct or fragment objectives.
Do not fill quadrants for the sake of completeness.

WRITING
Write like a strong UK policy analyst: natural, specific and economical. Bottom line first.
Summary 2-3 sentences. Maximum one concise point per PESTLE/SWOT field unless a second is
essential. Maximum two implications, two tensions and two questions. Avoid boilerplate.

HASHTAGS
Use only relevant tags from: #AIxSmartData #PolicyGap #Opportunity #Threat #Monitor
#Urgent #Strategic #ImplementationRisk #RegulatoryChange #ConsumerProtection
#Interoperability #DigitalIdentity #Consent #Competition #EconomicSecurity #OpenFinance
#OpenProperty #Energy #Transport #Trade #Fraud #International #TrustFramework
#DataPortability. Maximum eight.
""").strip()


def _synthesis_instructions(mode: str) -> str:
    cadence = (
        "For WEEKLY mode, emphasise what actually changed, what has evidence behind it, and what merely merits monitoring."
        if mode == "weekly"
        else
        "For MONTHLY mode, emphasise recurring evidence-backed themes and distinguish them from isolated monitoring signals."
    )
    return dedent(f"""
    Prepare a concise {mode} synthesis for a UK Smart Data policy professional.
    {cadence}
    Do not convert commercial announcements into UK policy recommendations. Do not call a
    development a policy gap, opportunity or threat unless the supplied analyses contain
    evidence for that judgement. Prefer an empty section to filler. Use natural UK policy prose.
    """).strip()


def _article_prompt(article: Article, profile: dict, mode: str) -> str:
    max_chars = max(1600, int(os.getenv("RADAR_LLM_ARTICLE_CHARS", "3600")))
    body = article.text[:max_chars] if article.text else article.snippet[:max_chars]
    level = evidence_level(article)
    evidence_type = "full/partial page extract" if article.text else "RSS/discovery metadata only"
    return dedent(f"""
    MODE: {mode.upper()}
    INTEREST FRAME: {profile.get('mission', '')}
    EVIDENCE TYPE: {evidence_type}
    EVIDENCE LEVEL: {level}

    TITLE: {article.title}
    SOURCE: {article.source_name}
    AUTHOR: {article.author or 'Unknown'}
    PUBLISHED: {article.published_at or 'Unknown'}
    URL: {article.canonical_url}
    HEURISTIC RELEVANCE: {article.heuristic_score}
    MATCHED SIGNALS: {', '.join(article.matched_terms)}

    EVIDENCE
    {body}

    Produce a complete JSON object matching the schema. If evidence is only a signal,
    describe why it may be worth monitoring but do not infer UK adoption, policy gaps,
    regulatory shortcomings, market effects or recommendations.
    """).strip()


def _ollama_chat(*, prompt: str, system: str, schema: dict[str, Any], num_predict: int | None = None) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240"))
    predict = num_predict or int(os.getenv("OLLAMA_NUM_PREDICT", "1500"))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "6144")),
            "num_predict": predict,
        },
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


def _openai_parse(*, prompt: str, system: str, model_cls):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[openai]'` to use LLM_BACKEND=openai") from exc
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
        instructions=system,
        input=prompt,
        text_format=model_cls,
    )
    for output in response.output:
        if getattr(output, "type", None) == "message":
            for item in output.content:
                if getattr(item, "type", None) == "output_text" and getattr(item, "parsed", None):
                    return item.parsed
    raise RuntimeError("OpenAI returned no parsed structured output")


def _ollama_parse_with_retry(*, prompt: str, system: str, model_cls):
    schema = model_cls.model_json_schema()
    first = _ollama_chat(prompt=prompt, system=system, schema=schema)
    try:
        return model_cls.model_validate_json(first)
    except Exception as first_error:
        retry_prompt = prompt + "\n\nReturn a shorter COMPLETE JSON object. Omit unsupported PESTLE/SWOT content."
        second = _ollama_chat(
            prompt=retry_prompt,
            system=system,
            schema=schema,
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT_RETRY", "2100")),
        )
        try:
            return model_cls.model_validate_json(second)
        except Exception as second_error:
            raise RuntimeError(f"Local model returned invalid JSON twice: {first_error}; retry: {second_error}") from second_error


def _topic_phrase(article: Article) -> str:
    hay = " ".join([article.title, article.snippet, " ".join(article.matched_terms)]).lower()
    topics: list[str] = []
    if any(x in hay for x in ("agent", "artificial intelligence", " ai ", "llm")):
        topics.append("AI-enabled services")
    if any(x in hay for x in ("consent", "oauth", "identity", "trust")):
        topics.append("consent, identity or trust")
    if any(x in hay for x in ("api", "interoperab", "portability", "standard")):
        topics.append("interoperability and standards")
    if any(x in hay for x in ("open finance", "payment", "bank")):
        topics.append("open finance and payments")
    if "smart data" in hay:
        topics.append("Smart Data")
    return ", ".join(dict.fromkeys(topics)) or "the wider Smart Data and AI agenda"


def _apply_evidence_guardrails(article: Article, analysis: ArticleAnalysis) -> ArticleAnalysis:
    level = evidence_level(article)
    if level == "strong":
        return analysis

    if level == "partial":
        analysis.priority.evidence_strength = min(analysis.priority.evidence_strength, 3)
        if analysis.confidence == "high":
            analysis.confidence = "medium"
        return analysis

    # Headline/RSS-only items are signals, not a basis for UK policy conclusions.
    topic = _topic_phrase(article)
    analysis.confidence = "low"
    analysis.priority.evidence_strength = min(analysis.priority.evidence_strength, 2)
    analysis.priority.policy_advancement = min(analysis.priority.policy_advancement, 2)
    analysis.priority.opportunity = min(analysis.priority.opportunity, 3)
    analysis.priority.implementation_risk = min(analysis.priority.implementation_risk, 2)
    analysis.priority.urgency = min(analysis.priority.urgency, 3)
    analysis.priority.impact = min(analysis.priority.impact, 3)
    analysis.priority.consequences = min(analysis.priority.consequences, 3)
    analysis.priority.strategic_significance = min(analysis.priority.strategic_significance, 3)
    analysis.priority.rationale = (
        f"The headline is relevant to {topic}, but the radar does not have enough source text to support a UK policy conclusion. "
        "Treat this as a monitoring signal pending verification of the original article."
    )
    analysis.bottom_line = (
        f"{article.title} is a relevant signal for {topic}. The radar currently has headline/RSS evidence only, "
        "so it should be monitored rather than treated as a confirmed UK policy development."
    )
    analysis.summary = (
        f"{article.source_name or 'The publisher'} reported the development described in the headline. "
        "The full source text was not available to the radar, so no further factual claims are inferred."
    )
    analysis.why_it_matters = (
        f"The subject intersects with {topic}, which is relevant to the Smart Data programme. "
        "Its practical UK significance cannot be assessed reliably from the available metadata alone."
    )
    analysis.government_smart_data_perspective = (
        "Monitor and verify the original source. Do not infer a UK policy gap, regulatory weakness, adoption case or implementation requirement from this signal alone."
    )
    analysis.smart_data_ai_link = (
        f"Potential relevance to {topic}; the available metadata is not enough to establish a concrete UK Smart Data or AI policy implication."
    )
    analysis.source_perspective = (
        f"Monitoring signal from {article.source_name or 'the named publisher'}; the radar has not inferred the source's motives, position or UK applicability from metadata alone."
    )
    analysis.pestle = PestleAnalysis()
    analysis.swot = SwotAnalysis()
    analysis.policy_or_market_implications = ["Verify the original source before drawing any UK policy or market implication."]
    analysis.key_claims = []
    analysis.tensions_or_tradeoffs = []
    analysis.follow_up_questions = ["Does the full source contain evidence with a concrete implication for UK Smart Data policy or implementation?"]
    analysis.hashtags = ["#Monitor"]
    analysis.tags = []
    return analysis


def fallback_signal_analysis(article: Article) -> ArticleAnalysis:
    topic = _topic_phrase(article)
    relevance = max(25, min(70, round(article.heuristic_score)))
    return ArticleAnalysis(
        relevance_score=relevance,
        relevance_tier="high" if relevance >= 60 else "medium" if relevance >= 35 else "low",
        primary_category="monitoring_signal",
        bottom_line=f"{article.title} is a relevant signal for {topic}; source evidence is too limited for a firm UK policy conclusion.",
        summary=f"{article.source_name or 'The publisher'} reported the development in the headline. The radar could not obtain enough source text for a fuller verified assessment.",
        why_it_matters=f"The subject intersects with {topic}, but its UK significance remains unverified.",
        government_smart_data_perspective="Monitor the item and verify the original source before drawing policy implications.",
        smart_data_ai_link=f"Potential relevance to {topic}; no stronger claim is made without source text.",
        pestle=PestleAnalysis(),
        swot=SwotAnalysis(),
        priority=PrioritySignals(
            urgency=2, impact=2, consequences=2, policy_advancement=1, opportunity=2,
            monitoring_need=4, strategic_significance=2, implementation_risk=1,
            novelty=3, evidence_strength=1,
            rationale="Relevant discovery signal, but evidence is insufficient for policy judgement."
        ),
        policy_or_market_implications=["Verify the original source before drawing any UK policy or market implication."],
        key_claims=[],
        source_perspective="Monitoring signal only; the source's position has not been independently assessed.",
        tensions_or_tradeoffs=[],
        follow_up_questions=["Does the full source provide evidence that materially changes the UK Smart Data policy picture?"],
        hashtags=["#Monitor"], tags=[], confidence="low",
    )


def analyse_article(article: Article, profile: dict, mode: str = "weekly") -> ArticleAnalysis:
    if len(article.text or "") + len(article.snippet or "") < 80:
        return fallback_signal_analysis(article)
    if mode not in {"weekly", "monthly"}:
        raise ValueError("mode must be 'weekly' or 'monthly'")

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    prompt = _article_prompt(article, profile, mode)
    try:
        if backend == "ollama":
            result = _ollama_parse_with_retry(prompt=prompt, system=SYSTEM_INSTRUCTIONS, model_cls=ArticleAnalysis)
        elif backend == "openai":
            result = _openai_parse(prompt=prompt, system=SYSTEM_INSTRUCTIONS, model_cls=ArticleAnalysis)
        else:
            raise ValueError(f"Unsupported LLM_BACKEND={backend!r}")
    except Exception:
        return fallback_signal_analysis(article)
    return _apply_evidence_guardrails(article, result)


def analyse_digest(items: list[dict[str, Any]], mode: str = "weekly") -> DigestSynthesis | None:
    if not items:
        return None
    compact = "\n\n".join(
        f"RANK: {idx}\nTITLE: {item['title']}\nSOURCE: {item['source']}\nPRIORITY: {item['priority_score']}\n"
        f"BOTTOM LINE: {item['bottom_line']}\nWHY IT MATTERS: {item['why_it_matters']}\nHASHTAGS: {', '.join(item['hashtags'])}"
        for idx, item in enumerate(items[:12], start=1)
    )
    prompt = f"MODE: {mode.upper()}\n\nSYNTHESISE THESE RANKED ITEMS:\n\n" + compact
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    try:
        if backend == "ollama":
            result = _ollama_parse_with_retry(prompt=prompt, system=_synthesis_instructions(mode), model_cls=DigestSynthesis)
        elif backend == "openai":
            result = _openai_parse(prompt=prompt, system=_synthesis_instructions(mode), model_cls=DigestSynthesis)
        else:
            return None
        result.mode = mode
        return result
    except Exception:
        return None
