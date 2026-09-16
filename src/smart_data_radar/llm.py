from __future__ import annotations

import os
import re
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

SYSTEM_INSTRUCTIONS = dedent("""
You are the analytical editor for a UK Smart Data and AI policy-intelligence radar.
Assess relevance and significance from a UK Government / Smart Data programme
perspective. Do not promote or oppose any source or organisation.

Editorial priorities:
- Primary: Smart Data + AI, including agentic delegation, consent, identity,
  APIs/trust frameworks, AI governance, interoperability, data portability and
  AI-enabled services.
- Secondary: Smart Data policy and implementation across finance, property,
  energy, transport, retail, trade, agri-food and international data sharing.
- Analyse implications for secure/trusted data sharing, consumer benefit, growth,
  competition, innovation, interoperability, implementation and governance.
- Separate reported facts, source claims/opinion and your own analysis.
- Raidiam is monitored for insight, not endorsed or opposed.
- Never fabricate corroboration. If evidence is sparse, lower confidence.

PESTLE
Apply PESTLE to UK Government / Smart Data implications, not generic business
analysis. Political = ownership/priorities/coordination; Economic = growth,
competition, productivity, market structure, consumers/SMEs; Social = trust,
inclusion/accessibility; Technological = APIs, standards, identity, AI,
interoperability, security; Legal = DUAA/data protection/consumer or sector
rules/liability/consent. Environmental only when materially relevant.
Do not invent points merely to fill categories.

SWOT
Use the position of the UK Smart Data programme. Strength = existing UK
capability/design that helps; Weakness = internal programme/design/governance
gap exposed; Opportunity = external development HMG could exploit; Threat =
external development that could obstruct, fragment or undermine objectives.

PRIORITY
Score urgency, impact, consequences, policy_advancement, opportunity,
monitoring_need, strategic_significance, implementation_risk, novelty and
evidence_strength from 1-5, with a short rationale.

WRITING
Write like a strong UK policy analyst in plain English. Bottom line: 1-2
sentences. Summary: 2-3 sentences. Keep every PESTLE and SWOT category to no
more than one concise point unless a second is essential. Use no more than two
policy implications, two tensions and two follow-up questions. Avoid boilerplate
phrases such as 'rapidly evolving landscape', 'underscores the importance',
'delve', 'robust', 'holistic' and repetitive 'this highlights'. Do not force a
recommendation where monitoring is enough.

HASHTAGS
Use only relevant tags from: #AIxSmartData #PolicyGap #Opportunity #Threat
#Monitor #Urgent #Strategic #ImplementationRisk #RegulatoryChange
#ConsumerProtection #Interoperability #DigitalIdentity #Consent #Competition
#EconomicSecurity #OpenFinance #OpenProperty #Energy #Transport #Trade #Fraud
#International #TrustFramework #DataPortability. Maximum eight.
""").strip()


def _synthesis_instructions(mode: str) -> str:
    cadence = (
        "For WEEKLY mode, emphasise what changed this week, what needs attention soon, "
        "emerging risks/opportunities and what should be monitored next."
        if mode == "weekly"
        else
        "For MONTHLY mode, emphasise persistent themes, cumulative implications, recurring "
        "gaps, structural opportunities/threats and signals for the next quarter or longer."
    )
    return dedent(f"""
    Prepare a concise {mode} intelligence synthesis for a UK Smart Data policy
    professional. Identify only patterns supported by the supplied analyses. Distinguish
    regulatory/policy change from commercial advocacy and commentary. Raidiam is one
    monitored perspective, neither endorsed nor opposed.

    {cadence}

    Focus first on Smart Data + AI and second on wider Smart Data. Rank by policy
    significance, not publicity. Surface policy gaps, opportunities, threats, tensions
    and monitoring needs. Use natural, concise UK policy prose.
    """).strip()


def _article_prompt(article: Article, profile: dict, mode: str) -> str:
    interest = profile.get("mission", "")
    max_chars = max(1800, int(os.getenv("RADAR_LLM_ARTICLE_CHARS", "4500")))
    body = article.text[:max_chars] or article.snippet[:max_chars]
    return dedent(f"""
    MODE: {mode.upper()}
    INTEREST FRAME: {interest}

    ARTICLE
    Title: {article.title}
    Source: {article.source_name}
    Author: {article.author or 'Unknown'}
    Published: {article.published_at or 'Unknown'}
    URL: {article.canonical_url}
    Heuristic relevance: {article.heuristic_score}
    Matched signals: {', '.join(article.matched_terms)}

    EXTRACT
    {body}
    """).strip()


def _ollama_chat(*, prompt: str, system: str, schema: dict[str, Any]) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "950")),
        },
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    return data["message"]["content"]


def _openai_parse(*, prompt: str, system: str, model_cls):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[openai]'` to use LLM_BACKEND=openai") from exc
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    response = client.responses.parse(
        model=model,
        instructions=system,
        input=prompt,
        text_format=model_cls,
    )
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for item in output.content:
            if getattr(item, "type", None) == "output_text" and getattr(item, "parsed", None):
                return item.parsed
    raise RuntimeError("OpenAI returned no parsed structured output")


def _clean_excerpt(value: str, limit: int = 850) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return "The source extract is limited; the title and metadata provide the main evidence available to the radar."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen = " ".join(sentences[:2]).strip()
    if len(chosen) > limit:
        chosen = chosen[: limit - 1].rstrip() + "…"
    return chosen


def _clamp_1_5(value: int) -> int:
    return max(1, min(5, value))


def fallback_analysis(article: Article, mode: str = "weekly") -> ArticleAnalysis:
    """Fast, grounded fallback used when the local LLM times out.

    It deliberately avoids pretending to have performed semantic analysis beyond the
    supplied text/metadata. The item remains rankable and visible instead of vanishing
    from the weekly briefing.
    """
    matched = " ".join(article.matched_terms).lower()
    haystack = f"{article.title} {article.snippet} {article.text[:1200]}".lower()
    summary = _clean_excerpt(article.snippet or article.text)
    score = int(max(0, min(100, round(article.heuristic_score))))

    if "ai + smart data" in matched or ("smart data" in haystack and (" ai " in f" {haystack} " or "artificial intelligence" in haystack)):
        category = "ai_smart_data"
        link_text = "The item sits directly at the Smart Data/AI intersection and should be read for implications for trusted data access, delegation, governance or AI-enabled services."
    elif "open finance" in haystack or "open_finance" in matched:
        category = "open_finance"
        link_text = "The item is primarily a Smart Data/open finance development; any AI implications are secondary unless the source provides specific evidence."
    elif "property" in haystack:
        category = "open_property"
        link_text = "The item is primarily a Smart Data/property development; monitor interoperability, identity and trusted data-sharing implications."
    elif "energy" in haystack:
        category = "energy"
        link_text = "The item is primarily a Smart Data/energy development; monitor consumer, interoperability and data-governance implications."
    elif "transport" in haystack:
        category = "transport"
        link_text = "The item is primarily a Smart Data/transport development; monitor interoperability and consumer-use-case implications."
    else:
        category = "smart_data"
        link_text = "The item is relevant to the wider Smart Data agenda; the AI connection is not strong enough in the supplied extract to overstate it."

    policy_topics: list[str] = []
    if any(k in matched + " " + haystack for k in ["interoperab", "api", "standard", "portability"]):
        policy_topics.append("interoperability and standards")
    if any(k in matched + " " + haystack for k in ["consent", "identity", "trust", "authorised", "authorized"]):
        policy_topics.append("trust, consent and identity")
    if any(k in matched + " " + haystack for k in ["govern", "regulat", "fca", "duaa", "data use"]):
        policy_topics.append("governance and regulatory implementation")
    if any(k in matched + " " + haystack for k in ["competition", "market", "growth", "consumer"]):
        policy_topics.append("competition and consumer outcomes")
    if any(k in matched + " " + haystack for k in [" ai ", "artificial intelligence", "agentic", "llm"]):
        policy_topics.append("AI-enabled use and accountability")
    if not policy_topics:
        policy_topics.append("Smart Data implementation and evidence")

    topics_phrase = ", ".join(policy_topics[:3])
    perspective = (
        f"For UK Smart Data policy, this is worth monitoring because it touches {topics_phrase}. "
        "The fallback assessment keeps the item in the ranked briefing, but the source should be opened before relying on any detailed policy conclusion."
    )

    technical = []
    legal = []
    political = []
    economic = []
    social = []
    environmental = []

    if any(k in topics_phrase for k in ["interoperability", "AI-enabled", "identity"]):
        technical.append("Assess whether the development changes requirements for APIs, standards, identity, interoperability or AI integration across Smart Data schemes.")
    if any(k in topics_phrase for k in ["governance", "consent", "accountability"]):
        legal.append("Check the development against data protection, DUAA powers, consent/accountability and relevant sector rules before treating it as implementation-ready.")
    if "governance" in topics_phrase or any(k in haystack for k in ["government", "regulator", "fca", "policy"]):
        political.append("Monitor whether the development affects institutional ownership, regulatory sequencing or cross-government coordination for Smart Data.")
    if any(k in topics_phrase for k in ["competition", "consumer"]):
        economic.append("Consider possible effects on competition, innovation, market entry and consumer value as the development matures.")
    if any(k in topics_phrase for k in ["trust", "consent", "consumer"]):
        social.append("Consumer trust, meaningful consent, accessibility and inclusion are relevant implementation tests if the development moves beyond experimentation.")
    if "energy" in category and any(k in haystack for k in ["carbon", "climate", "emission", "net zero"]):
        environmental.append("The item may have a material environmental dimension through energy use or decarbonisation outcomes; evidence should be checked in the source.")

    pestle = PestleAnalysis(
        political=political,
        economic=economic,
        social=social,
        technological=technical,
        legal=legal,
        environmental=environmental,
    )

    swot = SwotAnalysis(
        strengths=["Existing UK Smart Data work on trusted, interoperable data sharing provides a policy frame for assessing this development."],
        weaknesses=["The item may expose implementation or governance dependencies that are not resolved in the supplied extract; further evidence is needed before drawing a firm conclusion."],
        opportunities=[f"Use the development as evidence for {topics_phrase}, particularly where it can inform scheme design or cross-sector learning."],
        threats=["If similar developments scale without compatible standards or clear accountability, they could add fragmentation or consumer-trust risk to the wider Smart Data ecosystem."],
    )

    base = _clamp_1_5(round(1 + score / 25))
    urgency = 4 if any(k in haystack for k in ["launch", "announc", "new rule", "deadline", "consultation", "roadmap"]) else 3
    evidence_strength = 3 if len(article.text) >= 800 else 2
    priority = PrioritySignals(
        urgency=_clamp_1_5(urgency),
        impact=base,
        consequences=_clamp_1_5(base),
        policy_advancement=_clamp_1_5(max(2, base - 1)),
        opportunity=_clamp_1_5(base),
        monitoring_need=_clamp_1_5(max(3, base)),
        strategic_significance=_clamp_1_5(base),
        implementation_risk=_clamp_1_5(max(2, base - 1)),
        novelty=_clamp_1_5(max(2, base - 1)),
        evidence_strength=evidence_strength,
        rationale=(
            "Retained because it is a fresh, high-scoring Smart Data signal. The local LLM did not complete, "
            "so priority is based on the radar's relevance signals and should be treated as provisional."
        ),
    )

    if score >= 80:
        tier = "must_read"
    elif score >= 60:
        tier = "high"
    elif score >= 35:
        tier = "medium"
    else:
        tier = "low"

    tags = [t for t in article.matched_terms[:12]]
    return ArticleAnalysis(
        relevance_score=score,
        relevance_tier=tier,
        primary_category=category,
        bottom_line=f"{article.title} is a fresh Smart Data signal from {article.source_name or 'the monitored source set'}. It has been retained for policy review even though the local AI analysis timed out.",
        summary=summary,
        why_it_matters=perspective,
        government_smart_data_perspective=perspective,
        smart_data_ai_link=link_text,
        pestle=pestle,
        swot=swot,
        priority=priority,
        policy_or_market_implications=[
            f"Check the original source for concrete implications for {topics_phrase}.",
            "Compare the development with current UK Smart Data scheme design before treating it as transferable across sectors.",
        ],
        key_claims=[],
        source_perspective=f"Monitored source: {article.source_name or 'unknown'}. The fallback does not infer the source's motives or endorse its position.",
        tensions_or_tradeoffs=["Speed of innovation versus interoperability, accountability and consumer trust may require attention if the development progresses."],
        follow_up_questions=[
            "What has materially changed in the last seven days, and is the change policy, implementation or commentary?",
            "Does the source provide evidence that would justify changing UK Smart Data policy or is monitoring sufficient?",
        ],
        hashtags=["#Monitor"],
        tags=tags,
        confidence="low" if evidence_strength == 2 else "medium",
    )


def analyse_article(article: Article, profile: dict, mode: str = "weekly") -> ArticleAnalysis:
    if len(article.text) < 250 and len(article.snippet) < 80:
        raise ValueError("Not enough extracted text for reliable LLM analysis")
    if mode not in {"weekly", "monthly"}:
        raise ValueError("mode must be 'weekly' or 'monthly'")

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    prompt = _article_prompt(article, profile, mode)
    if backend == "ollama":
        raw = _ollama_chat(
            prompt=prompt,
            system=SYSTEM_INSTRUCTIONS,
            schema=ArticleAnalysis.model_json_schema(),
        )
        return ArticleAnalysis.model_validate_json(raw)
    if backend == "openai":
        return _openai_parse(prompt=prompt, system=SYSTEM_INSTRUCTIONS, model_cls=ArticleAnalysis)
    raise ValueError(f"Unsupported LLM_BACKEND={backend!r}")


def analyse_digest(items: list[dict[str, Any]], mode: str = "weekly") -> DigestSynthesis | None:
    if not items:
        return None
    if mode not in {"weekly", "monthly"}:
        raise ValueError("mode must be 'weekly' or 'monthly'")

    compact = "\n\n".join(
        f"RANK: {idx}\nTITLE: {item['title']}\nSOURCE: {item['source']}\n"
        f"PRIORITY SCORE: {item['priority_score']}\nBOTTOM LINE: {item['bottom_line']}\n"
        f"WHY IT MATTERS: {item['why_it_matters']}\nHASHTAGS: {', '.join(item['hashtags'])}"
        for idx, item in enumerate(items[:20], start=1)
    )
    prompt = f"MODE: {mode.upper()}\n\nSYNTHESISE THESE RANKED ITEMS:\n\n" + compact
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    system = _synthesis_instructions(mode)
    if backend == "ollama":
        raw = _ollama_chat(
            prompt=prompt,
            system=system,
            schema=DigestSynthesis.model_json_schema(),
        )
        result = DigestSynthesis.model_validate_json(raw)
        result.mode = mode
        return result
    if backend == "openai":
        result = _openai_parse(prompt=prompt, system=system, model_cls=DigestSynthesis)
        result.mode = mode
        return result
    return None
