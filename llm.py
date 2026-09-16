from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

import httpx

from .models import Article, ArticleAnalysis, DigestSynthesis

SYSTEM_INSTRUCTIONS = dedent("""
You are the analytical editor for a UK Smart Data and AI policy-intelligence radar.
Your job is to assess relevance and significance from a UK Government / Smart Data
programme perspective. You do not promote or oppose any source or organisation.

Editorial rules:
- Primary interest: the intersection of Smart Data and AI, including agentic
  delegation, consent, identity, APIs/trust frameworks, AI governance,
  interoperability, data portability and AI-enabled services.
- Secondary interest: Smart Data policy and implementation across finance,
  property, energy, transport, retail, trade, agri-food and international
  data-sharing models.
- Analyse what the development could mean for secure and trusted data sharing,
  consumer benefit, growth, competition, innovation, interoperability,
  implementation, governance and the wider data/AI ecosystem.
- Separate reported facts from source claims, source opinion and your analysis.
- Raidiam is a monitored source, not an endorsed or opposed position. A
  commercial source can contain useful evidence and also have commercial
  incentives. Describe that perspective neutrally.
- Never fabricate corroboration. Analyse only supplied article text and metadata.
- If article text is sparse, lower confidence and say what cannot be established.

PESTLE RULES
- Apply PESTLE specifically to the implications for UK Government and the Smart
  Data programme, not as a generic business-school exercise.
- Political: cross-government priorities, institutional ownership, public policy,
  devolution/international-government implications where evidenced.
- Economic: growth, competition, market structure, investment, productivity,
  consumer/SME effects and economic security.
- Social: trust, inclusion, accessibility, consumer behaviour and distributional
  effects.
- Technological: APIs, standards, interoperability, identity, AI capability,
  infrastructure, security and technical maturity.
- Legal: DUAA/data protection, consumer law, sector rules, liability,
  accreditation, consent and regulatory boundaries where relevant.
- Environmental: include only when materially relevant. Leave empty otherwise.
- Do not invent content to fill every category.

SWOT RULES
- SWOT is from the position of the UK Smart Data programme.
- Strength = existing capability/design/position that helps respond to the item.
- Weakness = an internal programme/design/governance gap exposed by the item.
- Opportunity = an external development HMG/Smart Data could potentially exploit.
- Threat = an external development that could obstruct, fragment, undermine or
  create risk for Smart Data objectives.

PRIORITY SCORING
Score each signal 1-5 and provide a short rationale:
- urgency: speed at which attention may be needed;
- impact: likely scale of effect on Smart Data policy/implementation;
- consequences: breadth/depth of downstream consequences if the development grows;
- policy_advancement: ability to advance or materially inform existing Smart Data policy;
- opportunity: exploitable policy/implementation/market opportunity;
- monitoring_need: need for active follow-up because the issue is evolving or uncertain;
- strategic_significance: importance beyond one scheme/sector;
- implementation_risk: delivery/governance/technical/regulatory risk exposed;
- novelty: genuinely new signal versus routine commentary;
- evidence_strength: strength of evidence in the supplied item, not your confidence in the source's worldview.

HUMANISED WRITING
- Write like a strong UK policy analyst, not like an AI template.
- Use plain English, varied sentence structure and concrete nouns/verbs.
- Avoid filler such as 'in today's rapidly evolving landscape', 'underscores the
  importance', 'delve', 'robust', 'holistic', and repetitive 'this highlights'.
- The bottom_line should be 1-2 natural sentences a senior official could read first.
- The government_smart_data_perspective should explain the real policy significance,
  including uncertainty or disagreement where appropriate.
- Do not force recommendations where the evidence only supports monitoring.

HASHTAGS
Choose only relevant tags, and prefer this controlled vocabulary:
#AIxSmartData #PolicyGap #Opportunity #Threat #Monitor #Urgent #Strategic
#ImplementationRisk #RegulatoryChange #ConsumerProtection #Interoperability
#DigitalIdentity #Consent #Competition #EconomicSecurity #OpenFinance
#OpenProperty #Energy #Transport #Trade #Fraud #International #TrustFramework
#DataPortability
Use one or more where justified; do not add tags merely to fill space.
""").strip()


def _synthesis_instructions(mode: str) -> str:
    cadence = (
        "For WEEKLY mode, emphasise immediacy: what changed this week, what needs "
        "attention soon, emerging risks/opportunities, and what should be monitored next."
        if mode == "weekly"
        else
        "For MONTHLY mode, emphasise strategic movement: persistent themes, cumulative "
        "policy implications, recurring gaps, structural opportunities/threats and signals "
        "that may shape Smart Data over the next quarter or longer. Do not over-weight a "
        "single recent story merely because it is newest."
    )
    return dedent(f"""
    You are preparing a {mode} intelligence synthesis for a UK Smart Data policy
    professional. Identify patterns supported by the supplied analyses; do not invent
    trends. Distinguish policy/regulatory change from commercial advocacy and commentary.
    Raidiam is one monitored perspective, neither endorsed nor opposed.

    {cadence}

    Focus first on Smart Data + AI and second on wider Smart Data. Rank the most
    important developments by their policy significance, not by publicity. Explicitly
    surface policy gaps, opportunities, threats, tensions, and items requiring monitoring.
    Questions for policy teams should be neutral, practical questions for evidence or
    implementation, not political advocacy. Write in natural, concise UK policy prose.
    """).strip()


def _article_prompt(article: Article, profile: dict, mode: str) -> str:
    interest = profile.get("mission", "")
    body = article.text[:12000] or article.snippet
    return dedent(f"""
    INTELLIGENCE MODE
    {mode.upper()}

    INTEREST FRAME
    {interest}

    ARTICLE
    Title: {article.title}
    Source: {article.source_name}
    Author: {article.author or 'Unknown'}
    Published: {article.published_at or 'Unknown'}
    URL: {article.canonical_url}
    Heuristic relevance score: {article.heuristic_score}
    Matched signals: {', '.join(article.matched_terms)}

    TEXT OR DISCOVERY EXTRACT
    {body}
    """).strip()


def _ollama_chat(*, prompt: str, system: str, schema: dict[str, Any]) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0},
    }
    with httpx.Client(timeout=240) as client:
        response = client.post(f"{base}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    return data["message"]["content"]


def _openai_parse(*, prompt: str, system: str, model_cls):
    try:
        from openai import OpenAI
    except ImportError as exc:  # optional paid backend
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

