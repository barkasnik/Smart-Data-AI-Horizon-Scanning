from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

import httpx

from .models import Article, ArticleAnalysis, DigestSynthesis

SYSTEM_INSTRUCTIONS = dedent("""
You are the analytical editor for a UK Smart Data and AI policy-intelligence radar.
Assess developments from a UK Government / Smart Data programme perspective. Do not
promote or oppose any source or organisation.

NON-NEGOTIABLE QUALITY RULES
- Analyse the specific event, announcement, research finding, regulatory move or argument
  in the supplied evidence. Never turn an organisation homepage or navigation page into
  a policy development.
- Do not write generic Smart Data boilerplate. Every PESTLE/SWOT point must connect to a
  concrete fact, claim or implication in this item. If there is no material point for a
  category, return an empty list.
- Distinguish reported fact, source claim/opinion and your own analysis. Never invent
  corroboration, dates, legal changes, market effects or government intent.
- If only title/RSS snippet evidence is available, keep claims narrow, mark confidence
  low or medium, and do not pretend you read the full article.
- Raidiam is one monitored commercial source. Treat it neither as authoritative by
  default nor as an opponent; identify commercial perspective where relevant.

EDITORIAL PRIORITIES
Primary: Smart Data + AI, including agentic delegation, consent, digital identity,
APIs/trust frameworks, AI governance, interoperability, data portability and AI-enabled
services. Secondary: Smart Data implementation across finance, property, energy,
transport, retail, trade, agri-food and international data sharing.

PESTLE
Apply PESTLE only where evidenced: Political = ownership/priorities/coordination;
Economic = growth, competition, productivity, market structure, consumers/SMEs;
Social = trust, inclusion/accessibility; Technological = APIs, standards, identity, AI,
interoperability, security; Legal = DUAA/data protection/consumer or sector rules,
liability/consent; Environmental only if materially relevant.

SWOT
Use the position of the UK Smart Data programme. Strength = existing UK capability that
specifically helps with this development; Weakness = a programme/design/governance gap
specifically exposed; Opportunity = an external development HMG could exploit; Threat =
an external development that could obstruct, fragment or undermine objectives. Do not
fill all four quadrants unless the evidence supports them.

PRIORITY
Score urgency, impact, consequences, policy_advancement, opportunity, monitoring_need,
strategic_significance, implementation_risk, novelty and evidence_strength from 1-5.
Evidence strength must reflect the supplied source evidence, not confidence in your prose.

WRITING
Write like a strong UK policy analyst: natural, specific and economical. Bottom line first.
Summary 2-3 sentences. Maximum one concise point per PESTLE/SWOT field unless a second is
essential. Maximum two implications, two tensions and two follow-up questions. Avoid
'rapidly evolving landscape', 'underscores the importance', 'robust', 'holistic', 'delve',
'leveraging' and vague phrases such as 'this highlights'.

HASHTAGS
Use only relevant tags from: #AIxSmartData #PolicyGap #Opportunity #Threat #Monitor
#Urgent #Strategic #ImplementationRisk #RegulatoryChange #ConsumerProtection
#Interoperability #DigitalIdentity #Consent #Competition #EconomicSecurity #OpenFinance
#OpenProperty #Energy #Transport #Trade #Fraud #International #TrustFramework
#DataPortability. Maximum eight.
""").strip()


def _synthesis_instructions(mode: str) -> str:
    cadence = (
        "For WEEKLY mode, emphasise what actually changed in the period, immediate policy "
        "implications and what merits monitoring next."
        if mode == "weekly"
        else
        "For MONTHLY mode, emphasise recurring themes, cumulative implications, structural "
        "gaps/opportunities and signals likely to matter beyond the current month."
    )
    return dedent(f"""
    Prepare a concise {mode} synthesis for a UK Smart Data policy professional.
    {cadence}

    Only state patterns supported by at least two supplied items unless clearly labelled as
    a single high-significance development. Distinguish government/regulatory change from
    commercial advocacy and commentary. Do not invent a policy gap merely to populate a
    section. Prefer an empty list to generic filler. Use natural UK policy prose.
    """).strip()


def _article_prompt(article: Article, profile: dict, mode: str) -> str:
    interest = profile.get("mission", "")
    max_chars = max(1600, int(os.getenv("RADAR_LLM_ARTICLE_CHARS", "3600")))
    body = article.text[:max_chars] if article.text else article.snippet[:max_chars]
    evidence_type = "full/partial page extract" if article.text else "RSS/discovery metadata only"
    return dedent(f"""
    MODE: {mode.upper()}
    INTEREST FRAME: {interest}
    EVIDENCE TYPE: {evidence_type}

    TITLE: {article.title}
    SOURCE: {article.source_name}
    AUTHOR: {article.author or 'Unknown'}
    PUBLISHED: {article.published_at or 'Unknown'}
    URL: {article.canonical_url}
    HEURISTIC RELEVANCE: {article.heuristic_score}
    MATCHED SIGNALS: {', '.join(article.matched_terms)}

    EVIDENCE
    {body}

    Produce a complete JSON object matching the schema. Be concise enough to finish the
    object. If evidence is too thin for a particular PESTLE/SWOT category, use [].
    """).strip()


def _ollama_chat(
    *,
    prompt: str,
    system: str,
    schema: dict[str, Any],
    num_predict: int | None = None,
) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240"))
    predict = num_predict or int(os.getenv("OLLAMA_NUM_PREDICT", "1500"))
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


def _ollama_parse_with_retry(*, prompt: str, system: str, model_cls):
    schema = model_cls.model_json_schema()
    first = _ollama_chat(prompt=prompt, system=system, schema=schema)
    try:
        return model_cls.model_validate_json(first)
    except Exception as first_error:
        retry_prompt = (
            prompt
            + "\n\nYour previous structured response was incomplete or invalid. Return a shorter, "
              "complete JSON object now. Keep each prose field concise and omit unsupported "
              "PESTLE/SWOT points rather than expanding them."
        )
        retry_tokens = int(os.getenv("OLLAMA_NUM_PREDICT_RETRY", "2100"))
        second = _ollama_chat(
            prompt=retry_prompt,
            system=system,
            schema=schema,
            num_predict=retry_tokens,
        )
        try:
            return model_cls.model_validate_json(second)
        except Exception as second_error:
            raise RuntimeError(
                f"Local model returned invalid structured JSON twice: {first_error}; retry: {second_error}"
            ) from second_error


def analyse_article(article: Article, profile: dict, mode: str = "weekly") -> ArticleAnalysis:
    evidence_len = len(article.text or "") + len(article.snippet or "")
    if evidence_len < 80:
        raise ValueError("Not enough source evidence for analysis")
    if mode not in {"weekly", "monthly"}:
        raise ValueError("mode must be 'weekly' or 'monthly'")

    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    prompt = _article_prompt(article, profile, mode)
    if backend == "ollama":
        return _ollama_parse_with_retry(
            prompt=prompt,
            system=SYSTEM_INSTRUCTIONS,
            model_cls=ArticleAnalysis,
        )
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
        for idx, item in enumerate(items[:12], start=1)
    )
    prompt = f"MODE: {mode.upper()}\n\nSYNTHESISE THESE RANKED ITEMS:\n\n" + compact
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    system = _synthesis_instructions(mode)
    if backend == "ollama":
        result = _ollama_parse_with_retry(
            prompt=prompt,
            system=system,
            model_cls=DigestSynthesis,
        )
        result.mode = mode
        return result
    if backend == "openai":
        result = _openai_parse(prompt=prompt, system=system, model_cls=DigestSynthesis)
        result.mode = mode
        return result
    return None
