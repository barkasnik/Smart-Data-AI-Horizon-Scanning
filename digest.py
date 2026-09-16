from __future__ import annotations

import html
import json
from pathlib import Path

from .models import AnalysedArticle, DigestSynthesis
from .utils import utcnow


def _date(value) -> str:
    return value.strftime("%Y-%m-%d") if value else "Date unknown"


def _bullet_section(lines: list[str], heading: str, values: list[str]) -> None:
    if not values:
        return
    lines.append(f"**{heading}:**")
    lines.extend([f"- {v}" for v in values])
    lines.append("")


def _framework_section(lines: list[str], heading: str, data: dict[str, list[str]]) -> None:
    populated = [(k, v) for k, v in data.items() if v]
    if not populated:
        return
    lines += [f"### {heading}", ""]
    for key, values in populated:
        lines.append(f"**{key.title()}:**")
        lines.extend([f"- {v}" for v in values])
        lines.append("")


def render_markdown(items: list[AnalysedArticle], synthesis: DigestSynthesis | None = None, mode: str = "weekly") -> str:
    generated = utcnow().strftime("%Y-%m-%d %H:%M UTC")
    label = "Weekly" if mode == "weekly" else "Monthly"
    lines = [
        f"# Smart Data & AI Radar — {label} Briefing",
        "",
        f"Generated: **{generated}**",
        "",
        "> Analytical lens: UK Government / Smart Data programme. Sources are monitored for insight, not endorsement. Facts, source viewpoints and radar analysis are kept distinct.",
        "",
    ]
    if synthesis:
        lines += [
            "## What matters in this period",
            "",
            f"### {synthesis.headline}",
            "",
            synthesis.executive_summary,
            "",
        ]
        sections = [
            ("Ranked priorities", synthesis.ranked_priorities),
            ("Emerging patterns", synthesis.emerging_patterns),
            ("AI + Smart Data", synthesis.ai_smart_data_developments),
            ("Wider Smart Data", synthesis.wider_smart_data_developments),
            ("Policy gaps", synthesis.policy_gaps),
            ("Opportunities", synthesis.opportunities),
            ("Threats", synthesis.threats),
            ("Tensions to watch", synthesis.tensions_to_watch),
            ("Monitoring list", synthesis.monitoring_list),
            ("Questions for policy teams", synthesis.questions_for_policy_teams),
        ]
        for heading, values in sections:
            _bullet_section(lines, heading, values)
        lines += ["---", ""]

    if not items:
        lines += ["No articles passed the relevance and analysis thresholds in this run.", ""]

    for i, item in enumerate(sorted(items, key=lambda x: x.final_score, reverse=True), start=1):
        a, x = item.article, item.analysis
        p = x.priority
        lines += [
            f"## {i}. [{a.title}]({a.canonical_url})",
            "",
            f"**Overall rank:** {item.final_score:.1f}/100 · **Policy priority:** {item.priority_score:.1f}/100 · **Relevance:** {x.relevance_score}/100 · **Tier:** {x.relevance_tier}",
            "",
            f"**Source:** {a.source_name or 'Unknown'} · **Published:** {_date(a.published_at)}",
            "",
            f"**Bottom line:** {x.bottom_line}",
            "",
            f"**What happened:** {x.summary}",
            "",
            f"**Why it matters for Government Smart Data:** {x.government_smart_data_perspective}",
            "",
            f"**AI / Smart Data connection:** {x.smart_data_ai_link}",
            "",
            f"**Source perspective:** {x.source_perspective}",
            "",
            f"**Priority rationale:** {p.rationale}",
            "",
            "**Priority signals (1–5):**",
            "",
            f"Urgency **{p.urgency}** · Impact **{p.impact}** · Consequences **{p.consequences}** · Policy advancement **{p.policy_advancement}** · Opportunity **{p.opportunity}** · Monitoring **{p.monitoring_need}** · Strategic significance **{p.strategic_significance}** · Implementation risk **{p.implementation_risk}** · Novelty **{p.novelty}** · Evidence **{p.evidence_strength}**",
            "",
        ]

        _framework_section(lines, "PESTLE — Government / Smart Data perspective", x.pestle.model_dump())
        _framework_section(lines, "SWOT — UK Smart Data programme", x.swot.model_dump())
        _bullet_section(lines, "Policy / market implications", x.policy_or_market_implications)
        _bullet_section(lines, "Tensions / trade-offs", x.tensions_or_tradeoffs)
        _bullet_section(lines, "Questions to pursue", x.follow_up_questions)
        lines += [
            f"**Hashtags:** {' '.join(x.hashtags) if x.hashtags else '#Monitor'}",
            "",
            f"**Confidence:** {x.confidence}",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def render_html(items: list[AnalysedArticle], synthesis: DigestSynthesis | None = None, mode: str = "weekly") -> str:
    label = "Weekly" if mode == "weekly" else "Monthly"
    synthesis_html = ""
    if synthesis:
        sections = []
        for heading, values in [
            ("Ranked priorities", synthesis.ranked_priorities),
            ("Policy gaps", synthesis.policy_gaps),
            ("Opportunities", synthesis.opportunities),
            ("Threats", synthesis.threats),
            ("Monitoring list", synthesis.monitoring_list),
        ]:
            if values:
                lis = "".join(f"<li>{html.escape(v)}</li>" for v in values)
                sections.append(f"<h3>{html.escape(heading)}</h3><ul>{lis}</ul>")
        synthesis_html = (
            f"<section><h2>What matters in this period</h2><h3>{html.escape(synthesis.headline)}</h3>"
            f"<p>{html.escape(synthesis.executive_summary)}</p>{''.join(sections)}</section>"
        )

    cards = []
    for item in sorted(items, key=lambda x: x.final_score, reverse=True):
        a, x = item.article, item.analysis
        implications = "".join(f"<li>{html.escape(v)}</li>" for v in x.policy_or_market_implications)
        pestle = "".join(
            f"<li><strong>{html.escape(k.title())}:</strong> {html.escape(' | '.join(v))}</li>"
            for k, v in x.pestle.model_dump().items() if v
        )
        swot = "".join(
            f"<li><strong>{html.escape(k.title())}:</strong> {html.escape(' | '.join(v))}</li>"
            for k, v in x.swot.model_dump().items() if v
        )
        hashtags = " ".join(html.escape(v) for v in x.hashtags)
        cards.append(f"""
        <article>
          <h2><a href="{html.escape(a.canonical_url)}">{html.escape(a.title)}</a></h2>
          <p class="meta">Rank {item.final_score:.1f}/100 · Policy priority {item.priority_score:.1f}/100 · {html.escape(x.relevance_tier)} · {html.escape(a.source_name or 'Unknown')} · {_date(a.published_at)}</p>
          <p class="bottom"><strong>Bottom line:</strong> {html.escape(x.bottom_line)}</p>
          <p><strong>What happened:</strong> {html.escape(x.summary)}</p>
          <p><strong>Why it matters for Government Smart Data:</strong> {html.escape(x.government_smart_data_perspective)}</p>
          <p><strong>Priority rationale:</strong> {html.escape(x.priority.rationale)}</p>
          <h3>PESTLE</h3><ul>{pestle}</ul>
          <h3>SWOT</h3><ul>{swot}</ul>
          <h3>Implications</h3><ul>{implications}</ul>
          <p class="tags">{hashtags}</p>
        </article>
        """)
    generated = utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Data & AI Radar — {label}</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:980px;margin:auto;padding:2rem;line-height:1.55;color:#161616}}
a{{color:inherit}}
article,section{{border:1px solid #ddd;border-radius:12px;padding:1.2rem 1.4rem;margin:1rem 0}}.meta,.tags{{color:#555}}.bottom{{font-size:1.06rem}}h1{{margin-bottom:.2rem}}
</style></head><body><h1>Smart Data & AI Radar — {label} Briefing</h1><p>Generated {generated}</p>
<p><em>UK Government / Smart Data analytical lens. Sources are monitored for insight, not endorsement.</em></p>{synthesis_html}{''.join(cards)}</body></html>"""


def write_outputs(
    items: list[AnalysedArticle],
    out_dir: str | Path = "output",
    synthesis: DigestSynthesis | None = None,
    mode: str = "weekly",
) -> tuple[Path, Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{mode}-latest"
    md = out / f"{stem}.md"
    js = out / f"{stem}.json"
    ht = out / f"{stem}.html"
    md.write_text(render_markdown(items, synthesis, mode), encoding="utf-8")
    # Public-safe JSON: keep provenance and analysis, but never publish the
    # scraped article body (or discovery snippet) into a public repository.
    public_articles = []
    for item in items:
        row = item.model_dump(mode="json")
        row["article"].pop("text", None)
        row["article"].pop("snippet", None)
        public_articles.append(row)

    payload = {
        "mode": mode,
        "generated_at": utcnow().isoformat(),
        "synthesis": synthesis.model_dump(mode="json") if synthesis else None,
        "articles": public_articles,
    }
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ht.write_text(render_html(items, synthesis, mode), encoding="utf-8")
    return md, js, ht

