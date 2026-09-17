from __future__ import annotations

import os
from datetime import timedelta, timezone
from pathlib import Path

import typer

from .config import load_default_config
from .dedupe import dedupe_articles, dedupe_candidates
from .digest import write_outputs
from .discover import discover_index_sources, discover_search
from .extract import RobotsCache, extract_candidate
from .llm import analyse_article, analyse_digest
from .models import AnalysedArticle
from .prioritise import final_rank_score, normalise_hashtags, priority_score
from .quality import content_quality, is_hub, select_diverse
from .score import heuristic_score
from .storage import Store
from .utils import utcnow

app = typer.Typer(add_completion=False, help="Smart Data & AI news intelligence radar")


def substantive(analysis) -> bool:
    """Reject vague or under-evidenced prose before it reaches the public briefing."""
    swot_n = sum(
        len(getattr(analysis.swot, key))
        for key in ("strengths", "weaknesses", "opportunities", "threats")
    )
    return (
        analysis.relevance_score >= 35
        and len((analysis.bottom_line or "").strip()) >= 45
        and len((analysis.summary or "").strip()) >= 55
        and len((analysis.government_smart_data_perspective or "").strip()) >= 70
        and swot_n >= 1
        and len(analysis.policy_or_market_implications) >= 1
    )


@app.command()
def run(
    mode: str = typer.Option("weekly"),
    days: int | None = typer.Option(None),
    candidate_limit: int = typer.Option(int(os.getenv("RADAR_CANDIDATE_LIMIT", "160"))),
    analyse_limit: int | None = typer.Option(None),
    min_score: float = typer.Option(float(os.getenv("RADAR_MIN_SCORE", "30"))),
    backend: str = typer.Option(os.getenv("SEARCH_BACKEND", "google_news")),
    db: Path = typer.Option(Path("radar.db")),
    out_dir: Path = typer.Option(Path("output")),
    no_llm: bool = typer.Option(False),
    no_synthesis: bool = typer.Option(False),
) -> None:
    if mode not in {"weekly", "monthly"}:
        raise typer.BadParameter("--mode must be weekly or monthly")

    profile, sources = load_default_config()
    days = days if days is not None else (7 if mode == "weekly" else 30)
    default_desired = 8 if mode == "weekly" else 12
    desired = int(os.getenv("RADAR_ANALYSE_LIMIT", str(default_desired))) if analyse_limit is None else analyse_limit
    attempts = max(desired * 2, int(os.getenv("RADAR_ANALYSIS_ATTEMPTS", "18")))
    per_query = int(os.getenv("RADAR_RESULTS_PER_QUERY", "10"))
    quality_floor = float(os.getenv("RADAR_CONTENT_QUALITY", "45"))

    typer.echo(f"Mode: {mode} · window: {days} days · desired maximum: {desired}")

    candidates = []
    try:
        candidates.extend(discover_search(profile, backend=backend, per_query=per_query, days=days))
    except Exception as exc:
        typer.echo(f"Search discovery warning: {exc}")
    candidates.extend(discover_index_sources(sources))
    candidates = dedupe_candidates(candidates)
    typer.echo(f"Discovered {len(candidates)} unique candidates")

    cutoff = utcnow() - timedelta(days=days)
    robots = RobotsCache()
    recent: list = []

    for candidate in candidates:
        try:
            article = extract_candidate(candidate, robots)
        except Exception:
            continue

        if is_hub(article):
            continue

        published = article.published_at
        if published and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)

        # Horizon scanning should be time-bounded. Undated pages are excluded rather
        # than treated as fresh simply because they were rediscovered this week.
        if not published or published < cutoff:
            continue

        quality = content_quality(article)
        if quality < quality_floor:
            continue

        score, matched = heuristic_score(article, profile, sources)
        article.heuristic_score = score
        article.matched_terms = matched
        if score < min_score:
            continue

        recent.append(article)

    recent = dedupe_articles(recent)[:candidate_limit]
    pool = select_diverse(
        recent,
        limit=min(len(recent), attempts),
        max_govuk=3,
        max_per_domain=2,
    )
    typer.echo(f"Selected {len(pool)} recent, evidence-bearing candidates for analysis")

    store = Store(db)
    for article in pool:
        store.save_article(article)

    if no_llm:
        store.close()
        return

    analysed: list[AnalysedArticle] = []
    for article in pool:
        if len(analysed) >= desired:
            break
        try:
            analysis = analyse_article(article, profile, mode=mode)
        except Exception as exc:
            typer.echo(f"LLM skip after retry: {article.title[:72]} ({exc})")
            continue

        if not substantive(analysis):
            typer.echo(f"Analysis reject as too weak/generic: {article.title[:72]}")
            continue

        analysis.hashtags = normalise_hashtags(analysis, article.matched_terms)
        pscore = priority_score(analysis, mode)
        final = final_rank_score(
            heuristic=article.heuristic_score,
            relevance=analysis.relevance_score,
            priority=pscore,
        )
        item = AnalysedArticle(
            article=article,
            analysis=analysis,
            final_score=final,
            priority_score=pscore,
            mode=mode,
        )
        analysed.append(item)
        store.save_analysis(item)

    analysed.sort(key=lambda x: x.final_score, reverse=True)

    if mode == "weekly" and len(analysed) < 4:
        typer.echo(
            f"Quality note: only {len(analysed)} stories passed this week. "
            "Publishing a shorter briefing rather than padding it with weak or stale items."
        )

    synthesis = None
    if len(analysed) >= 2 and not no_synthesis:
        try:
            synthesis = analyse_digest([
                {
                    "title": item.article.title,
                    "source": item.article.source_name,
                    "priority_score": item.priority_score,
                    "bottom_line": item.analysis.bottom_line,
                    "why_it_matters": item.analysis.government_smart_data_perspective,
                    "hashtags": item.analysis.hashtags,
                }
                for item in analysed
            ], mode=mode)
        except Exception as exc:
            typer.echo(f"Synthesis warning: {exc}")

    store.close()
    md, js, ht = write_outputs(analysed, out_dir, synthesis, mode=mode)
    typer.echo(f"Wrote {md}, {js} and {ht} with {len(analysed)} ranked items")


@app.command("show-profile")
def show_profile() -> None:
    profile, _ = load_default_config()
    typer.echo(profile.get("mission", ""))


if __name__ == "__main__":
    app()
