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
from .score import heuristic_score
from .storage import Store
from .utils import utcnow

app = typer.Typer(add_completion=False, help="Smart Data & AI news intelligence radar")


@app.command()
def run(
    mode: str = typer.Option("weekly", help="Weekly or monthly intelligence mode"),
    days: int | None = typer.Option(None, help="Override recency window; defaults to 7 weekly / 30 monthly"),
    candidate_limit: int = typer.Option(int(os.getenv("RADAR_CANDIDATE_LIMIT", "50"))),
    analyse_limit: int | None = typer.Option(None, help="Override LLM analysis limit; defaults to 10 weekly / 16 monthly"),
    min_score: float = typer.Option(float(os.getenv("RADAR_MIN_SCORE", "35")), help="Minimum heuristic score before LLM analysis"),
    backend: str = typer.Option(os.getenv("SEARCH_BACKEND", "google_news")),
    db: Path = typer.Option(Path("radar.db")),
    out_dir: Path = typer.Option(Path("output")),
    no_llm: bool = typer.Option(False, help="Discover/rank only; do not call the LLM"),
    no_synthesis: bool = typer.Option(False, help="Skip cross-article synthesis"),
) -> None:
    if mode not in {"weekly", "monthly"}:
        raise typer.BadParameter("--mode must be weekly or monthly")
    profile, sources = load_default_config()
    days = days if days is not None else (7 if mode == "weekly" else 30)
    if analyse_limit is None:
        env_limit = os.getenv("RADAR_ANALYSE_LIMIT")
        analyse_limit = int(env_limit) if env_limit else (8 if mode == "weekly" else 12)

    typer.echo(f"Mode: {mode} · window: {days} days · discovery backend: {backend}")

    candidates = []
    try:
        candidates.extend(discover_search(profile, backend=backend, days=days))
    except Exception as exc:
        typer.echo(f"Search discovery warning: {exc}")
    candidates.extend(discover_index_sources(sources))
    candidates = dedupe_candidates(candidates)
    typer.echo(f"Discovered {len(candidates)} unique candidates")

    robots = RobotsCache()
    articles = []
    cutoff = utcnow() - timedelta(days=days)
    for candidate in candidates:
        try:
            article = extract_candidate(candidate, robots)
        except Exception:
            continue
        if article.published_at:
            pub = article.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if pub < cutoff and "index:" not in article.discovery_method:
                continue
        score, matched = heuristic_score(article, profile, sources)
        article.heuristic_score = score
        article.matched_terms = matched
        if score >= min_score:
            articles.append(article)

    articles = dedupe_articles(articles)[:candidate_limit]
    typer.echo(f"{len(articles)} candidates passed heuristic relevance")

    store = Store(db)
    for article in articles:
        store.save_article(article)

    if no_llm:
        typer.echo("LLM analysis skipped (--no-llm). Ranked articles saved to SQLite.")
        store.close()
        return

    analysed: list[AnalysedArticle] = []
    for article in articles[:analyse_limit]:
        try:
            analysis = analyse_article(article, profile, mode=mode)
            analysis.hashtags = normalise_hashtags(analysis, article.matched_terms)
        except Exception as exc:
            typer.echo(f"LLM skip: {article.title[:70]} ({exc})")
            continue
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

    synthesis = None
    if analysed and not no_synthesis:
        try:
            synthesis = analyse_digest([
                {
                    "title": i.article.title,
                    "source": i.article.source_name,
                    "priority_score": i.priority_score,
                    "bottom_line": i.analysis.bottom_line,
                    "why_it_matters": i.analysis.government_smart_data_perspective,
                    "hashtags": i.analysis.hashtags,
                }
                for i in analysed
            ], mode=mode)
        except Exception as exc:
            typer.echo(f"Synthesis warning: {exc}")

    store.close()
    md, js, ht = write_outputs(analysed, out_dir, synthesis, mode=mode)
    typer.echo(f"Wrote {md}, {js} and {ht}")


@app.command("show-profile")
def show_profile() -> None:
    profile, _ = load_default_config()
    typer.echo(profile.get("mission", ""))
    typer.echo("\nPrimary search phrases:")
    for phrase in profile.get("primary", {}).get("phrases", []):
        typer.echo(f"- {phrase}")


if __name__ == "__main__":
    app()

