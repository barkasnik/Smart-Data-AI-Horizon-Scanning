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
from .quality import content_quality, is_hub, is_strategic_watch, select_diverse
from .score import heuristic_score
from .storage import Store
from .utils import utcnow

app = typer.Typer(add_completion=False, help="Smart Data & AI news intelligence radar")

def substantive(analysis) -> bool:
    swot_n = sum(len(getattr(analysis.swot, k)) for k in ("strengths","weaknesses","opportunities","threats"))
    return (
        len((analysis.bottom_line or "").strip()) >= 35
        and len((analysis.government_smart_data_perspective or "").strip()) >= 55
        and swot_n >= 1
        and len(analysis.policy_or_market_implications) >= 1
    )

@app.command()
def run(
    mode: str = typer.Option("weekly"),
    days: int | None = typer.Option(None),
    candidate_limit: int = typer.Option(int(os.getenv("RADAR_CANDIDATE_LIMIT", "140"))),
    analyse_limit: int | None = typer.Option(None),
    min_score: float = typer.Option(float(os.getenv("RADAR_MIN_SCORE", "28"))),
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
    target = int(os.getenv("RADAR_MIN_WEEKLY_ITEMS", "7")) if mode == "weekly" else 10
    desired = int(os.getenv("RADAR_ANALYSE_LIMIT", "10")) if analyse_limit is None else analyse_limit
    desired = max(target, desired)
    attempts = max(desired, int(os.getenv("RADAR_ANALYSIS_ATTEMPTS", "24")))
    per_query = int(os.getenv("RADAR_RESULTS_PER_QUERY", "12"))

    typer.echo(f"Mode: {mode} · window: {days} days · target: {target}")

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
    strategic: list = []

    for c in candidates:
        try:
            a = extract_candidate(c, robots)
        except Exception:
            continue
        if is_hub(a):
            continue
        q = content_quality(a)
        if q < 32:
            continue
        score, matched = heuristic_score(a, profile, sources)
        a.heuristic_score = score
        a.matched_terms = matched
        if score < min_score:
            continue

        pub = a.published_at
        if pub and pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        if pub and pub >= cutoff:
            recent.append(a)
        elif is_strategic_watch(a) and score >= 45 and q >= 45:
            strategic.append(a)

    recent = dedupe_articles(recent)[:candidate_limit]
    strategic = dedupe_articles(strategic)[:20]

    pool = select_diverse(recent, limit=max(attempts, target), max_govuk=2, max_per_domain=3)
    if mode == "weekly" and strategic:
        already = {a.canonical_url for a in pool}
        for a in strategic:
            if a.canonical_url not in already:
                a.matched_terms = list(dict.fromkeys(a.matched_terms + ["strategic_watchlist"]))
                pool.append(a)
                already.add(a.canonical_url)
            if sum("strategic_watchlist" in x.matched_terms for x in pool) >= 2:
                break

    pool = select_diverse(pool, limit=min(len(pool), attempts), max_govuk=2, max_per_domain=3)
    typer.echo(f"Selected {len(pool)} candidates for analysis: {len(recent)} recent, {len(strategic)} strategic-watch candidates")

    store = Store(db)
    for a in pool:
        store.save_article(a)

    if no_llm:
        store.close()
        return

    analysed: list[AnalysedArticle] = []
    for a in pool:
        if len(analysed) >= desired:
            break
        try:
            analysis = analyse_article(a, profile, mode=mode)
        except Exception as exc:
            typer.echo(f"LLM skip: {a.title[:72]} ({exc})")
            continue
        if not substantive(analysis):
            typer.echo(f"Analysis reject: {a.title[:72]}")
            continue

        if "strategic_watchlist" in a.matched_terms:
            if not analysis.bottom_line.startswith("Strategic watchlist — "):
                analysis.bottom_line = "Strategic watchlist — " + analysis.bottom_line
            analysis.hashtags = list(dict.fromkeys(["#Strategic", "#Monitor"] + analysis.hashtags))

        analysis.hashtags = normalise_hashtags(analysis, a.matched_terms)
        ps = priority_score(analysis, mode)
        fs = final_rank_score(heuristic=a.heuristic_score, relevance=analysis.relevance_score, priority=ps)
        item = AnalysedArticle(article=a, analysis=analysis, final_score=fs, priority_score=ps, mode=mode)
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
    typer.echo(f"Wrote {md}, {js} and {ht} with {len(analysed)} ranked items")

@app.command("show-profile")
def show_profile() -> None:
    profile, _ = load_default_config()
    typer.echo(profile.get("mission", ""))

if __name__ == "__main__":
    app()
