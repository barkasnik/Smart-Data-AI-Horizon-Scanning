from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import AnalysedArticle, Article
from .utils import utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  canonical_url TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_name TEXT,
  author TEXT,
  published_at TEXT,
  discovered_at TEXT NOT NULL,
  discovery_method TEXT,
  heuristic_score REAL,
  article_json TEXT NOT NULL,
  analysis_json TEXT,
  final_score REAL
);

CREATE TABLE IF NOT EXISTS analyses (
  canonical_url TEXT NOT NULL,
  mode TEXT NOT NULL,
  analysed_at TEXT NOT NULL,
  priority_score REAL NOT NULL,
  final_score REAL NOT NULL,
  analysis_json TEXT NOT NULL,
  PRIMARY KEY (canonical_url, mode, analysed_at)
);
"""


class Store:
    def __init__(self, path: str | Path = "radar.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def seen(self, canonical_url: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE canonical_url = ?", (canonical_url,)
        ).fetchone()
        return row is not None

    def save_article(self, article: Article) -> None:
        self.conn.execute(
            """
            INSERT INTO articles (
              canonical_url,title,source_name,author,published_at,discovered_at,
              discovery_method,heuristic_score,article_json
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(canonical_url) DO UPDATE SET
              title=excluded.title,
              source_name=excluded.source_name,
              author=excluded.author,
              published_at=COALESCE(excluded.published_at, articles.published_at),
              heuristic_score=excluded.heuristic_score,
              article_json=excluded.article_json
            """,
            (
                article.canonical_url,
                article.title,
                article.source_name,
                article.author,
                article.published_at.isoformat() if article.published_at else None,
                article.discovered_at.isoformat(),
                article.discovery_method,
                article.heuristic_score,
                article.model_dump_json(),
            ),
        )
        self.conn.commit()

    def save_analysis(self, item: AnalysedArticle) -> None:
        self.save_article(item.article)
        analysis_json = item.analysis.model_dump_json()
        self.conn.execute(
            "UPDATE articles SET analysis_json=?, final_score=? WHERE canonical_url=?",
            (analysis_json, item.final_score, item.article.canonical_url),
        )
        self.conn.execute(
            """
            INSERT INTO analyses (
              canonical_url, mode, analysed_at, priority_score, final_score, analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.article.canonical_url,
                item.mode,
                utcnow().isoformat(),
                item.priority_score,
                item.final_score,
                analysis_json,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

