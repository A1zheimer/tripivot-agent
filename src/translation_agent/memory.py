from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any

from .models import DocumentPlan, GlossaryEntry, Language
from .preprocessing import term_occurs


class WorkingMemory:
    def __init__(self, window_size: int = 4) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._items: deque[tuple[str, str]] = deque(maxlen=window_size)

    def add(self, source: str, translation: str) -> None:
        self._items.append((source, translation))

    def context(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()


class EpisodicMemory:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path)
        self._lock = RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translation TEXT NOT NULL,
                glossary_ids TEXT NOT NULL,
                confidence REAL,
                revisions INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (document_id, chunk_id)
            )
            """
        )
        self._connection.commit()

    def remember(
        self,
        document_id: str,
        chunk_id: str,
        source_text: str,
        translation: str,
        *,
        glossary_ids: list[str] | None = None,
        confidence: float | None = None,
        revisions: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO episodes (
                    document_id, chunk_id, source_text, translation,
                    glossary_ids, confidence, revisions, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, chunk_id) DO UPDATE SET
                    translation = excluded.translation,
                    glossary_ids = excluded.glossary_ids,
                    confidence = excluded.confidence,
                    revisions = excluded.revisions,
                    metadata = excluded.metadata
                """,
                (
                    document_id,
                    chunk_id,
                    source_text,
                    translation,
                    json.dumps(glossary_ids or []),
                    confidence,
                    revisions,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            self._connection.commit()

    def document_episodes(self, document_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT chunk_id, source_text, translation, glossary_ids,
                       confidence, revisions, metadata
                FROM episodes
                WHERE document_id = ?
                ORDER BY rowid
                """,
                (document_id,),
            )
            return [
                {
                    "chunk_id": row[0],
                    "source_text": row[1],
                    "translation": row[2],
                    "glossary_ids": json.loads(row[3]),
                    "confidence": row[4],
                    "revisions": row[5],
                    "metadata": json.loads(row[6]),
                }
                for row in cursor
            ]

    def close(self) -> None:
        self._connection.close()


class GlossaryMemory:
    def __init__(self, entries: list[GlossaryEntry] | None = None) -> None:
        self.entries = entries or []

    @classmethod
    def from_jsonl(cls, path: Path) -> GlossaryMemory:
        entries: list[GlossaryEntry] = []
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                payload = json.loads(line)
                entries.append(GlossaryEntry(**payload))
        return cls(entries)

    def retrieve(
        self,
        source_text: str,
        source_language: Language,
        target_language: Language,
        *,
        domain: str = "general",
        top_k: int = 20,
    ) -> list[GlossaryEntry]:
        matches: list[tuple[int, float, int, GlossaryEntry, str]] = []
        for entry in self.entries:
            source_term = entry.term(source_language)
            target_term = entry.term(target_language)
            if not source_term or not target_term:
                continue
            if not term_occurs(source_text, source_term, source_language):
                continue
            domain_score = 1 if entry.domain in {"general", domain} else 0
            matches.append((domain_score, entry.confidence, len(source_term), entry, source_term))
        # Prefer domain matches, then the LONGEST term, then confidence: when a
        # short term (「系统」) overlaps a longer match (「操作系统」), only the
        # longer one is kept so fragments cannot displace real terminology —
        # even when the short entry carries higher confidence.
        matches.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
        kept: list[GlossaryEntry] = []
        kept_terms: list[str] = []
        for _score, _confidence, _length, entry, source_term in matches:
            source_key = (
                source_term.casefold() if source_language is Language.ENGLISH else source_term
            )
            duplicate = source_key in kept_terms
            proper_overlap = any(
                (source_key in other or other in source_key) and source_key != other
                for other in kept_terms
            )
            if duplicate or proper_overlap:
                continue
            kept.append(entry)
            kept_terms.append(source_key)
            if len(kept) >= top_k:
                break
        return kept

    def conflicting_terms(
        self,
        source_language: Language,
        target_language: Language,
    ) -> dict[str, list[str]]:
        """Source terms that map to several different targets in the whole
        glossary — injected into prompts only as the top-ranked one, but
        surfaced as a warning so curators can resolve the conflict."""
        groups: dict[str, set[str]] = defaultdict(set)
        for entry in self.entries:
            source_term = entry.term(source_language)
            target_term = entry.term(target_language)
            if source_term and target_term:
                source_key = (
                    source_term.casefold()
                    if source_language is Language.ENGLISH
                    else source_term
                )
                groups[source_key].add(target_term)
        return {
            source_term: sorted(targets)
            for source_term, targets in groups.items()
            if len(targets) > 1
        }


class ProgressMemory:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, plan: DocumentPlan) -> Path:
        path = self.directory / f"{plan.document_id}.progress.json"
        payload = json.dumps(asdict(plan), ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
        return path

    def load(self, document_id: str) -> DocumentPlan | None:
        """Read back a progress snapshot, or ``None`` when absent/corrupt."""
        path = self.directory / f"{document_id}.progress.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return DocumentPlan.from_dict(payload)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None


class MemorySystem:
    def __init__(
        self,
        state_directory: Path,
        glossary: GlossaryMemory | None = None,
        *,
        working_window: int = 4,
    ) -> None:
        self.working = WorkingMemory(working_window)
        self.episodic = EpisodicMemory(state_directory / "episodes.sqlite3")
        self.semantic = glossary or GlossaryMemory()
        self.progress = ProgressMemory(state_directory / "progress")

    def close(self) -> None:
        self.episodic.close()
