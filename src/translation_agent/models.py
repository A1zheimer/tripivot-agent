from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Language(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh"
    BURMESE = "my"

    @property
    def label(self) -> str:
        return {"en": "English", "zh": "Simplified Chinese", "my": "Burmese"}[self.value]


@dataclass(frozen=True, slots=True)
class TranslationDirection:
    source: Language
    target: Language

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("source and target languages must differ")


@dataclass(frozen=True, slots=True)
class StyleGuide:
    register: str = "formal"
    domain: str = "general"
    audience: str = "general"
    instructions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    concept_id: str
    terms: dict[str, str]
    domain: str = "general"
    definition: str = ""
    source: str = "curated"
    confidence: float = 1.0

    def term(self, language: Language) -> str | None:
        value = self.terms.get(language.value)
        return value.strip() if value and value.strip() else None


@dataclass(slots=True)
class Chunk:
    id: str
    section_id: str
    order: int
    source_text: str
    status: str = "pending"
    translation: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    id: str
    title: str
    order: int
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(slots=True)
class DocumentPlan:
    document_id: str
    direction: TranslationDirection
    style: StyleGuide
    sections: list[Section]
    glossary: list[GlossaryEntry] = field(default_factory=list)

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for section in self.sections for chunk in section.chunks]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentPlan:
        """Rebuild a plan from a progress snapshot, tolerating older schemas."""

        sections: list[Section] = []
        for section_payload in payload.get("sections", []):
            chunks: list[Chunk] = []
            for chunk_payload in section_payload.get("chunks", []):
                chunks.append(
                    Chunk(
                        id=chunk_payload["id"],
                        section_id=chunk_payload["section_id"],
                        order=int(chunk_payload["order"]),
                        source_text=chunk_payload["source_text"],
                        status=chunk_payload.get("status", "pending"),
                        translation=chunk_payload.get("translation", ""),
                        notes=list(chunk_payload.get("notes", [])),
                    )
                )
            sections.append(
                Section(
                    id=section_payload["id"],
                    title=section_payload.get("title", ""),
                    order=int(section_payload.get("order", 0)),
                    chunks=chunks,
                )
            )
        style_payload = payload.get("style", {})
        direction_payload = payload.get("direction", {})
        return cls(
            document_id=payload["document_id"],
            direction=TranslationDirection(
                Language(direction_payload["source"]),
                Language(direction_payload["target"]),
            ),
            style=StyleGuide(
                register=style_payload.get("register", "formal"),
                domain=style_payload.get("domain", "general"),
                audience=style_payload.get("audience", "general"),
                instructions=tuple(style_payload.get("instructions", ())),
            ),
            sections=sections,
            glossary=[
                GlossaryEntry(
                    concept_id=item["concept_id"],
                    terms=dict(item.get("terms", {})),
                    domain=item.get("domain", "general"),
                    definition=item.get("definition", ""),
                    source=item.get("source", "curated"),
                    confidence=float(item.get("confidence", 1.0)),
                )
                for item in payload.get("glossary", [])
            ],
        )


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    text: str
    direction: TranslationDirection
    style: StyleGuide
    glossary: tuple[GlossaryEntry, ...] = ()
    previous_context: tuple[tuple[str, str], ...] = ()
    previous_translation: str | None = None
    revision_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendResult:
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReflectionIssue:
    chunk_id: str
    kind: str
    message: str
    severity: str = "warning"
    expected: str | None = None


@dataclass(slots=True)
class TranslationReport:
    document_id: str
    source_language: str
    target_language: str
    total_chunks: int
    completed_chunks: int = 0
    failed_chunks: int = 0
    terminology_hits: int = 0
    revisions: int = 0
    issues: list[ReflectionIssue] = field(default_factory=list)
    backend_metadata: list[dict[str, Any]] = field(default_factory=list)
    ingestion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranslationArtifact:
    text: str
    report: TranslationReport
    plan: DocumentPlan
