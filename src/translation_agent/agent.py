from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

from .backends import TranslationBackend
from .memory import MemorySystem
from .models import (
    Chunk,
    DocumentPlan,
    GlossaryEntry,
    Language,
    ReflectionIssue,
    Section,
    StyleGuide,
    TranslationArtifact,
    TranslationDirection,
    TranslationReport,
    TranslationRequest,
)
from .preprocessing import chunk_sentences, normalize_text, split_sentences, term_occurs

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^(\s*)(?:([-*+])|(\d{1,9}[.)]))\s+(.+)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _fence_marker(line: str) -> str | None:
    """Return the fence marker (``` or ~~~) if the line opens/closes one."""
    match = _FENCE_RE.match(line)
    return match.group(1) if match else None


def _closes_fence(open_marker: str, line: str) -> bool:
    """CommonMark: a fence closes only on the SAME character and a marker at
    least as long — so a ``` line inside a ```` block stays code."""
    marker = _fence_marker(line)
    return (
        marker is not None
        and marker[0] == open_marker[0]
        and len(marker) >= len(open_marker)
    )


def _frontmatter_span(lines: list[str]) -> int:
    """Length of a CLOSED frontmatter block (lines[0] is '---'), or 0 when the
    block never closes — in which case the '---' is body content, not
    frontmatter, and must never swallow the document."""
    for index in range(1, min(len(lines), 100)):
        candidate = lines[index]
        if not candidate[:1].isspace() and candidate.strip() in {"---", "..."}:
            return index + 1
    return 0


def _table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and bool(_TABLE_ROW_RE.match(lines[index]))
        and bool(_TABLE_SEPARATOR_RE.match(lines[index + 1]))
    )


def _table_line(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    cells: list[str] = []
    escaped = False
    buffer: list[str] = []
    for character in stripped:
        if escaped and character == "|":
            buffer.append("|")
            escaped = False
            continue
        if escaped:
            buffer.append("\\")
            escaped = False
        if character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)
    if escaped:
        buffer.append("\\")
    cells.append("".join(buffer).strip())
    return cells


def _render_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"


def _split_list_block(block: str) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    current: tuple[str, str, list[str]] | None = None
    for line in block.splitlines():
        match = _LIST_RE.match(line)
        if match:
            if current is not None:
                items.append((current[0], current[1], "\n".join(current[2])))
            indent, bullet, ordered, content = match.groups()
            current = (indent, bullet or ordered, [content])
        elif current is not None and line.strip():
            current[2].append(line.strip())
        elif not line.strip():
            if current is not None:
                items.append((current[0], current[1], "\n".join(current[2])))
                current = None
    if current is not None:
        items.append((current[0], current[1], "\n".join(current[2])))
    return items


_LEADING_WS_RE = re.compile(r"[ \t]*")
# GFM indented code block: 4+ spaces or a tab, when it starts a block (not a
# list continuation). Pandoc emits DOCX code this way, so it must be protected
# exactly like fenced code.
_INDENTED_CODE_RE = re.compile(r"^(?: {4,}|\t)")


def _normalize_prose_line(line: str) -> str:
    """normalize_text for a single line, but keep leading indentation — it is
    structural (nested lists, indented code) and must survive planning."""
    leading = _LEADING_WS_RE.match(line).group(0)
    core = normalize_text(line)
    return leading + core if core else ""


def _normalize_document(text: str) -> str:
    """Normalize prose while keeping fenced and indented code blocks intact
    (``#`` lines included; only trailing whitespace is stripped), so code
    survives planning. A ``` fence is closed only by another ``` line — a ~~~
    line inside it stays code, and vice versa. Leading indentation of prose
    lines is preserved for the block classifier (nested lists, indented code).
    A CLOSED frontmatter block is kept byte-verbatim (YAML must not be
    unescaped or re-indented); an unclosed one is body content."""
    lines = text.splitlines()
    if lines and lines[0].rstrip() == "---":
        span = _frontmatter_span(lines)
        if span:
            frontmatter = "\n".join(lines[:span]).rstrip()
            rest = _normalize_document("\n".join(lines[span:]))
            return "\n".join(part for part in (frontmatter, rest) if part)
    parts: list[str] = []
    prose: list[str] = []
    fence: list[str] = []
    open_marker: str | None = None
    for line in text.splitlines():
        marker = _fence_marker(line)
        if open_marker is not None:
            fence.append(line.rstrip())
            if _closes_fence(open_marker, line):
                open_marker = None
                parts.append("\n".join(fence))
                fence = []
            continue
        if marker is not None:
            if prose:
                parts.append("\n".join(_normalize_prose_line(item) for item in prose).strip("\n"))
                prose = []
            fence.append(line.rstrip())
            open_marker = marker
            continue
        prose.append(line)
    if prose:
        parts.append("\n".join(_normalize_prose_line(item) for item in prose).strip("\n"))
    if fence:  # unclosed fence: keep verbatim
        parts.append("\n".join(fence))
    return "\n".join(part for part in parts if part)


class HierarchicalPlanner:
    def __init__(self, chunk_sentence_count: int = 6) -> None:
        self.chunk_sentence_count = chunk_sentence_count

    def plan(
        self,
        text: str,
        direction: TranslationDirection,
        style: StyleGuide,
        memory: MemorySystem,
    ) -> DocumentPlan:
        normalized = _normalize_document(text)
        if not normalized:
            raise ValueError("document is empty")
        document_id = hashlib.sha256(
            f"{direction.source.value}>{direction.target.value}\0{normalized}".encode()
        ).hexdigest()[:16]
        raw_sections = self._sections(normalized)
        sections: list[Section] = []
        chunk_order = 0
        for section_order, (heading, level, body) in enumerate(raw_sections):
            section_id = f"s{section_order:04d}"
            chunks: list[Chunk] = []
            if heading:
                chunks.append(
                    Chunk(
                        id=f"{section_id}-c{chunk_order:05d}",
                        section_id=section_id,
                        order=chunk_order,
                        source_text=heading,
                        notes=[f"heading:{level}"],
                    )
                )
                chunk_order += 1
            for paragraph_index, (paragraph, block_kind) in enumerate(body):
                if block_kind == "code":
                    chunks.append(
                        Chunk(
                            id=f"{section_id}-c{chunk_order:05d}",
                            section_id=section_id,
                            order=chunk_order,
                            source_text=paragraph,
                            notes=[f"paragraph:{paragraph_index}", "code"],
                        )
                    )
                    chunk_order += 1
                    continue
                if block_kind == "frontmatter":
                    chunks.append(
                        Chunk(
                            id=f"{section_id}-c{chunk_order:05d}",
                            section_id=section_id,
                            order=chunk_order,
                            source_text=paragraph,
                            notes=[f"paragraph:{paragraph_index}", "frontmatter"],
                        )
                    )
                    chunk_order += 1
                    continue
                if block_kind == "list":
                    items = _split_list_block(paragraph)
                    for item_index, (indent, marker, item_text) in enumerate(items):
                        chunks.append(
                            Chunk(
                                id=f"{section_id}-c{chunk_order:05d}",
                                section_id=section_id,
                                order=chunk_order,
                                source_text=item_text,
                                notes=[
                                    f"paragraph:{paragraph_index}",
                                    f"list-part:{item_index}",
                                    f"list-indent:{len(indent)}",
                                    f"list-marker:{marker}",
                                ],
                            )
                        )
                        chunk_order += 1
                    continue
                if block_kind == "table":
                    table_lines = paragraph.splitlines()
                    for row_index, row in enumerate(table_lines):
                        if row_index == 1:
                            chunks.append(
                                Chunk(
                                    id=f"{section_id}-c{chunk_order:05d}",
                                    section_id=section_id,
                                    order=chunk_order,
                                    source_text=row,
                                    notes=[
                                        f"paragraph:{paragraph_index}",
                                        "table-separator",
                                    ],
                                )
                            )
                            chunk_order += 1
                            continue
                        cells = _split_table_row(row)
                        chunks.append(
                            Chunk(
                                id=f"{section_id}-c{chunk_order:05d}",
                                section_id=section_id,
                                order=chunk_order,
                                source_text="\n".join(cells),
                                notes=[
                                    f"paragraph:{paragraph_index}",
                                    f"table-row:{row_index}",
                                    f"table-columns:{len(cells)}",
                                ],
                            )
                        )
                        chunk_order += 1
                    continue
                sentences = split_sentences(paragraph, direction.source)
                for part_index, chunk_text in enumerate(
                    chunk_sentences(sentences, self.chunk_sentence_count)
                ):
                    chunks.append(
                        Chunk(
                            id=f"{section_id}-c{chunk_order:05d}",
                            section_id=section_id,
                            order=chunk_order,
                            source_text=chunk_text,
                            notes=[
                                f"paragraph:{paragraph_index}",
                                f"part:{part_index}",
                            ],
                        )
                    )
                    chunk_order += 1
            sections.append(
                Section(
                    id=section_id,
                    title=heading,
                    order=section_order,
                    chunks=chunks,
                )
            )

        # plan.glossary stays empty: enforcement and injection both use the
        # per-chunk retrieved set, so there is no document-level glossary to
        # precompute (the field is kept for schema compatibility).
        return DocumentPlan(
            document_id=document_id,
            direction=direction,
            style=style,
            sections=sections,
        )

    @staticmethod
    def _sections(text: str) -> list[tuple[str, int, list[tuple[str, str]]]]:
        sections: list[tuple[str, int, list[tuple[str, str]]]] = []
        heading = ""
        level = 0
        body_lines: list[str] = []

        def flush() -> None:
            nonlocal body_lines
            blocks: list[tuple[str, str]] = []
            index = 0
            while index < len(body_lines):
                line = body_lines[index]
                if not line.strip():
                    index += 1
                    continue
                marker = _fence_marker(line)
                if marker is not None:
                    end = index + 1
                    while end < len(body_lines):
                        closing = _fence_marker(body_lines[end])
                        if (
                            closing is not None
                            and closing[0] == marker[0]
                            and len(closing) >= len(marker)
                        ):
                            break
                        end += 1
                    closing = min(end + 1, len(body_lines))
                    blocks.append(("\n".join(body_lines[index:closing]), "code"))
                    index = end + 1
                    continue
                if _table_start(body_lines, index):
                    end = index + 1
                    while end < len(body_lines) and _table_line(body_lines[end]):
                        end += 1
                    blocks.append(("\n".join(body_lines[index:end]), "table"))
                    index = end
                    continue
                if _LIST_RE.match(line):
                    end = index + 1
                    while end < len(body_lines) and body_lines[end].strip():
                        if _table_start(body_lines, end):
                            break
                        end += 1
                    blocks.append(("\n".join(body_lines[index:end]), "list"))
                    index = end
                    continue
                if _INDENTED_CODE_RE.match(line):
                    # Indented code block (pandoc's DOCX output form): consume
                    # the run of indented lines, allowing blank lines that are
                    # followed by more indented code.
                    end = index
                    last_indented = index
                    while end < len(body_lines):
                        candidate = body_lines[end]
                        if candidate.strip():
                            if not _INDENTED_CODE_RE.match(candidate):
                                break
                            last_indented = end
                        end += 1
                    blocks.append(("\n".join(body_lines[index : last_indented + 1]), "code"))
                    index = last_indented + 1
                    continue
                paragraph: list[str] = []
                while index < len(body_lines) and body_lines[index].strip():
                    if _table_start(body_lines, index):
                        break
                    current_marker = _fence_marker(body_lines[index])
                    if current_marker is not None:
                        break
                    paragraph.append(body_lines[index])
                    index += 1
                if paragraph:
                    blocks.append(("\n".join(paragraph), "prose"))
            if heading or blocks:
                sections.append((heading, level, blocks))
            body_lines = []

        lines = text.splitlines()
        start = 0
        frontmatter: list[str] = []
        start = 0
        if lines and lines[0].strip() == "---":
            span = _frontmatter_span(lines)
            if span:
                frontmatter = lines[:span]
                start = span

        open_marker: str | None = None
        body_start = start
        if frontmatter:
            sections.append(("", 0, [("\n".join(frontmatter), "frontmatter")]))
        for line in lines[body_start:]:
            marker = _fence_marker(line)
            if open_marker is not None:
                body_lines.append(line)
                if _closes_fence(open_marker, line):
                    open_marker = None
                continue
            match = _HEADING_RE.match(line)
            if match:
                flush()
                heading = match.group(2)
                level = len(match.group(1))
            else:
                body_lines.append(line)
                if marker is not None:
                    open_marker = marker
        flush()
        return sections


class TerminologyReflector:
    """Checks exactly the glossary that was injected into the prompt, so the
    agent never enforces a requirement the model was never told about."""

    def inspect(
        self,
        chunk: Chunk,
        glossary: list[GlossaryEntry] | tuple[GlossaryEntry, ...],
        direction: TranslationDirection,
    ) -> list[ReflectionIssue]:
        issues: list[ReflectionIssue] = []
        for entry in glossary:
            source_term = entry.term(direction.source)
            target_term = entry.term(direction.target)
            if (
                not source_term
                or not target_term
                or not term_occurs(chunk.source_text, source_term, direction.source)
            ):
                continue
            if not term_occurs(chunk.translation, target_term, direction.target):
                issues.append(
                    ReflectionIssue(
                        chunk_id=chunk.id,
                        kind="terminology",
                        message=f"required translation missing for {source_term!r}",
                        severity="error",
                        expected=target_term,
                    )
                )
        return issues

    def audit_document(
        self,
        plan: DocumentPlan,
        active_by_chunk: dict[str, list[GlossaryEntry]],
    ) -> list[ReflectionIssue]:
        direction = plan.direction
        observed: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for chunk in plan.chunks:
            if chunk.status != "completed" or "code" in chunk.notes or not chunk.translation:
                continue
            for entry in active_by_chunk.get(chunk.id, []):
                source_term = entry.term(direction.source)
                target_term = entry.term(direction.target)
                if not source_term or not target_term:
                    continue
                if not term_occurs(chunk.source_text, source_term, direction.source):
                    continue
                variant = (
                    target_term
                    if term_occurs(chunk.translation, target_term, direction.target)
                    else "<missing>"
                )
                observed[source_term][variant].append(chunk.id)
        issues: list[ReflectionIssue] = []
        for source_term, variants in observed.items():
            missing = variants.get("<missing>", [])
            real = [variant for variant in variants if variant != "<missing>"]
            if missing:
                issues.append(
                    ReflectionIssue(
                        chunk_id="document",
                        kind="global_terminology",
                        message=(
                            f"missing required translation for {source_term!r} "
                            f"in {len(missing)} chunk(s): {missing[:5]}"
                        ),
                        severity="error",
                    )
                )
            if len(real) > 1:
                issues.append(
                    ReflectionIssue(
                        chunk_id="document",
                        kind="global_terminology",
                        message=(
                            f"inconsistent translations for {source_term!r}: {sorted(real)}"
                        ),
                        severity="error",
                    )
                )
        return issues


class LongHorizonTranslationAgent:
    def __init__(
        self,
        backend: TranslationBackend,
        state_directory: Path,
        *,
        memory: MemorySystem | None = None,
        planner: HierarchicalPlanner | None = None,
        reflector: TerminologyReflector | None = None,
        max_revision_rounds: int = 1,
    ) -> None:
        self.backend = backend
        self.memory = memory or MemorySystem(state_directory)
        self.planner = planner or HierarchicalPlanner()
        self.reflector = reflector or TerminologyReflector()
        self.max_revision_rounds = max_revision_rounds

    def translate_document(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        *,
        style: StyleGuide | None = None,
        resume: bool = True,
        continue_on_error: bool = True,
    ) -> TranslationArtifact:
        direction = TranslationDirection(source_language, target_language)
        style = style or StyleGuide()
        self.memory.working.clear()
        plan = self.planner.plan(text, direction, style, self.memory)
        if resume:
            self._restore_completed(plan)
        report = TranslationReport(
            document_id=plan.document_id,
            source_language=source_language.value,
            target_language=target_language.value,
            total_chunks=len(plan.chunks),
        )
        revision_enabled = getattr(self.backend, "revision_capable", True)
        active_by_chunk: dict[str, list[GlossaryEntry]] = {}
        self.memory.progress.save(plan)

        for chunk in plan.chunks:
            if any(
                note in chunk.notes for note in ("code", "frontmatter", "table-separator")
            ):
                # Structural blocks are preserved verbatim, never sent to a model.
                chunk.translation = chunk.source_text
                chunk.status = "completed"
                report.completed_chunks += 1
                self.memory.progress.save(plan)
                continue

            if chunk.status == "completed" and chunk.translation:
                # Restored from a previous run: reuse the translation and keep
                # the working-memory context flowing for the chunks after it.
                report.completed_chunks += 1
                self.memory.working.add(chunk.source_text, chunk.translation)
                active = self.memory.semantic.retrieve(
                    chunk.source_text,
                    source_language,
                    target_language,
                    domain=style.domain,
                )
                active_by_chunk[chunk.id] = active
                restored_issues = self.reflector.inspect(chunk, active, direction)
                confirmed = len(active) - sum(
                    1 for issue in restored_issues if issue.kind == "terminology"
                )
                report.terminology_hits += max(0, confirmed)
                continue

            chunk.status = "running"
            active_glossary = self.memory.semantic.retrieve(
                chunk.source_text,
                source_language,
                target_language,
                domain=style.domain,
            )
            active_by_chunk[chunk.id] = active_glossary
            request = TranslationRequest(
                text=chunk.source_text,
                direction=direction,
                style=style,
                glossary=tuple(active_glossary),
                previous_context=self.memory.working.context(),
            )
            issues: list[ReflectionIssue] = []
            revisions = 0
            try:
                result = self.backend.translate(request)
                if not result.text or not result.text.strip():
                    raise RuntimeError("backend returned an empty translation")
                chunk.translation = result.text
                report.backend_metadata.append(result.metadata)
                issues = self.reflector.inspect(chunk, active_glossary, direction)
                while issues and revision_enabled and revisions < self.max_revision_rounds:
                    revision_request = TranslationRequest(
                        text=chunk.source_text,
                        direction=direction,
                        style=style,
                        glossary=tuple(active_glossary),
                        previous_context=self.memory.working.context(),
                        previous_translation=chunk.translation,
                        revision_notes=tuple(
                            self._revision_note(issue) for issue in issues
                        ),
                    )
                    result = self.backend.translate(revision_request)
                    if not result.text or not result.text.strip():
                        raise RuntimeError("backend returned an empty revision")
                    chunk.translation = result.text
                    report.backend_metadata.append(result.metadata)
                    revisions += 1
                    report.revisions += 1
                    issues = self.reflector.inspect(chunk, active_glossary, direction)
            except Exception as error:
                chunk.status = "failed"
                # Drop any draft: a half-revised translation that failed term
                # checks must not leak into the rendered output.
                chunk.translation = ""
                chunk.notes.append(f"error:{type(error).__name__}:{error}")
                report.failed_chunks += 1
                report.issues.append(
                    ReflectionIssue(
                        chunk_id=chunk.id,
                        kind="chunk_error",
                        message=f"{type(error).__name__}: {error}",
                        severity="error",
                    )
                )
                self.memory.progress.save(plan)
                if not continue_on_error:
                    raise
                continue
            chunk.status = "completed"
            report.issues.extend(issues)
            terminology_issues = sum(1 for issue in issues if issue.kind == "terminology")
            report.terminology_hits += len(active_glossary) - terminology_issues
            report.completed_chunks += 1
            self.memory.working.add(chunk.source_text, chunk.translation)
            self.memory.episodic.remember(
                plan.document_id,
                chunk.id,
                chunk.source_text,
                chunk.translation,
                glossary_ids=[entry.concept_id for entry in active_glossary],
                confidence=result.confidence,
                revisions=revisions,
                metadata=result.metadata,
            )
            self.memory.progress.save(plan)

        report.issues.extend(self.reflector.audit_document(plan, active_by_chunk))
        report.issues.extend(self._glossary_conflicts(direction))
        return TranslationArtifact(text=self._render(plan), report=report, plan=plan)

    def _glossary_conflicts(self, direction: TranslationDirection) -> list[ReflectionIssue]:
        conflicts = self.memory.semantic.conflicting_terms(
            direction.source,
            direction.target,
        )
        return [
            ReflectionIssue(
                chunk_id="document",
                kind="glossary_conflict",
                message=(
                    f"glossary defines multiple translations for {source_term!r}: "
                    f"{targets}; only the top-ranked one is enforced"
                ),
                severity="warning",
            )
            for source_term, targets in conflicts.items()
        ]

    def _restore_completed(self, plan: DocumentPlan) -> None:
        saved = self.memory.progress.load(plan.document_id)
        if saved is None or saved.direction != plan.direction or saved.style != plan.style:
            return
        previous = {
            chunk.id: chunk
            for chunk in saved.chunks
            if chunk.status == "completed" and chunk.translation
        }
        if not previous:
            return
        for chunk in plan.chunks:
            saved_chunk = previous.get(chunk.id)
            if saved_chunk is None or saved_chunk.source_text != chunk.source_text:
                continue
            chunk.status = "completed"
            chunk.translation = saved_chunk.translation

    @staticmethod
    def _revision_note(issue: ReflectionIssue) -> str:
        if issue.expected:
            return f"{issue.message}; the required translation is {issue.expected!r}"
        return issue.message

    @staticmethod
    def _render(plan: DocumentPlan) -> str:
        rendered: list[str] = []
        previous_paragraph: str | None = None
        for chunk in plan.chunks:
            heading_note = next(
                (note for note in chunk.notes if note.startswith("heading:")),
                None,
            )
            if heading_note:
                level = int(heading_note.split(":", 1)[1])
                if rendered and rendered[-1] != "":
                    rendered.append("")
                title = chunk.translation or chunk.source_text
                rendered.extend([f"{'#' * level} {title}", ""])
                previous_paragraph = None
                continue
            paragraph_note = next(
                (note for note in chunk.notes if note.startswith("paragraph:")),
                "paragraph:0",
            )
            if "frontmatter" in chunk.notes:
                rendered.extend([chunk.translation or chunk.source_text, ""])
                previous_paragraph = None
                continue
            if "code" in chunk.notes:
                if rendered and rendered[-1] != "":
                    rendered.append("")
                rendered.extend([chunk.translation or chunk.source_text, ""])
                previous_paragraph = None
                continue
            if "table-separator" in chunk.notes:
                if (
                    previous_paragraph is not None
                    and paragraph_note != previous_paragraph
                ):
                    rendered.append("")
                rendered.append(chunk.translation or chunk.source_text)
                previous_paragraph = paragraph_note
                continue
            if any(note.startswith("table-row:") for note in chunk.notes):
                if (
                    previous_paragraph is not None
                    and paragraph_note != previous_paragraph
                ):
                    rendered.append("")
                columns = int(
                    next(
                        (
                            note.split(":", 1)[1]
                            for note in chunk.notes
                            if note.startswith("table-columns:")
                        ),
                        "0",
                    )
                )
                translation = chunk.translation or chunk.source_text
                cells = translation.splitlines()
                if columns and len(cells) == columns:
                    rendered.append(_render_table_row(cells))
                else:
                    rendered.append(translation)
                previous_paragraph = paragraph_note
                continue
            list_indent = next(
                (note.split(":", 1)[1] for note in chunk.notes if note.startswith("list-indent:")),
                "0",
            )
            list_marker = next(
                (note.split(":", 1)[1] for note in chunk.notes if note.startswith("list-marker:")),
                "-",
            )
            if any(note.startswith("list-marker:") for note in chunk.notes):
                if (
                    previous_paragraph is not None
                    and paragraph_note != previous_paragraph
                ):
                    rendered.append("")
                text = chunk.translation or chunk.source_text
                prefix = " " * int(list_indent) + list_marker + " "
                rendered.append(prefix + text.replace("\n", "\n" + " " * int(list_indent)))
                previous_paragraph = paragraph_note
                continue
            if previous_paragraph is not None and paragraph_note != previous_paragraph:
                rendered.append("")
            # Failed chunks keep the source text as a visible placeholder; the
            # report lists them so the reader knows what to re-run.
            rendered.append(chunk.translation or chunk.source_text)
            previous_paragraph = paragraph_note
        return "\n".join(rendered).strip() + "\n"

    def close(self) -> None:
        self.memory.close()
