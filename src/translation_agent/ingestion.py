from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .models import Language
from .preprocessing import looks_like_zawgyi, script_ratio


class IngestionError(RuntimeError):
    """Raised when a document cannot be converted into canonical Markdown."""


@dataclass(frozen=True, slots=True)
class IngestionWarning:
    message: str
    severity: str = "warning"
    blocking: bool = False


@dataclass(slots=True)
class IngestionResult:
    markdown: str
    source_format: str
    source_path: Path
    source_sha256: str
    converter: str
    source_metrics: dict[str, Any]
    markdown_metrics: dict[str, Any]
    warnings: list[IngestionWarning] = field(default_factory=list)


_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST_MARKER_RE = re.compile(r"^(\s*)(?:([-*+])|(\d{1,9}[.)]))\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_MAX_BYTES = 50 * 1024 * 1024


def _fence_marker(line: str) -> str | None:
    """Return the fence marker (``` or ~~~) if the line opens/closes one."""
    match = _FENCE_RE.match(line)
    return match.group(1) if match else None


def ingest_document(
    path: Path,
    *,
    source_language: Language | None = None,
    max_bytes: int = _MAX_BYTES,
) -> IngestionResult:
    """Convert a supported document into the Agent's canonical Markdown form.

    The translation Agent deliberately understands only canonical GFM. This
    boundary lets DOCX/PDF/HTML support evolve without pushing document-layout
    concerns into the long-horizon translation loop.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise IngestionError(f"input document does not exist: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise IngestionError(
            f"input document is {size} bytes; limit is {max_bytes} bytes"
        )

    source_format = _detect_format(path)
    digest = _sha256(path)
    warnings: list[IngestionWarning] = []
    source_metrics: dict[str, Any] = {"bytes": size}

    if source_format == "markdown":
        markdown = _decode_text(path)
        converter = "canonical-markdown"
    elif source_format == "text":
        text = _decode_text(path)
        markdown = _text_to_markdown(text)
        converter = "plain-text-wrapper"
    elif source_format in {"docx", "html"}:
        if source_format == "docx":
            source_metrics.update(_inspect_docx(path, warnings))
        markdown, converter = _run_pandoc(path, source_format)
    elif source_format == "pdf":
        markdown, pdf_metrics, pdf_warnings = _pdf_to_markdown(path)
        source_metrics.update(pdf_metrics)
        warnings.extend(pdf_warnings)
        converter = "pdfplumber-heuristic"
    else:  # pragma: no cover - _detect_format raises before this branch.
        raise IngestionError(f"unsupported source format: {source_format}")

    markdown = _canonicalize_markdown(markdown)
    markdown_metrics = markdown_statistics(markdown)
    _validate_markdown(markdown, markdown_metrics, warnings)
    if source_language is not None:
        _validate_language(markdown, source_language, warnings)

    return IngestionResult(
        markdown=markdown,
        source_format=source_format,
        source_path=path,
        source_sha256=digest,
        converter=converter,
        source_metrics=source_metrics,
        markdown_metrics=markdown_metrics,
        warnings=warnings,
    )


def ensure_ingestable(result: IngestionResult, *, strict: bool = False) -> None:
    failures = [
        warning.message
        for warning in result.warnings
        if warning.blocking or strict
    ]
    if failures:
        raise IngestionError("ingestion failed validation:\n- " + "\n- ".join(failures))


def ingestion_report(result: IngestionResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ingested_at": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(result.source_path),
            "format": result.source_format,
            "sha256": result.source_sha256,
            **result.source_metrics,
        },
        "converter": result.converter,
        "markdown": result.markdown_metrics,
        "warnings": [asdict(warning) for warning in result.warnings],
    }


def write_ingestion_artifacts(
    result: IngestionResult,
    markdown_path: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(result.markdown, encoding="utf-8")
    report_path.write_text(
        json.dumps(ingestion_report(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, report_path


def _strips_frontmatter(markdown: str) -> str:
    """Drop a closed YAML frontmatter block; the agent translates it verbatim
    and separately, so it must not pollute prose/table statistics."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return markdown
    for index in range(1, min(len(lines), 100)):
        if not lines[index][:1].isspace() and lines[index].strip() in {"---", "..."}:
            return "\n".join(lines[index + 1 :])
    return markdown


def markdown_statistics(markdown: str) -> dict[str, Any]:
    markdown = _strips_frontmatter(markdown)
    lines = markdown.splitlines()
    open_marker: str | None = None
    headings = 0
    paragraphs = 0
    list_items = 0
    table_rows = 0
    tables = 0
    code_blocks = 0
    images = 0
    links = 0
    paragraph_started = False
    table_started = False

    for line in lines:
        marker = _fence_marker(line)
        if marker is not None:
            if open_marker is None:
                open_marker = marker
                code_blocks += 1
                if paragraph_started:
                    paragraphs += 1
                    paragraph_started = False
                if table_started:
                    tables += 1
                    table_started = False
            elif marker[0] == open_marker[0] and len(marker) >= len(open_marker):
                open_marker = None
            continue
        if open_marker is not None:
            continue
        stripped = line.strip()
        if not stripped:
            if paragraph_started:
                paragraphs += 1
                paragraph_started = False
            if table_started:
                tables += 1
                table_started = False
            continue
        if stripped.startswith("#") and re.match(r"^#{1,6}\s+", stripped):
            headings += 1
            continue
        if _TABLE_ROW_RE.match(stripped):
            table_rows += 1
            table_started = True
            continue
        if _LIST_MARKER_RE.match(stripped):
            list_items += 1
            if paragraph_started:
                paragraphs += 1
                paragraph_started = False
            continue
        images += len(re.findall(r"!\[[^\]]*\]\([^)]+\)", stripped))
        links += len(re.findall(r"(?<!\!)\[[^\]]+\]\([^)]+\)", stripped))
        paragraph_started = True

    if paragraph_started:
        paragraphs += 1
    if table_started:
        tables += 1
    return {
        "characters": len(markdown),
        "lines": len(lines),
        "headings": headings,
        "paragraphs": paragraphs,
        "list_items": list_items,
        "tables": tables,
        "table_rows": table_rows,
        "code_blocks": code_blocks,
        "images": images,
        "links": links,
    }


def _format_from_suffix(suffix: str) -> str:
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def _detect_format(path: Path) -> str:
    prefix = path.open("rb").read(16)
    suffix = path.suffix.lower()
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        if not prefix.startswith(b"PK\x03\x04"):
            # Empty span / EOCD-only zip: not a readable document, and its
            # NUL bytes must never reach the canonical Markdown.
            raise IngestionError("ZIP-based input is empty or unsupported; expected .docx")
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as error:
            raise IngestionError("ZIP-based input is corrupt") from error
        if "[Content_Types].xml" in names and "word/document.xml" in names:
            return "docx"
        if suffix in {".md", ".markdown", ".txt", ".html", ".htm"}:
            raise IngestionError(
                f"{suffix or 'text'} file has a ZIP signature and is not Markdown/HTML/text"
            )
        raise IngestionError("unsupported ZIP-based document; expected .docx")

    # BOMs are text signatures by themselves (UTF-16 BOMs are never valid
    # UTF-8, so checking them first keeps UTF-16 support reachable).
    if prefix.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        return _format_from_suffix(suffix)
    # Decode the WHOLE file: a 16-byte prefix can cut a multi-byte character
    # in half and would misclassify CJK/Burmese-leading documents as binary.
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise IngestionError(
            "input is neither PDF/DOCX nor UTF-8 text; convert the file to "
            "UTF-8 (e.g. GBK/Shift-JIS → UTF-8) and retry"
        ) from error
    return _format_from_suffix(suffix)


def _decode_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        if data.startswith(b"\xef\xbb\xbf"):
            return data.decode("utf-8-sig")
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            # "utf-16" honours the BOM: it detects endianness AND strips it.
            return data.decode("utf-16")
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IngestionError(
            "text input must be UTF-8, UTF-8 BOM, or UTF-16 with BOM"
        ) from error


def _text_to_markdown(text: str) -> str:
    # Keep paragraphs as the semantic unit. The planner performs its own prose
    # normalization, so changing hard wrapping here would lose author intent.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canonicalize_markdown(markdown: str) -> str:
    return markdown.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _validate_markdown(
    markdown: str,
    metrics: dict[str, Any],
    warnings: list[IngestionWarning],
) -> None:
    if not markdown.strip():
        warnings.append(
            IngestionWarning("converted document is empty", severity="error", blocking=True)
        )
        return
    open_marker: str | None = None
    for line in markdown.splitlines():
        match = _FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group(1)
        if open_marker is None:
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker):
            open_marker = None
    if open_marker is not None:
        warnings.append(
            IngestionWarning(
                "unclosed fenced code block", severity="error", blocking=True
            )
        )
    if metrics["characters"] < 20:
        warnings.append(
            IngestionWarning(
                "converted document has fewer than 20 characters",
                severity="warning",
            )
        )
    if re.search(r"<(?:html|body|div|h[1-6]|p|br)\b", markdown, re.IGNORECASE):
        warnings.append(
            IngestionWarning(
                "document contains raw HTML tags; tags will reach the model "
                "as text unless the input is converted (use an .html input)"
            )
        )
    # Zawgyi is a data-corruption risk regardless of the declared language,
    # so this check runs even without --source.
    if looks_like_zawgyi(markdown):
        warnings.append(
            IngestionWarning(
                "probable Zawgyi text detected in source document",
                severity="error",
                blocking=True,
            )
        )


def _validate_language(
    markdown: str,
    language: Language,
    warnings: list[IngestionWarning],
) -> None:
    prose = "\n".join(
        line
        for line in markdown.splitlines()
        if not _FENCE_RE.match(line) and not _TABLE_ROW_RE.match(line)
    )
    ratio = script_ratio(prose, language)
    minimum = {
        Language.ENGLISH: 0.20,
        Language.CHINESE: 0.08,
        Language.BURMESE: 0.10,
    }[language]
    if ratio < minimum:
        warnings.append(
            IngestionWarning(
                f"source language mismatch: expected {language.value}, "
                f"script ratio={ratio:.3f}",
                severity="warning",
            )
        )


def _run_pandoc(path: Path, source_format: str) -> tuple[str, str]:
    executable = "pandoc"
    command = [
        executable,
        str(path),
        "--from",
        "docx" if source_format == "docx" else "html",
        "--to",
        "gfm",
        "--wrap=none",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as error:
        raise IngestionError(
            f"{source_format.upper()} ingestion requires pandoc; "
            "install pandoc or convert the document to Markdown first"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise IngestionError("pandoc conversion timed out after 180 seconds") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        raise IngestionError(f"pandoc failed ({completed.returncode}): {detail}")
    version = _pandoc_version()
    return completed.stdout, f"pandoc {version}" if version else "pandoc"


def _pandoc_version() -> str:
    try:
        completed = subprocess.run(
            ["pandoc", "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.splitlines()[0].split()[-1] if completed.stdout else ""


def _inspect_docx(path: Path, warnings: list[IngestionWarning]) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            document = archive.read("word/document.xml")
            comments = archive.read("word/comments.xml") if "word/comments.xml" in names else b""
            media_count = sum(name.startswith("word/media/") for name in names)
    except (OSError, zipfile.BadZipFile, UnicodeError) as error:
        raise IngestionError("DOCX document is corrupt") from error

    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as error:
        raise IngestionError("DOCX document.xml is invalid XML") from error
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = sum(1 for node in root.iter(f"{namespace}p"))
    tables = sum(1 for node in root.iter(f"{namespace}tbl"))
    insertions = sum(1 for node in root.iter(f"{namespace}ins"))
    deletions = sum(1 for node in root.iter(f"{namespace}del"))
    comment_count = comments.count(b"<w:comment ")

    if insertions or deletions:
        warnings.append(
            IngestionWarning(
                f"DOCX contains tracked changes ({insertions} insertions, {deletions} deletions)",
                severity="error",
                blocking=True,
            )
        )
    if comment_count:
        warnings.append(
            IngestionWarning(
                f"DOCX contains {comment_count} comment(s); comments are not translated",
            )
        )
    if "word/vbaProject.bin" in names:
        warnings.append(
            IngestionWarning(
                "DOCX contains a VBA project", severity="error", blocking=True
            )
        )
    return {
        "paragraphs": paragraphs,
        "tables": tables,
        "tracked_insertions": insertions,
        "tracked_deletions": deletions,
        "comments": comment_count,
        "media_files": media_count,
    }


def _pdf_to_markdown(
    path: Path,
) -> tuple[str, dict[str, Any], list[IngestionWarning]]:
    try:
        import pdfplumber
    except ImportError as error:
        raise IngestionError(
            "PDF ingestion requires pdfplumber; install the project with [documents]"
        ) from error

    pages: list[list[str]] = []
    with pdfplumber.open(path) as document:
        page_count = len(document.pages)
        for page in document.pages:
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:  # pdfplumber can raise per-page parser errors.
                text = ""
            pages.append(text.replace("\r\n", "\n").replace("\r", "\n").splitlines())

    nonempty_pages = sum(bool(page) for page in pages)
    characters = sum(len("\n".join(page)) for page in pages)
    warnings: list[IngestionWarning] = []
    if page_count and nonempty_pages < page_count:
        warnings.append(
            IngestionWarning(
                f"{page_count - nonempty_pages} PDF page(s) contain no extractable text",
                severity="warning",
            )
        )
    if page_count and characters / page_count < 30:
        warnings.append(
            IngestionWarning(
                "PDF has little extractable text; it may be scanned and require OCR",
                severity="error",
                blocking=True,
            )
        )

    repeated = _repeated_edge_lines(pages)
    removed_occurrences = 0
    rendered_pages: list[str] = []
    for page in pages:
        edges = {page[0].strip(), page[-1].strip()} if page else set()
        kept: list[str] = []
        for line in page:
            stripped = line.strip()
            if stripped and stripped in repeated and stripped in edges:
                removed_occurrences += 1
                continue
            if stripped:
                kept.append(line)
        rendered_pages.append("\n".join(kept).strip())
    markdown = "\n\n".join(page for page in rendered_pages if page)
    if repeated:
        warnings.append(
            IngestionWarning(
                f"removed {removed_occurrences} running header/footer line(s) "
                f"({len(repeated)} distinct, repeated at the edge of ≥80% of pages): "
                f"{sorted(repeated)[:3]}",
                severity="warning",
            )
        )
    metrics = {
        "pages": page_count,
        "text_pages": nonempty_pages,
        "empty_pages": page_count - nonempty_pages,
        "characters": characters,
        "removed_repeated_edge_lines": len(repeated),
        "note": "PDF ingestion is heuristic; complex layout, tables, and "
        "reading order may be imperfect",
    }
    return markdown, metrics, warnings


def _repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    """Lines appearing at a page EDGE on ≥80% of pages. The high bar keeps
    legitimately repeated body sentences (refrains, per-page conclusions)
    alive; only true running headers/footers are removed, and only at the
    edges where they occur."""
    if len(pages) < 3:
        return set()
    edge_lines: set[str] = set()
    for page in pages:
        if not page:
            continue
        edge_lines.add(page[0].strip())
        edge_lines.add(page[-1].strip())
    threshold = max(3, -(-int(len(pages) * 0.8 * 100) // 100))  # ceil(0.8*n)
    return {
        line
        for line in edge_lines
        if line
        and sum(
            bool(page) and line in {page[0].strip(), page[-1].strip()}
            for page in pages
        )
        >= threshold
    }


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()
