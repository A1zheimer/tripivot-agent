from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .models import GlossaryEntry, Language
from .preprocessing import is_clean_parallel_pair, normalize_myanmar, normalize_text

OPUS_API = "https://opus.nlpl.eu/opusapi/"
ALT_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    corpus: str
    source_language: Language
    target_language: Language
    max_records: int = 10_000
    license: str = "check-source-license"


@dataclass(slots=True)
class BuildStats:
    corpus: str
    source_language: str
    target_language: str
    raw_pairs: int = 0
    accepted_pairs: int = 0
    duplicate_pairs: int = 0
    rejected: Counter[str] | None = None
    splits: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.rejected is None:
            self.rejected = Counter()
        if self.splits is None:
            self.splits = Counter()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rejected"] = dict(self.rejected or {})
        result["splits"] = dict(self.splits or {})
        return result


def _request_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "translation-agent/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resolve_opus_download(spec: CorpusSpec) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "corpus": spec.corpus,
            "source": spec.source_language.value,
            "target": spec.target_language.value,
            "preprocessing": "moses",
            "version": "latest",
        }
    )
    payload = _request_json(f"{OPUS_API}?{query}")
    candidates = payload.get("corpora", [])
    if not candidates:
        raise RuntimeError(
            f"OPUS has no Moses archive for {spec.corpus} "
            f"{spec.source_language.value}-{spec.target_language.value}"
        )
    return candidates[0]


def download_file(url: str, destination: Path, timeout: int = 120) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "translation-agent/0.1"})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary,
    ):
        temporary_path = Path(temporary.name)
        while chunk := response.read(1024 * 1024):
            hasher.update(chunk)
            temporary.write(chunk)
    temporary_path.replace(destination)
    return hasher.hexdigest()


def _language_member(archive: zipfile.ZipFile, language: Language) -> str:
    suffix = f".{language.value}"
    candidates = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and Path(name).name.endswith(suffix)
    ]
    if not candidates:
        raise RuntimeError(f"archive has no Moses text member ending in {suffix}")
    return min(candidates, key=lambda name: (name.count("/"), len(name)))


def iter_opus_pairs(
    archive_path: Path,
    source_language: Language,
    target_language: Language,
) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        source_member = _language_member(archive, source_language)
        target_member = _language_member(archive, target_language)
        with (
            archive.open(source_member) as source_binary,
            archive.open(target_member) as target_binary,
            io.TextIOWrapper(source_binary, encoding="utf-8") as source_file,
            io.TextIOWrapper(target_binary, encoding="utf-8") as target_file,
        ):
            for line_number, pair in enumerate(
                zip_longest(source_file, target_file),
                start=1,
            ):
                source, target = pair
                if source is None or target is None:
                    raise RuntimeError(f"unaligned Moses files at line {line_number}")
                yield source.rstrip("\n"), target.rstrip("\n")


def _normalize_for_language(text: str, language: Language) -> str:
    if language is Language.BURMESE:
        return normalize_myanmar(text)
    return normalize_text(text)


def _record_split(record_id: str) -> str:
    bucket = int(record_id[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def build_parallel_corpus(
    archive_path: Path,
    output_directory: Path,
    spec: CorpusSpec,
    *,
    download_metadata: dict[str, Any] | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    pair_name = f"{spec.source_language.value}-{spec.target_language.value}"
    base_name = f"{spec.corpus}.{pair_name}"
    split_paths = {
        split: output_directory / f"{base_name}.{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    handles = {split: path.open("w", encoding="utf-8") for split, path in split_paths.items()}
    stats = BuildStats(spec.corpus, spec.source_language.value, spec.target_language.value)
    seen: set[str] = set()
    try:
        for source_raw, target_raw in iter_opus_pairs(
            archive_path,
            spec.source_language,
            spec.target_language,
        ):
            stats.raw_pairs += 1
            try:
                source = _normalize_for_language(source_raw, spec.source_language)
                target = _normalize_for_language(target_raw, spec.target_language)
            except ValueError:
                stats.rejected["zawgyi"] += 1
                continue
            accepted, reason = is_clean_parallel_pair(
                source,
                target,
                spec.source_language,
                spec.target_language,
            )
            if not accepted:
                stats.rejected[reason] += 1
                continue
            digest = hashlib.sha256(
                f"{spec.source_language.value}\0{source}\0"
                f"{spec.target_language.value}\0{target}".encode()
            ).hexdigest()
            if digest in seen:
                stats.duplicate_pairs += 1
                continue
            seen.add(digest)
            record = {
                "id": digest[:20],
                "source_language": spec.source_language.value,
                "target_language": spec.target_language.value,
                "source_text": source,
                "target_text": target,
                "corpus": spec.corpus,
                "license": spec.license,
            }
            pivot_text = (
                source
                if spec.source_language is Language.CHINESE
                else target if spec.target_language is Language.CHINESE else ""
            )
            split_key = hashlib.sha256(pivot_text.encode()).hexdigest() if pivot_text else digest
            split = _record_split(split_key)
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.accepted_pairs += 1
            stats.splits[split] += 1
            if stats.accepted_pairs >= spec.max_records:
                break
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "spec": {
            "corpus": spec.corpus,
            "source_language": spec.source_language.value,
            "target_language": spec.target_language.value,
            "max_records": spec.max_records,
            "license": spec.license,
        },
        "archive": {
            "path": str(archive_path),
            "sha256": archive_sha256,
            "metadata": download_metadata or {},
        },
        "outputs": {split: str(path) for split, path in split_paths.items()},
        "stats": stats.to_dict(),
    }
    manifest_path = output_directory / f"{base_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def collect_opus_corpus(
    spec: CorpusSpec,
    raw_directory: Path,
    output_directory: Path,
    *,
    force_download: bool = False,
) -> dict[str, Any]:
    pair_name = f"{spec.source_language.value}-{spec.target_language.value}"
    metadata_path = raw_directory / spec.corpus / f"{pair_name}.opus.json"
    previous_manifest_path = output_directory / f"{spec.corpus}.{pair_name}.manifest.json"
    metadata: dict[str, Any] | None = None
    if not force_download:
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        elif previous_manifest_path.exists():
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            metadata = previous_manifest.get("archive", {}).get("metadata") or None

    if metadata and metadata.get("url"):
        archive_name = Path(urllib.parse.urlparse(metadata["url"]).path).name
    else:
        archive_name = f"{pair_name}.txt.zip"
    archive_path = raw_directory / spec.corpus / archive_name

    if force_download or not archive_path.exists():
        metadata = resolve_opus_download(spec)
        archive_name = Path(urllib.parse.urlparse(metadata["url"]).path).name
        archive_path = raw_directory / spec.corpus / archive_name
    elif metadata is None:
        metadata = {
            "corpus": spec.corpus,
            "source": spec.source_language.value,
            "target": spec.target_language.value,
            "preprocessing": "moses",
            "local_reuse": True,
        }

    if force_download or not archive_path.exists():
        archive_sha256 = download_file(metadata["url"], archive_path)
    else:
        archive_sha256 = _sha256_file(archive_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return build_parallel_corpus(
        archive_path,
        output_directory,
        spec,
        download_metadata=metadata,
        archive_sha256=archive_sha256,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def align_on_pivot(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    pivot_language: Language = Language.CHINESE,
) -> dict[str, int]:
    first = list(_read_jsonl(first_path))
    second = list(_read_jsonl(second_path))
    pivot_index: dict[str, dict[str, Any] | None] = {}
    for record in second:
        pivot = _text_for_language(record, pivot_language)
        pivot_index[pivot] = record if pivot not in pivot_index else None
    # Ambiguity must be checked on BOTH sides: a pivot repeated in the first
    # file would otherwise silently pair one opposite record with several
    # near-duplicate triples.
    first_pivot_counts = Counter(
        _text_for_language(record, pivot_language) for record in first
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    ambiguous = 0
    ambiguous_first = 0
    with output_path.open("w", encoding="utf-8") as output:
        for left in first:
            pivot = _text_for_language(left, pivot_language)
            if pivot not in pivot_index:
                continue
            right = pivot_index[pivot]
            if right is None or first_pivot_counts[pivot] > 1:
                if right is None:
                    ambiguous += 1
                else:
                    ambiguous_first += 1
                continue
            terms: dict[str, str] = {pivot_language.value: pivot}
            for record in (left, right):
                terms[record["source_language"]] = record["source_text"]
                terms[record["target_language"]] = record["target_text"]
            if set(terms) != {"my", "zh", "en"}:
                continue
            digest = hashlib.sha256(
                "\0".join(terms[language] for language in ("my", "zh", "en")).encode()
            ).hexdigest()
            output.write(
                json.dumps(
                    {
                        "id": digest[:20],
                        **terms,
                        "corpora": sorted({left["corpus"], right["corpus"]}),
                        "alignment": "exact-pivot-match",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return {
        "written": written,
        "ambiguous_pivots": ambiguous,
        "ambiguous_first_pivots": ambiguous_first,
    }


def _text_for_language(record: dict[str, Any], language: Language) -> str:
    if record["source_language"] == language.value:
        return record["source_text"]
    if record["target_language"] == language.value:
        return record["target_text"]
    raise ValueError(f"record does not contain {language.value}")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build_glossary(seed_path: Path, output_path: Path) -> list[GlossaryEntry]:
    entries: list[GlossaryEntry] = []
    with seed_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"concept_id", "en", "zh", "my"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"glossary seed must contain columns: {sorted(required)}")
        for row in reader:
            entry = GlossaryEntry(
                concept_id=row["concept_id"],
                terms={language: row[language] for language in ("en", "zh", "my")},
                domain=row.get("domain", "general"),
                definition=row.get("definition", ""),
                source=row.get("source", "curated"),
                confidence=float(row.get("confidence", "1.0")),
            )
            entries.append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entries


def copy_license_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
