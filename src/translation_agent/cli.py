from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import LongHorizonTranslationAgent
from .backends import NllbBackend, OpenAICompatibleBackend
from .corpus import (
    ALT_LICENSE,
    CorpusSpec,
    align_on_pivot,
    build_glossary,
    build_parallel_corpus,
    collect_opus_corpus,
)
from .domain_corpus import build_internship_corpora
from .ingestion import (
    IngestionError,
    IngestionResult,
    ensure_ingestable,
    ingest_document,
    write_ingestion_artifacts,
)
from .memory import GlossaryMemory, MemorySystem
from .models import Language, StyleGuide


def _language(value: str) -> Language:
    try:
        return Language(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("language must be one of: en, zh, my") from error


def _pair(value: str) -> tuple[Language, Language]:
    try:
        source, target = value.split("-", 1)
        return Language(source), Language(target)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("pair must look like en-zh or my-zh") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translation-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest", help="convert a document to canonical Markdown before translation"
    )
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--report", type=Path)
    ingest.add_argument("--source", type=_language)
    ingest.add_argument("--strict", action="store_true")
    ingest.set_defaults(handler=_ingest)

    translate = subparsers.add_parser("translate", help="translate a long document")
    translate.add_argument("--input", type=Path, required=True)
    translate.add_argument("--output", type=Path, required=True)
    translate.add_argument("--source", type=_language, required=True)
    translate.add_argument("--target", type=_language, required=True)
    translate.add_argument("--backend", choices=("openai", "nllb"), default="openai")
    translate.add_argument("--model", required=True)
    translate.add_argument("--base-url")
    translate.add_argument("--api-key")
    translate.add_argument("--glossary", type=Path, default=Path("data/processed/glossary.jsonl"))
    translate.add_argument("--state-dir", type=Path, default=Path(".translation-agent"))
    translate.add_argument("--report", type=Path)
    translate.add_argument("--ingestion-report", type=Path)
    translate.add_argument(
        "--strict-ingestion",
        action="store_true",
        help="reject a non-Markdown input when any ingestion warning is present",
    )
    translate.add_argument("--domain", default="general")
    translate.add_argument("--register", default="formal")
    translate.add_argument("--timeout", type=int, default=180)
    translate.add_argument("--max-retries", type=int, default=2)
    translate.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the progress snapshot and retranslate from scratch",
    )
    translate.set_defaults(handler=_translate)

    corpus = subparsers.add_parser("corpus", help="collect and build parallel corpora")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)

    collect = corpus_subparsers.add_parser("collect-opus", help="download and filter one OPUS pair")
    collect.add_argument("--corpus", default="ALT")
    collect.add_argument("--pair", type=_pair, required=True)
    collect.add_argument("--max-records", type=int, default=10_000)
    collect.add_argument("--license", default=ALT_LICENSE)
    collect.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    collect.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    collect.add_argument("--force-download", action="store_true")
    collect.set_defaults(handler=_collect_opus)

    archive = corpus_subparsers.add_parser(
        "build-archive",
        help="filter a previously downloaded OPUS Moses archive",
    )
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--corpus", required=True)
    archive.add_argument("--pair", type=_pair, required=True)
    archive.add_argument("--max-records", type=int, default=10_000)
    archive.add_argument("--license", default="check-source-license")
    archive.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    archive.set_defaults(handler=_build_archive)

    bootstrap = corpus_subparsers.add_parser(
        "bootstrap",
        help="build ALT my-zh/en-zh corpora and exact-pivot trilingual data",
    )
    bootstrap.add_argument("--max-records", type=int, default=10_000)
    bootstrap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    bootstrap.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    bootstrap.add_argument(
        "--glossary-seed",
        type=Path,
        default=Path("data/seeds/glossary.tsv"),
    )
    bootstrap.add_argument("--force-download", action="store_true")
    bootstrap.set_defaults(handler=_bootstrap)
    glossary_seed = corpus_subparsers.add_parser(
        "fill-domains",
        help="collect internship domain corpora (tech/intl/finance) and optionally distill",
    )
    glossary_seed.add_argument("--per-domain", type=int, default=10_000)
    glossary_seed.add_argument("--my-parallel", type=int, default=20_000)
    glossary_seed.add_argument("--glossary-terms", type=int, default=20_000)
    glossary_seed.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    glossary_seed.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    glossary_seed.add_argument(
        "--distill-backend",
        choices=("nllb", "openai"),
        default=None,
    )
    glossary_seed.add_argument("--collect-only", action="store_true")
    glossary_seed.add_argument(
        "--glossary-only",
        action="store_true",
        help="distill glossary from processed passages/candidates; skip passage distill",
    )
    glossary_seed.set_defaults(handler=_fill_domains)

    clean_glossary = corpus_subparsers.add_parser(
        "clean-glossary",
        help="drop unusable distilled entries from a glossary JSONL (in place)",
    )
    clean_glossary.add_argument("--glossary", type=Path, required=True)
    clean_glossary.set_defaults(handler=_clean_glossary)

    clean_distilled = corpus_subparsers.add_parser(
        "clean-distilled",
        help="drop degenerate repetition-loop rows from distilled parallel corpora",
    )
    clean_distilled.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    clean_distilled.set_defaults(handler=_clean_distilled)
    return parser


def _ingest(args: argparse.Namespace) -> int:
    result = ingest_document(args.input, source_language=args.source)
    ensure_ingestable(result, strict=args.strict)
    report_path = args.report or args.output.with_suffix(
        args.output.suffix + ".ingestion.json"
    )
    markdown_path, report_path = write_ingestion_artifacts(
        result, args.output, report_path
    )
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "report": str(report_path),
                "source_format": result.source_format,
                "warnings": len(result.warnings),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _prepare_translation_input(args: argparse.Namespace) -> tuple[Path, IngestionResult]:
    result = ingest_document(args.input, source_language=args.source)
    ensure_ingestable(result, strict=args.strict_ingestion)
    markdown_path = args.output.with_name(f"{args.output.stem}.source.md")
    report_path = args.ingestion_report or args.output.with_name(
        f"{args.output.stem}.ingestion.json"
    )
    write_ingestion_artifacts(result, markdown_path, report_path)
    return markdown_path, result


def _translate(args: argparse.Namespace) -> int:
    input_path, ingestion = _prepare_translation_input(args)
    if args.backend == "nllb":
        backend = NllbBackend(args.model)
    else:
        backend = OpenAICompatibleBackend(
            args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    glossary = GlossaryMemory.from_jsonl(args.glossary)
    memory = MemorySystem(args.state_dir, glossary)
    agent = LongHorizonTranslationAgent(backend, args.state_dir, memory=memory)
    try:
        artifact = agent.translate_document(
            input_path.read_text(encoding="utf-8"),
            args.source,
            args.target,
            style=StyleGuide(register=args.register, domain=args.domain),
            resume=not args.no_resume,
        )
    finally:
        agent.close()
    artifact.report.ingestion = {
        "source_format": ingestion.source_format,
        "source_sha256": ingestion.source_sha256,
        "converter": ingestion.converter,
        "warnings": len(ingestion.warnings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(artifact.text, encoding="utf-8")
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(artifact.report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "report": str(report_path)}, ensure_ascii=False))
    if artifact.report.failed_chunks:
        print(
            json.dumps(
                {
                    "warning": "some chunks failed; the output keeps their source text",
                    "failed_chunks": artifact.report.failed_chunks,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


def _clean_glossary(args: argparse.Namespace) -> int:
    from .domain_corpus import clean_glossary_file

    stats = clean_glossary_file(args.glossary)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _clean_distilled(args: argparse.Namespace) -> int:
    from .domain_corpus import clean_distilled_files

    stats = clean_distilled_files(args.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _spec(args: argparse.Namespace) -> CorpusSpec:
    source, target = args.pair
    return CorpusSpec(
        corpus=args.corpus,
        source_language=source,
        target_language=target,
        max_records=args.max_records,
        license=args.license,
    )


def _collect_opus(args: argparse.Namespace) -> int:
    manifest = collect_opus_corpus(
        _spec(args),
        args.raw_dir,
        args.output_dir,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _build_archive(args: argparse.Namespace) -> int:
    manifest = build_parallel_corpus(args.archive, args.output_dir, _spec(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _bootstrap(args: argparse.Namespace) -> int:
    manifests = []
    for pair in (
        (Language.BURMESE, Language.CHINESE),
        (Language.ENGLISH, Language.CHINESE),
    ):
        manifests.append(
            collect_opus_corpus(
                CorpusSpec(
                    corpus="ALT",
                    source_language=pair[0],
                    target_language=pair[1],
                    max_records=args.max_records,
                    license=ALT_LICENSE,
                ),
                args.raw_dir,
                args.output_dir,
                force_download=args.force_download,
            )
        )

    alignments = {}
    for split in ("train", "validation", "test"):
        alignments[split] = align_on_pivot(
            args.output_dir / f"ALT.my-zh.{split}.jsonl",
            args.output_dir / f"ALT.en-zh.{split}.jsonl",
            args.output_dir / f"ALT.my-zh-en.{split}.jsonl",
        )
    glossary_entries = build_glossary(
        args.glossary_seed,
        args.output_dir / "glossary.jsonl",
    )
    summary = {
        "parallel_corpora": manifests,
        "trilingual_alignment": alignments,
        "glossary_entries": len(glossary_entries),
    }
    summary_path = args.output_dir / "bootstrap.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


def _fill_domains(args: argparse.Namespace) -> int:
    summary = build_internship_corpora(
        args.raw_dir,
        args.output_dir,
        per_domain=args.per_domain,
        my_parallel=args.my_parallel,
        glossary_terms=args.glossary_terms,
        distill_backend=args.distill_backend,
        collect_only=args.collect_only or (args.distill_backend is None and not args.glossary_only),
        glossary_only=args.glossary_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except IngestionError as error:
        print(
            json.dumps(
                {"error": "ingestion failed", "detail": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
