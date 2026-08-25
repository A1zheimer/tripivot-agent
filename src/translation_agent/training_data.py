from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .memory import GlossaryMemory
from .models import GlossaryEntry, Language

DIRECTIONS: dict[str, tuple[Language, Language]] = {
    "zh-en": (Language.CHINESE, Language.ENGLISH),
    "en-zh": (Language.ENGLISH, Language.CHINESE),
    "zh-my": (Language.CHINESE, Language.BURMESE),
    "my-zh": (Language.BURMESE, Language.CHINESE),
}


@dataclass(frozen=True, slots=True)
class TranslationExample:
    id: str
    split: str
    domain: str
    direction: str
    source_language: str
    target_language: str
    source_text: str
    target_text: str


def parse_domains(value: Iterable[str]) -> list[str]:
    domains = list(dict.fromkeys(item.strip().lower() for item in value if item.strip()))
    supported = {"tech", "intl", "finance"}
    unsupported = set(domains) - supported
    if unsupported:
        raise ValueError(
            f"unsupported domains: {sorted(unsupported)}; expected {sorted(supported)}"
        )
    if not domains:
        raise ValueError("at least one domain is required")
    return domains


def parse_directions(value: Iterable[str]) -> list[str]:
    directions = []
    for item in value:
        direction = item.strip().lower()
        if not direction:
            continue
        if direction not in DIRECTIONS:
            raise ValueError(
                f"unsupported direction {direction!r}; expected one of {sorted(DIRECTIONS)}"
            )
        if direction not in directions:
            directions.append(direction)
    if not directions:
        raise ValueError("at least one direction is required")
    return directions


def load_glossary(path: Path | None) -> GlossaryMemory:
    if path is None or not path.exists():
        return GlossaryMemory()
    entries: list[GlossaryEntry] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entries.append(GlossaryEntry(**json.loads(line)))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"invalid glossary JSONL at {path}:{line_number}") from error
    return GlossaryMemory(entries)


def prepare_translation_examples(
    data_directory: Path,
    *,
    domains: Iterable[str],
    directions: Iterable[str],
    glossary: GlossaryMemory | None = None,
) -> tuple[list[TranslationExample], dict[str, object]]:
    """Load bilingual domain records and optionally create both translation directions.

    The source files are zh->en and zh->my. Requesting the reverse direction creates a
    mirrored example from the same trusted pair, which is intentional: it exposes Qwen to
    the project's target en->zh and my->zh inputs without inventing new text.
    """

    domain_names = parse_domains(domains)
    direction_names = parse_directions(directions)
    glossary = glossary or GlossaryMemory()
    examples: list[TranslationExample] = []
    seen: set[tuple[str, str, str, str]] = set()
    file_counts: dict[str, int] = {}
    rejected: Counter[str] = Counter()

    for domain in domain_names:
        for base_target in ("en", "my"):
            path = data_directory / f"zh-{base_target}.{domain}.jsonl"
            if not path.exists():
                rejected[f"missing_file:{path.name}"] += 1
                continue
            file_count = 0
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        source_text = str(record["source_text"]).strip()
                        target_text = str(record["target_text"]).strip()
                        split = str(record.get("split", "train")).strip()
                        record_id = str(record["id"])
                    except (ValueError, TypeError, KeyError):
                        rejected[f"invalid_record:{path.name}:{line_number}"] += 1
                        continue
                    if not source_text or not target_text:
                        rejected["empty_text"] += 1
                        continue
                    if split not in {"train", "validation", "test"}:
                        rejected["invalid_split"] += 1
                        continue
                    file_count += 1

                    zh = source_text
                    other = target_text
                    candidates = {
                        f"zh-{base_target}": (zh, other),
                        f"{base_target}-zh": (other, zh),
                    }
                    for direction_name in direction_names:
                        if direction_name not in candidates:
                            continue
                        source, target = candidates[direction_name]
                        source_language, target_language = DIRECTIONS[direction_name]
                        key = (direction_name, domain, split, f"{source}\0{target}")
                        if key in seen:
                            rejected["duplicate_pair"] += 1
                            continue
                        seen.add(key)
                        examples.append(
                            TranslationExample(
                                id=f"{direction_name}-{record_id}",
                                split=split,
                                domain=domain,
                                direction=direction_name,
                                source_language=source_language.value,
                                target_language=target_language.value,
                                source_text=source,
                                target_text=target,
                            )
                        )
            file_counts[path.name] = file_count

    examples.sort(key=lambda item: (item.split, item.direction, item.domain, item.id))
    stats: dict[str, object] = {
        "examples": len(examples),
        "by_split": Counter(item.split for item in examples),
        "by_direction": Counter(item.direction for item in examples),
        "by_domain": Counter(item.domain for item in examples),
        "by_direction_split": {
            direction: Counter(
                item.split for item in examples if item.direction == direction
            )
            for direction in direction_names
        },
        "file_records": file_counts,
        "rejected": rejected,
        "glossary_entries": len(glossary.entries),
    }
    return examples, normalize_counters(stats)


def normalize_counters(value: object) -> object:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: normalize_counters(item) for key, item in value.items()}
    return value


def write_examples(examples: Iterable[TranslationExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def prompt_messages(
    example: TranslationExample | Mapping[str, object],
    glossary: GlossaryMemory | None = None,
    *,
    top_k: int = 20,
) -> list[dict[str, str]]:
    def value(key: str) -> object:
        if isinstance(example, Mapping):
            return example[key]
        return getattr(example, key)

    source_language = Language(str(value("source_language")))
    target_language = Language(str(value("target_language")))
    terms = (glossary or GlossaryMemory()).retrieve(
        str(value("source_text")),
        source_language,
        target_language,
        domain=str(value("domain")),
        top_k=top_k,
    )
    lines = [
        f"Translate from {source_language.label} ({source_language.value}) "
        f"to {target_language.label} ({target_language.value}).",
        f"Domain: {value('domain')}",
        "Preserve meaning, numbers, named entities, and formatting. Return only the translation.",
    ]
    if terms:
        pairs = []
        for entry in terms:
            source_term = entry.term(source_language)
            target_term = entry.term(target_language)
            if source_term and target_term:
                pairs.append(f"- {source_term} => {target_term}")
        if pairs:
            lines.append("Required terminology:\n" + "\n".join(pairs))
    lines.append(f"Text to translate:\n{value('source_text')}")
    return [
        {
            "role": "system",
            "content": (
                "You are a professional multilingual document translator. "
                "Use required terminology exactly when applicable."
            ),
        },
        {"role": "user", "content": "\n\n".join(lines)},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen bilingual SFT examples without loading a model"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domains", nargs="+", default=["tech", "intl", "finance"]
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["zh-en", "en-zh", "zh-my", "my-zh"],
    )
    parser.add_argument("--glossary", type=Path, default=Path("data/processed/glossary.jsonl"))
    parser.add_argument("--no-glossary", action="store_true")
    args = parser.parse_args()

    glossary = None if args.no_glossary else load_glossary(args.glossary)
    examples, stats = prepare_translation_examples(
        args.data_dir,
        domains=args.domains,
        directions=args.directions,
        glossary=glossary,
    )
    count = write_examples(examples, args.output)
    stats_path = args.output.with_suffix(args.output.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps({"written": count, **stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"written": count, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
