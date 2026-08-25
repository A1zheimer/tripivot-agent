#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from translation_agent.domain_corpus import _entry_from_row
from translation_agent.models import GlossaryEntry

REVIEW_DECISIONS: dict[int, dict[str, str]] = {
    1: {"action": "FIX", "my": "ဗိသုကာပညာရှင်", "reason": "use singular occupational term"},
    5: {"action": "FIX", "my": "ဒါရိုက်တာဘုတ်အဖွဲ့", "reason": "canonical board-of-directors term"},
    7: {
        "action": "FIX",
        "en": "China National Petroleum Corporation",
        "my": "တရုတ်အမျိုးသာ်ရေနံကော်ပိုရေးရှင်း",
        "reason": "disambiguate the corporate entity",
    },
    8: {
        "action": "FIX",
        "en": "Commercial bank",
        "my": "စီးပွားရေးဘဏ်",
        "reason": "singular canonical term",
    },
    11: {
        "action": "FIX",
        "my": "ကွန်ဂိုဒီမိုကရက်တစ်သမ္မတနိုင်ငံ",
        "reason": "canonical country-name word order",
    },
    23: {
        "action": "FIX",
        "en": "Investment bank",
        "my": "ရင်းနှီးမြှုပ်နှံမှုဘဏ်",
        "reason": "singular canonical term",
    },
    37: {"action": "FIX", "en": "French Revolution", "reason": "remove unnecessary article"},
    40: {"action": "FIX", "my": "အမေရိကန်ကွန်ဂရက်", "reason": "name the legislature precisely"},
    63: {
        "action": "FIX",
        "my": "ကုလသမဂ္ဂဒုတိယအတွင်းရေးမှူးချုပ်",
        "reason": "canonical UN Deputy Secretary-General title",
    },
    64: {
        "action": "FIX",
        "my": "ပတ်ဝန်းကျင်ထိန်းသိမ်းရေး",
        "reason": "environmental-protection collocation",
    },
    86: {"action": "FIX", "my": "ကိုဆိုဗိုစစ်ပွဲ", "reason": "make Kosovo transliteration consistent"},
    90: {"action": "FIX", "en": "Treaty of London", "reason": "canonical treaty title"},
    100: {
        "action": "FIX",
        "my": "အစိုးရမဟုတ်သောအဖွဲ့အစည်းများ",
        "reason": "expand NGO rather than retaining acronym",
    },
    120: {"action": "FIX", "en": "Middle East War", "reason": "remove unnecessary article"},
    125: {
        "action": "FIX",
        "my": "ကုလသမဂ္ဂဖွံ့ဖြိုးမှုအစီအစဉ်",
        "reason": "UNDP is a programme, not a project",
    },
    129: {
        "action": "FIX",
        "my": "ကုလသမဂ္ဂအတွင်းရေးမှူးဌာန",
        "reason": "Secretariat is a department, not the Secretary-General",
    },
    132: {"action": "FIX", "my": "ဝါရှင်တန်ညီလာခံ", "reason": "conference noun and transliteration"},
    136: {"action": "FIX", "my": "ယာလတာညီလာခံ", "reason": "conference noun and transliteration"},
    143: {"action": "FIX", "my": "ကုဒ်သီအိုရီ", "reason": "coding theory, not code writing"},
    148: {"action": "FIX", "my": "ကွန်ပျူတာအန်နီမေးရှင်း", "reason": "computer animation term"},
    150: {
        "action": "FIX",
        "my": "ကွန်ပျူတာပုံတူဖန်တီးမှု",
        "reason": "simulation includes generation of a model",
    },
    154: {"action": "FIX", "my": "ဒေတာဖွဲ့စည်းပုံ", "reason": "data-structure collocation"},
    159: {"action": "FIX", "my": "ဖြန့်ကျက်ဒေတာဘေ့စ်", "reason": "formal distributed-database term"},
    162: {
        "action": "FIX",
        "my": "စွဲထည့်ထားသောစနစ်များ",
        "reason": "replace mixed partial-English phrase",
    },
    169: {
        "action": "FIX",
        "en": "Hunan Agricultural University",
        "my": "ဟူနန်စိုက်ပျိုးရေးတက္ကသိုလ်",
        "reason": "official university name and cleaner Burmese",
    },
    176: {"action": "FIX", "my": "အင်တာနက်စိစစ်ခြင်း", "reason": "censorship/screening collocation"},
    184: {"action": "FIX", "my": "မိုဘိုင်းစက်ပစ္စည်းများ", "reason": "devices, not general tools"},
    186: {
        "action": "FIX",
        "en": "Museum",
        "my": "ပြတိုက်",
        "reason": "singular canonical generic term",
    },
    189: {
        "action": "FIX",
        "my": "ပရိုဂရမ်ရေးနိုင်သောလော်ဂျစ်ထိန်းချုပ်ကိရိယာ",
        "reason": "replace sentence-like NLLB output with a noun phrase",
    },
    194: {
        "action": "FIX",
        "my": "အချိန်နှင့်တပြေးညီလည်ပတ်သောစနစ်",
        "reason": "real-time operating-system collocation",
    },
    195: {
        "action": "FIX",
        "en": "Minicomputer",
        "my": "မီနီကွန်ပျူတာ",
        "reason": "historical computer class, not arbitrary small computers",
    },
    197: {
        "action": "FIX",
        "en": "Software development kit",
        "my": "ဆော့ဝဲဖွံ့ဖြိုးရေးကိရိယာအစု",
        "reason": "canonical SDK term",
    },
    199: {
        "action": "FIX",
        "my": "ဆော့ဝဲအင်ဂျင်နီယာပညာ",
        "reason": "engineering discipline, not engineer",
    },
    203: {"action": "FIX", "my": "စနစ်စီမံခန့်ခွဲသူ", "reason": "system administrator occupation"},
    208: {
        "action": "FIX",
        "my": "စာသားတည်းဖြတ်ရေးဆော့ဝဲ",
        "reason": "editor software, not human editor",
    },
    209: {"action": "FIX", "my": "တွက်ချက်မှုသီအိုရီ", "reason": "theory of computation collocation"},
    210: {"action": "FIX", "my": "မြို့ပြစီမံခန့်ခွဲမှု", "reason": "urban-planning discipline"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Codex terminology review decisions")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--review", type=Path, default=Path("data/processed/glossary.silver.review.tsv")
    )
    parser.add_argument(
        "--output-review",
        type=Path,
        default=Path("data/processed/glossary.gold.review.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/glossary.gold.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/glossary.gold.report.json")
    )
    args = parser.parse_args()

    with args.review.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    if len(rows) != 214:
        raise ValueError(f"expected 214 silver review rows, found {len(rows)}")

    output_rows: list[dict[str, str]] = []
    accepted_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    fixes: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        decision = REVIEW_DECISIONS.get(index, {"action": "ACCEPT"})
        action = decision["action"]
        if action not in {"ACCEPT", "FIX", "REJECT"}:
            raise ValueError(f"invalid action at row {index}: {action}")
        reviewed = dict(row)
        reviewed["action"] = action
        reviewed["review_reason"] = decision.get(
            "reason", "all three languages and source title agree"
        )
        if action == "FIX":
            for field in ("en", "my"):
                if field in decision:
                    reviewed[field] = decision[field]
            fixes.append(
                {
                    "row": str(index),
                    "zh": reviewed["zh"],
                    "reason": reviewed["review_reason"],
                    "en": reviewed["en"],
                    "my": reviewed["my"],
                }
            )
        if action == "REJECT":
            rejected.append(
                {
                    "row": str(index),
                    "zh": reviewed["zh"],
                    "reason": reviewed["review_reason"],
                }
            )
        else:
            accepted_rows.append(reviewed)
        action_counts[action] += 1
        output_rows.append(reviewed)

    fieldnames = list(rows[0]) + ["review_reason"]
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    with args.output_review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    gold_entries: list[GlossaryEntry] = []
    seen_chinese: set[str] = set()
    for row in accepted_rows:
        chinese = row["zh"]
        if chinese in seen_chinese:
            continue
        seen_chinese.add(chinese)
        gold_entries.append(
            GlossaryEntry(
                concept_id=f"gold-{row['concept_id'].removeprefix('silver-')}",
                terms={"zh": chinese, "en": row["en"], "my": row["my"]},
                domain=row["domain"],
                definition="",
                source="codex-reviewed-wikipedia-title",
                confidence=0.95,
            )
        )
    curated = [_entry_from_row(row) for row in read_jsonl(args.data_dir / "glossary.jsonl")]
    for entry in curated:
        if entry.terms.get("zh") not in seen_chinese:
            gold_entries.append(entry)
            seen_chinese.add(entry.terms["zh"])

    write_jsonl(args.output, [asdict(entry) for entry in gold_entries])
    report = {
        "schema_version": 1,
        "input": str(args.review),
        "review_output": str(args.output_review),
        "output": str(args.output),
        "reviewer": "Codex model-assisted terminology review",
        "review_scope": "all 214 silver rows; zh/en/my compared with source title and evidence",
        "silver_rows": len(rows),
        "actions": action_counts,
        "accepted_or_fixed": len(accepted_rows),
        "rejected": len(rejected),
        "fixes": fixes,
        "rejected_rows": rejected,
        "gold_entries": len(gold_entries),
        "curated_entries_merged": len(gold_entries) - len(accepted_rows),
        "confidence": 0.95,
        "boundary": (
            "Model-assisted review, not a replacement for native Burmese translator sign-off; "
            "keep this distinction in reports."
        ),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "silver_rows",
                    "actions",
                    "accepted_or_fixed",
                    "rejected",
                    "gold_entries",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
