#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from translation_agent.domain_corpus import _entry_from_row
from translation_agent.models import GlossaryEntry, Language
from translation_agent.preprocessing import (
    has_clean_term_boundaries,
    looks_like_zawgyi,
    normalize_myanmar,
    script_ratio,
)

_BAD_ENGLISH_RE = re.compile(
    r"[%_]|could not|temporary folder|other organiser|organisation|error occurred|"
    r"failed to|please|warning|unknown|not available",
    re.IGNORECASE,
)

_CHINESE_REJECTS = {
    # Ambiguous without the Wikipedia page's exact entity/context, or the NLLB
    # roundtrip cannot establish a canonical three-language name.
    "荷兰银行",
    "上海交通",
    "航空公司",
    "中国台湾",
    "华盛顿海军条约",
    "西撒哈拉",
    "电子游戏软件",
    "电子科技大学",
}

_ENGLISH_FIXES = {
    "董事会": "Board of directors",
    "中国石油": "China National Petroleum",
    "博鳌亚洲论坛": "Boao Forum for Asia",
    "太阳能": "Solar energy",
    "英联邦": "Commonwealth of Nations",
    "风险投资": "Venture capital",
    "巴黎条约": "Treaty of Paris",
    "越南民主共和国": "Democratic Republic of Vietnam",
    "越南共和国": "Republic of Vietnam",
    "越南战争": "Vietnam War",
    "北京工业大学": "Beijing University of Technology",
    "计算化学": "Computational chemistry",
    "计算物理学": "Computational physics",
    "中国科学技术大学": "University of Science and Technology of China",
    "计算理论": "Theory of computation",
    "编码理论": "Coding theory",
    "哈佛结构": "Harvard architecture",
    "免疫系统": "Immune system",
    "互联网号码分配局": "Internet Assigned Numbers Authority",
    "脊椎动物": "Vertebrate",
    "自然数": "Natural number",
    "可编程逻辑控制器": "Programmable logic controller",
    "上海话": "Shanghai dialect",
    "排序算法": "Sorting algorithm",
    "流媒体": "Streaming media",
    "计算机系统结构": "Computer architecture",
    "系统生物学": "Systems biology",
    "系统科学": "Systems science",
    "正方形": "Square",
    "数据库表": "Database table",
    "联合国": "United Nations",
}

_BURMESE_FIXES = {
    "中国台湾": "ထိုင်ဝမ်",
    "华盛顿海军条约": "ဝါရှင်တန် ရေတပ်စာချုပ်",
    "西撒哈拉": "အနောက် ဆာဟာရ",
    "上海话": "ရှန်ဟိုင်းစကား",
    "脊椎动物": "ကျောရိုးရှိသတ္တဝါ",
    "数据库表": "ဒေတာဘေ့စ် ဇယား",
    "流媒体": "စတրီမ်မီဒီယာ",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chinese_similarity(left: str, right: str) -> float:
    """Conservative character-level equivalence for short Chinese terms."""
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return 0.0
    if left == right or left in right or right in left:
        # A containment match can be too generic: “大学” is contained in
        # “哥伦比亚大学”, but it does not establish the full entity identity.
        return min(len(left), len(right)) / max(len(left), len(right))
    matcher = SequenceMatcher(None, left, right)
    sequence_score = matcher.ratio()
    left_counter = Counter(left)
    right_counter = Counter(right)
    intersection = sum((left_counter & right_counter).values())
    union = sum((left_counter | right_counter).values())
    bag_score = intersection / union if union else 0.0
    # Proper names can differ by one regional character while remaining close;
    # short generic terms need a stricter score.
    threshold_penalty = 0.08 if min(len(left), len(right)) <= 3 else 0.0
    return max(sequence_score, bag_score) - threshold_penalty


def english_translation_rejection(english: str) -> str | None:
    english = english.strip()
    if not english:
        return "missing_english"
    if english[-1] in ".!?，。！？":
        return "english_sentence_punctuation"
    if re.search(r"\s['’]\s?|['’]\s", english):
        return "english_malformed_apostrophe"
    if any(character in english for character in ":;\"'“”‘’"):
        return "english_disallowed_punctuation"
    if len(english.split()) > 6:
        return "english_too_long"
    if _BAD_ENGLISH_RE.search(english):
        return "english_template_or_error_text"
    if script_ratio(english, Language.ENGLISH) < 0.45:
        return "english_script"
    return None


def load_title_evidence(data_dir: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for domain in ("tech", "intl", "finance"):
        for row in read_jsonl(data_dir / f"zh.{domain}.jsonl"):
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            item = evidence.setdefault(
                title, {"domains": set(), "url": str(row.get("url", ""))}
            )
            item["domains"].add(domain)
    return evidence


class ChineseBacktranslator:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.target_id = self.tokenizer.convert_tokens_to_ids("zho_Hans")

    def translate(self, texts: list[str], source_language: Language) -> list[str]:
        source_code = {
            Language.ENGLISH: "eng_Latn",
            Language.BURMESE: "mya_Mymr",
        }[source_language]
        outputs: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            self.tokenizer.src_lang = source_code
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.target_id,
                    max_new_tokens=96,
                    num_beams=1,
                )
            outputs.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a conservative trilingual glossary from exact Wikipedia titles"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--glossary", type=Path, default=Path("data/processed/glossary.domain.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/glossary.silver.jsonl")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/glossary.silver.report.json"),
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path("data/processed/glossary.safe.jsonl"),
    )
    parser.add_argument(
        "--model", default="facebook/nllb-200-distilled-600M"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-similarity", type=float, default=0.80)
    parser.add_argument(
        "--safe-min-similarity",
        type=float,
        default=0.90,
        help="Stricter safe-glossary threshold; silver keeps the lower recall threshold.",
    )
    args = parser.parse_args()

    evidence = load_title_evidence(args.data_dir)
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen_titles: set[str] = set()
    for row in read_jsonl(args.glossary):
        entry = _entry_from_row(row)
        title = entry.terms.get("zh", "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        if title not in evidence:
            continue
        if title in _CHINESE_REJECTS:
            rejected["ambiguous_or_uncanonicalizable"] += 1
            continue
        english = _ENGLISH_FIXES.get(title, entry.terms.get("en", "")).strip()
        burmese = _BURMESE_FIXES.get(title, entry.terms.get("my", "")).strip()
        reason = english_translation_rejection(english)
        if reason:
            rejected[reason] += 1
            continue
        try:
            burmese = normalize_myanmar(burmese)
        except ValueError:
            rejected["burmese_zawgyi"] += 1
            continue
        if looks_like_zawgyi(burmese) or script_ratio(burmese, Language.BURMESE) < 0.35:
            rejected["burmese_script"] += 1
            continue
        if not has_clean_term_boundaries(title):
            rejected["chinese_boundary"] += 1
            continue
        candidates.append(
            {
                "entry": entry,
                "title": title,
                "english": english,
                "burmese": burmese,
                "domains": sorted(evidence[title]["domains"]),
                "url": evidence[title]["url"],
            }
        )

    print(f"valid exact-title candidates before backtranslation: {len(candidates)}")
    backtranslator = ChineseBacktranslator(args.model, args.device, args.batch_size)
    english_back = backtranslator.translate(
        [item["english"] for item in candidates], Language.ENGLISH
    )
    burmese_back = backtranslator.translate(
        [item["burmese"] for item in candidates], Language.BURMESE
    )

    validated: list[dict[str, Any]] = []
    for item, en_zh, my_zh in zip(candidates, english_back, burmese_back, strict=True):
        title = item["title"]
        en_score = chinese_similarity(title, en_zh.strip())
        my_score = chinese_similarity(title, my_zh.strip())
        if en_score < args.min_similarity:
            rejected["english_backtranslation_mismatch"] += 1
            continue
        if my_score < args.min_similarity:
            rejected["burmese_backtranslation_mismatch"] += 1
            continue
        item["validation"] = {
            "english_backtranslation": en_zh.strip(),
            "english_similarity": round(en_score, 4),
            "burmese_backtranslation": my_zh.strip(),
            "burmese_similarity": round(my_score, 4),
            "source_title": title,
            "source_url": item["url"],
        }
        validated.append(item)

    # A case-insensitive English key must map to exactly one Chinese canonical
    # term. Conflicting groups are intentionally rejected rather than guessed.
    strict_items = [
        item
        for item in validated
        if min(
            item["validation"]["english_similarity"],
            item["validation"]["burmese_similarity"],
        )
        >= args.safe_min_similarity
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in strict_items:
        groups[item["english"].casefold()].append(item)
    safe: list[dict[str, Any]] = []
    conflict_groups = 0
    for group in groups.values():
        chinese_terms = {item["title"] for item in group}
        if len(chinese_terms) > 1:
            conflict_groups += 1
            rejected["casefold_english_conflict"] += len(group)
            continue
        safe.extend(group)

    safe.sort(key=lambda item: (item["domains"][0], item["english"].casefold()))
    entries: list[GlossaryEntry] = []
    output_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    review_rows: list[list[str]] = []
    for item in safe:
        entry = item["entry"]
        domains = item["domains"]
        safe_entry = GlossaryEntry(
            concept_id=f"silver-{entry.concept_id}",
            terms={"zh": item["title"], "en": item["english"], "my": item["burmese"]},
            domain=domains[0] if len(domains) == 1 else "multi",
            definition="",
            source="wikipedia-title+nllb-roundtrip",
            confidence=0.85,
        )
        entries.append(safe_entry)
        output_rows.append(asdict(safe_entry))
        evidence_rows.append(
            {
                "concept_id": safe_entry.concept_id,
                "zh": safe_entry.terms["zh"],
                "evidence": item["validation"],
            }
        )
        review_rows.append(
            [
                safe_entry.concept_id,
                safe_entry.domain,
                safe_entry.terms["zh"],
                safe_entry.terms["en"],
                safe_entry.terms["my"],
                item["validation"]["english_backtranslation"],
                str(item["validation"]["english_similarity"]),
                item["validation"]["burmese_backtranslation"],
                str(item["validation"]["burmese_similarity"]),
                item["validation"]["source_url"],
                "REVIEW",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    seed_path = args.data_dir / "glossary.jsonl"
    seed_entries: list[GlossaryEntry] = []
    if seed_path.exists():
        seed_entries = [_entry_from_row(row) for row in read_jsonl(seed_path)]
    existing_chinese = {entry.terms["zh"] for entry in entries}
    merged_entries = entries + [
        entry for entry in seed_entries if entry.terms.get("zh") not in existing_chinese
    ]
    with args.merged_output.open("w", encoding="utf-8") as stream:
        for entry in merged_entries:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    evidence_path = args.output.with_suffix(".evidence.jsonl")
    with evidence_path.open("w", encoding="utf-8") as stream:
        for row in evidence_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_path = args.output.with_suffix(".review.tsv")
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "concept_id",
                "domain",
                "zh",
                "en",
                "my",
                "english_backtranslation",
                "english_similarity",
                "burmese_backtranslation",
                "burmese_similarity",
                "source_url",
                "action",
            ]
        )
        writer.writerows(review_rows)

    report = {
        "schema_version": 1,
        "source_glossary": str(args.glossary),
        "output": str(args.output),
        "evidence_output": str(evidence_path),
        "review_worksheet": str(review_path),
        "raw_entries": len(read_jsonl(args.glossary)),
        "exact_title_candidates": len(candidates),
        "validated_before_conflict_dedup": len(validated),
        "strict_candidates_before_conflict_dedup": len(strict_items),
        "safe_entries": len(output_rows),
        "merged_safe_entries": len(merged_entries),
        "curated_seed_entries_merged": len(merged_entries) - len(entries),
        "merged_output": str(args.merged_output),
        "rejected": dict(rejected),
        "casefold_conflict_groups": conflict_groups,
        "domains": Counter(entry.domain for entry in entries),
        "validation": {
            "model": args.model,
            "device": args.device,
            "method": "English->Chinese and Burmese->Chinese NLLB backtranslation",
            "min_similarity": args.min_similarity,
            "safe_min_similarity": args.safe_min_similarity,
        },
        "review_status": "silver-machine-reviewed; human review required before gold",
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
