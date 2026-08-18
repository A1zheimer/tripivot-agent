from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import GlossaryEntry, Language
from .preprocessing import (
    chinese_char_count,
    has_clean_term_boundaries,
    is_clean_parallel_pair,
    is_degenerate_text,
    looks_like_zawgyi,
    normalize_myanmar,
    pack_passages,
    script_ratio,
    traditional_char_ratio,
)

WIKI_API = "https://zh.wikipedia.org/w/api.php"
WIKI_LICENSE = "CC-BY-SA-4.0"
USER_AGENT = (
    "TriPivot-Agent/0.1 (https://zh.wikipedia.org/; domain corpus collection for research)"
)
CUTOFF = "2023-01-01T00:00:00Z"

DOMAIN_CATEGORIES: dict[str, tuple[str, ...]] = {
    "tech": (
        "Category:计算机科学",
        "Category:人工智能",
        "Category:信息技术",
        "Category:软件工程",
        "Category:计算机网络",
        "Category:数据库",
        "Category:操作系统",
        "Category:信息安全",
        "Category:互联网",
        "Category:电子工程",
    ),
    "intl": (
        "Category:国际关系",
        "Category:国际组织",
        "Category:外交",
        "Category:联合国",
        "Category:国际贸易",
        "Category:国际法",
        "Category:国际政治",
        "Category:多边外交",
        "Category:国际会议",
        "Category:双边关系",
    ),
    "finance": (
        "Category:经济学",
        "Category:金融",
        "Category:财政",
        "Category:证券",
        "Category:银行",
        "Category:投资",
        "Category:保险",
        "Category:会计",
        "Category:宏观经济",
        "Category:货币",
    ),
}

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech": (
        "计算机",
        "软件",
        "算法",
        "人工智能",
        "芯片",
        "数据库",
        "操作系统",
        "互联网",
        "编程",
        "网络协议",
        "机器学习",
        "半导体",
        "数据中心",
        "源代码",
        "处理器",
    ),
    "intl": (
        "联合国",
        "外交",
        "条约",
        "双边",
        "国际法",
        "国际组织",
        "大使",
        "主权",
        "制裁",
        "多边",
        "世界贸易组织",
        "国际关系",
        "大使馆",
        "北约",
        "欧洲联盟",
    ),
    "finance": (
        "银行",
        "证券",
        "通胀",
        "利率",
        "财政",
        "货币",
        "股票",
        "债券",
        "保险",
        "投资",
        "汇率",
        "宏观经济",
        "金融",
        "基金",
        "贷款",
        "税收",
        "央行",
    ),
}

_STOP_TERMS = {
    "一个",
    "没有",
    "可以",
    "因为",
    "如果",
    "这个",
    "那个",
    "以及",
    "同时",
    "进行",
    "作为",
    "通过",
    "主要",
    "包括",
    "部分",
    "其他",
    "自己",
    "他们",
    "我们",
    "不是",
    "但是",
    "然后",
    "其中",
    "由于",
    "对于",
}


@dataclass(frozen=True, slots=True)
class DomainPassage:
    domain: str
    title: str
    url: str
    timestamp: str
    text: str
    source: str = "wikipedia-zh"

    def record_id(self) -> str:
        return hashlib.sha256(f"{self.domain}\0{self.text}".encode()).hexdigest()[:20]


def _wiki_get(params: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    url = f"{WIKI_API}?{query}"
    last_error: Exception | None = None
    for attempt in range(10):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            retry_after = error.headers.get("Retry-After") if error.headers else None
            if retry_after and str(retry_after).isdigit():
                wait = int(retry_after)
            else:
                wait = min(90, 15 * (attempt + 1))
            if error.code in {429, 500, 502, 503, 504} and attempt < 9:
                print(f"Wikipedia HTTP {error.code}, sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        if "error" in payload and payload["error"].get("code") == "maxlag":
            time.sleep(min(90, 5 * (attempt + 1)))
            continue
        time.sleep(0.35)
        return payload
    raise RuntimeError(f"Wikipedia API exceeded retry budget: {last_error}")


def _iter_category_pages(root: str, *, max_categories: int = 60) -> Iterable[int]:
    queued = [root]
    seen_cats: set[str] = set()
    seen_pages: set[int] = set()
    while queued and len(seen_cats) < max_categories:
        category = queued.pop(0)
        if category in seen_cats:
            continue
        seen_cats.add(category)
        continuation: dict[str, str] = {}
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": 50,
                "cmtype": "page|subcat",
            }
            params.update(continuation)
            payload = _wiki_get(params)
            for member in payload.get("query", {}).get("categorymembers", []):
                title = member.get("title", "")
                if member.get("ns") == 14 and title not in seen_cats:
                    queued.append(title)
                elif member.get("ns") == 0 and member.get("pageid") not in seen_pages:
                    page_id = int(member["pageid"])
                    seen_pages.add(page_id)
                    yield page_id
            if "continue" not in payload:
                break
            continuation = payload["continue"]
            time.sleep(0.05)


def _fetch_pages(page_ids: list[int]) -> list[dict[str, Any]]:
    if not page_ids:
        return []
    payload = _wiki_get(
        {
            "action": "query",
            "pageids": "|".join(str(item) for item in page_ids),
            "prop": "extracts|info|revisions",
            "exintro": 0,
            "explaintext": 1,
            "exlimit": 20,
            "inprop": "url",
            "rvprop": "timestamp",
            "redirects": 1,
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return [page for page in pages.values() if "extract" in page]


def _usable_page(page: dict[str, Any]) -> bool:
    title = page.get("title", "")
    if any(token in title for token in ("消歧义", "维基百科", "Template:", "Module:")):
        return False
    revisions = page.get("revisions") or []
    timestamp = revisions[0].get("timestamp", "") if revisions else ""
    return bool(timestamp >= CUTOFF)


def classify_domain(title: str, text: str, *, margin: float = 1.5) -> str | None:
    """Presence-based scoring with a clear-margin rule: a passage is assigned a
    domain only when the top keyword family wins by ``margin``× over the runner
    up. Ties and near-ties are discarded rather than mislabeled."""
    blob = f"{title}\n{text}"
    scores = {
        domain: sum(1 for keyword in keywords if keyword in blob)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (domain, score), (_, runner_up) = ranked[0], ranked[1]
    if score < 2:
        return None
    if score < margin * max(runner_up, 1):
        return None
    return domain


def collect_hf_wikipedia_passages(
    raw_directory: Path,
    *,
    target_per_domain: int = 10_000,
    min_cjk: int = 100,
    max_articles: int = 800_000,
) -> dict[str, list[DomainPassage]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install datasets to stream Wikipedia") from error

    raw_directory.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[DomainPassage]] = {domain: [] for domain in DOMAIN_CATEGORIES}
    seen: dict[str, set[str]] = {domain: set() for domain in DOMAIN_CATEGORIES}
    cache_paths = {
        domain: raw_directory / f"{domain}.passages.jsonl" for domain in DOMAIN_CATEGORIES
    }
    for domain, path in cache_paths.items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                if payload["text"] in seen[domain]:
                    continue
                seen[domain].add(payload["text"])
                collected[domain].append(DomainPassage(**payload))
        collected[domain] = collected[domain][:target_per_domain]
        print(f"{domain}: resumed {len(collected[domain])} cached passages", flush=True)
    if all(len(items) >= target_per_domain for items in collected.values()):
        return {domain: items[:target_per_domain] for domain, items in collected.items()}

    dataset = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
    handles = {domain: path.open("a", encoding="utf-8") for domain, path in cache_paths.items()}
    try:
        for index, row in enumerate(dataset, start=1):
            if all(len(items) >= target_per_domain for items in collected.values()):
                break
            if index > max_articles:
                break
            title = str(row.get("title") or "")
            url = str(row.get("url") or "")
            text = str(row.get("text") or "")
            domain = classify_domain(title, text)
            if domain is None or len(collected[domain]) >= target_per_domain:
                continue
            for passage_text in pack_passages(text, min_cjk=min_cjk):
                if len(collected[domain]) >= target_per_domain:
                    break
                if passage_text in seen[domain]:
                    continue
                if script_ratio(passage_text, Language.CHINESE) < 0.55:
                    continue
                passage = DomainPassage(
                    domain=domain,
                    title=title,
                    url=url,
                    timestamp="2023-11-01T00:00:00Z",
                    text=passage_text,
                )
                seen[domain].add(passage_text)
                collected[domain].append(passage)
                handles[domain].write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")
            if index % 2000 == 0:
                for handle in handles.values():
                    handle.flush()
                status = " ".join(f"{name}={len(items)}" for name, items in collected.items())
                print(f"hf wikipedia scanned {index} articles; {status}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    missing = {
        domain: len(items)
        for domain, items in collected.items()
        if len(items) < target_per_domain
    }
    if missing:
        raise RuntimeError(f"Wikipedia stream did not fill domains: {missing}")
    return {domain: items[:target_per_domain] for domain, items in collected.items()}


def collect_domain_passages(
    domain: str,
    raw_directory: Path,
    *,
    target: int = 10_000,
    min_cjk: int = 100,
) -> list[DomainPassage]:
    if domain not in DOMAIN_CATEGORIES:
        raise ValueError(f"unsupported domain: {domain}")
    raw_directory.mkdir(parents=True, exist_ok=True)
    cache_path = raw_directory / f"{domain}.passages.jsonl"
    passages: list[DomainPassage] = []
    seen_text: set[str] = set()
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                if payload["text"] in seen_text:
                    continue
                if chinese_char_count(payload["text"]) < min_cjk:
                    continue
                seen_text.add(payload["text"])
                passages.append(DomainPassage(**payload))
                if len(passages) >= target:
                    return passages[:target]

    with cache_path.open("a", encoding="utf-8") as stream:
        batch: list[int] = []

        def consume(page_ids: list[int]) -> bool:
            for page in _fetch_pages(page_ids):
                if not _usable_page(page):
                    continue
                timestamp = page["revisions"][0]["timestamp"]
                for text in pack_passages(page.get("extract", ""), min_cjk=min_cjk):
                    if text in seen_text or chinese_char_count(text) < min_cjk:
                        continue
                    if script_ratio(text, Language.CHINESE) < 0.55:
                        continue
                    passage = DomainPassage(
                        domain=domain,
                        title=page.get("title", ""),
                        url=page.get("fullurl") or page.get("canonicalurl", ""),
                        timestamp=timestamp,
                        text=text,
                    )
                    seen_text.add(text)
                    passages.append(passage)
                    stream.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")
                    if len(passages) % 200 == 0:
                        print(f"{domain}: {len(passages)}/{target} passages", flush=True)
                    if len(passages) >= target:
                        return True
            return False

        for page_id in _iter_category_pages_for_domain(domain):
            batch.append(page_id)
            if len(batch) < 8:
                continue
            done = consume(batch)
            batch = []
            time.sleep(0.2)
            if done:
                return passages[:target]
        if batch:
            consume(batch)
    if len(passages) < target:
        raise RuntimeError(f"{domain} only collected {len(passages)} passages, need {target}")
    return passages[:target]


def _iter_category_pages_for_domain(domain: str) -> Iterable[int]:
    seen: set[int] = set()
    for category in DOMAIN_CATEGORIES[domain]:
        for page_id in _iter_category_pages(category):
            if page_id in seen:
                continue
            seen.add(page_id)
            yield page_id


class _BatchTranslator:
    LANGUAGE_CODES = {
        Language.ENGLISH: "eng_Latn",
        Language.CHINESE: "zho_Hans",
        Language.BURMESE: "mya_Mymr",
    }

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self._openai = None
        self._tokenizer = None
        self._model = None
        self._opus_tokenizer = None
        self._opus_model = None
        self._device = "cpu"
        if backend == "openai":
            from .backends import OpenAICompatibleBackend

            model = os.getenv("TRANSLATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            self._openai = OpenAICompatibleBackend(model)
            return
        if backend != "nllb":
            raise ValueError("backend must be openai or nllb")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        print(f"distill translator device={self._device}", flush=True)
        nllb_name = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
        opus_name = os.getenv("OPUS_ZH_EN_MODEL", "Helsinki-NLP/opus-mt-zh-en")
        self._tokenizer = AutoTokenizer.from_pretrained(nllb_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name)
        self._model.to(self._device)
        self._model.eval()
        self._opus_tokenizer = AutoTokenizer.from_pretrained(opus_name)
        self._opus_model = AutoModelForSeq2SeqLM.from_pretrained(opus_name)
        self._opus_model.to(self._device)
        self._opus_model.eval()
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.max_new_tokens = 256
        if getattr(self._opus_model, "generation_config", None) is not None:
            self._opus_model.generation_config.max_new_tokens = 256

    def translate_many(self, texts: list[str], source: Language, target: Language) -> list[str]:
        if self._openai is not None:
            from .models import StyleGuide, TranslationDirection, TranslationRequest

            style = StyleGuide(domain="general", register="formal")
            direction = TranslationDirection(source, target)
            return [
                self._openai.translate(
                    TranslationRequest(text=text, direction=direction, style=style)
                ).text
                for text in texts
            ]
        import torch

        outputs: list[str] = []
        batch_size = 16
        use_opus = target is Language.ENGLISH and getattr(self, "_opus_model", None) is not None
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            if use_opus:
                encoded = self._opus_tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with torch.no_grad():
                    generated = self._opus_model.generate(**encoded, max_new_tokens=256)
                decoded = self._opus_tokenizer.batch_decode(generated, skip_special_tokens=True)
            else:
                assert self._tokenizer is not None and self._model is not None
                self._tokenizer.src_lang = self.LANGUAGE_CODES[source]
                bos = self._tokenizer.convert_tokens_to_ids(self.LANGUAGE_CODES[target])
                encoded = self._tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with torch.no_grad():
                    generated = self._model.generate(
                        **encoded,
                        forced_bos_token_id=bos,
                        max_new_tokens=256,
                    )
                decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(decoded)
        return [item.strip() for item in outputs]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _split_name(record_id: str) -> str:
    bucket = int(record_id[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def distill_passages(
    passages: list[DomainPassage],
    output_directory: Path,
    *,
    backend: str,
    targets: tuple[str, ...] = ("en", "my"),
    resume: bool = True,
    my_limit: int = 20_000,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    translator = _BatchTranslator(backend)
    output_directory.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = output_directory.parent / "raw" / "distill"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"backend": backend, "targets": list(targets), "domains": {}}

    by_domain: dict[str, list[DomainPassage]] = defaultdict(list)
    for passage in passages:
        by_domain[passage.domain].append(passage)

    for domain, items in by_domain.items():
        domain_stats: dict[str, Any] = {}
        for target in targets:
            work_items = items
            if target == "my" and my_limit:
                # Deterministic hash-order sampling instead of a prefix slice,
                # so the zh-my subset is not biased toward stream order.
                budget = max(1, my_limit // max(1, len(by_domain)) + 20)
                work_items = sorted(
                    items,
                    key=lambda passage: hashlib.sha256(passage.record_id().encode()).hexdigest(),
                )[:budget]
            cache_path = cache_dir / f"{domain}.zh-{target}.jsonl"
            done: dict[str, str] = {}
            if resume and cache_path.exists():
                with cache_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        payload = json.loads(line)
                        # Rejected ids are cached with an empty translation so
                        # resume does not pay for them twice.
                        done[payload["id"]] = payload.get("target_text", "")
            pending = [item for item in work_items if item.record_id() not in done]
            rejected: Counter[str] = Counter()
            target_language = Language.ENGLISH if target == "en" else Language.BURMESE
            with cache_path.open("a", encoding="utf-8") as stream:
                for start in range(0, len(pending), 8):
                    chunk = pending[start : start + 8]
                    translated_chunk = translator.translate_many(
                        [item.text for item in chunk],
                        Language.CHINESE,
                        target_language,
                    )
                    for item, translated in zip(chunk, translated_chunk, strict=True):
                        record_id = item.record_id()
                        reason = "accepted"
                        if target == "my":
                            try:
                                translated = normalize_myanmar(translated)
                            except ValueError:
                                reason = "zawgyi"
                                translated = ""
                        if reason == "accepted" and is_degenerate_text(translated):
                            reason = "degenerate"
                            translated = ""
                        if reason == "accepted":
                            accepted, reject_reason = is_clean_parallel_pair(
                                item.text,
                                translated,
                                Language.CHINESE,
                                target_language,
                                min_chars=80,
                                max_chars=2000,
                                max_length_ratio=12.0,
                            )
                            if not accepted:
                                reason = reject_reason
                                translated = ""
                        if reason != "accepted":
                            rejected[reason] += 1
                            stream.write(
                                json.dumps(
                                    {"id": record_id, "target_text": ""},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            done[record_id] = ""
                            continue
                        record = {
                            "id": record_id,
                            "domain": domain,
                            "source_language": "zh",
                            "target_language": target,
                            "source_text": item.text,
                            "target_text": translated,
                            "title": item.title,
                            "url": item.url,
                            "timestamp": item.timestamp,
                            "license": WIKI_LICENSE,
                            "distill_backend": backend,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done[record_id] = translated
                    stream.flush()
                    done_n = start + len(chunk)
                    if done_n % 1000 < 8 or done_n >= len(work_items):
                        progress = f"{min(done_n, len(work_items))}/{len(work_items)}"
                        print(f"distill {domain} zh-{target}: {progress}", flush=True)
            rows = []
            with cache_path.open(encoding="utf-8") as stream:
                for line in stream:
                    payload = json.loads(line)
                    if not payload.get("target_text"):
                        continue
                    payload["split"] = _split_name(payload["id"])
                    rows.append(payload)
            rows = rows[:10_000] if target == "en" else rows
            out_path = output_directory / f"zh-{target}.{domain}.jsonl"
            domain_stats[f"zh-{target}"] = _write_jsonl(out_path, rows)
            if rejected:
                domain_stats[f"zh-{target}_rejected"] = dict(rejected)
        stats["domains"][domain] = domain_stats
    return stats


# Unbounded CJK run: the previous {2,12} cap split long runs mid-word, so
# 5-character terms such as \u300c\u8ba1\u7b97\u673a\u79d1\u5b66\u300d could never be mined whole.
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")

# Function characters that almost never occur INSIDE a real 2-6 character
# Chinese term. Grams containing one are cross-word fragments such as
# \u300c\u4eba\u5de5\u667a\u80fd\u4e0e\u673a\u300d \u2014 a straddled word boundary.
_INTERIOR_STOP_CHARS = frozenset(
    "\u7684\u4e86\u662f\u5728\u4e0e\u6216\u53ca\u800c\u5bf9\u4e8e\u88ab\u628a\u6709\u5176\u8fd9\u4e0d\u4e5f\u5f88\u90fd\u5c31\u5e76\u5730"
)


def extract_term_candidates(texts: Iterable[str], *, limit: int = 20_000) -> list[str]:
    """Mine Chinese term candidates from domain passages.

    N-grams span 2-10 characters: real domain terms (「中华人民共和国外交部」) are
    longer than the previous 4-character cap. Fragments that start or end
    with a function character, or contain one in the interior, are rejected.
    """
    document_freq: Counter[str] = Counter()
    for text in texts:
        grams = set()
        for run in _CJK_RUN.findall(text):
            for size in range(2, min(10, len(run)) + 1):
                for index in range(0, len(run) - size + 1):
                    gram = run[index : index + size]
                    if gram in _STOP_TERMS:
                        continue
                    if not has_clean_term_boundaries(gram):
                        continue
                    if any(character in _INTERIOR_STOP_CHARS for character in gram):
                        continue
                    grams.add(gram)
        document_freq.update(grams)
    candidates = [
        term
        for term, freq in document_freq.items()
        if freq >= 4 and 3 <= len(term) <= 10
    ]
    candidates.sort(key=lambda term: (-document_freq[term], -len(term), term))
    selected: list[str] = []
    for term in candidates:
        if any(term in other for other in selected):
            continue
        selected = [other for other in selected if other not in term]
        selected.append(term)
        if len(selected) >= limit:
            break
    return selected


def glossary_entry_rejection(entry: GlossaryEntry) -> str | None:
    """Return why a (distilled) glossary entry is unusable, or None if clean."""
    zh = (entry.terms.get("zh") or "").strip()
    en = (entry.terms.get("en") or "").strip()
    my = (entry.terms.get("my") or "").strip()
    if not zh or not en or not my:
        return "missing_language"
    if entry.source == "curated":
        return None
    if not 2 <= len(zh) <= 12:
        return "zh_length"
    if zh in _STOP_TERMS or not has_clean_term_boundaries(zh):
        return "zh_boundary"
    if en[-1] in ".!?\u3002\uff01\uff1f":
        return "en_punctuation"
    if len(en.split()) > 6:
        return "en_word_count"
    if en.casefold() == zh.casefold():
        return "untranslated"
    return None


def clean_glossary_file(path: Path) -> dict[str, Any]:
    """Rewrite a glossary JSONL in place, dropping unusable distilled entries.

    Curated (human) entries are always kept. Returns before/after counts and
    per-reason removal statistics.
    """
    entries = [
        _entry_from_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept: list[GlossaryEntry] = []
    removed: Counter[str] = Counter()
    for entry in entries:
        reason = glossary_entry_rejection(entry)
        if reason is None:
            kept.append(entry)
        else:
            removed[reason] += 1
    with path.open("w", encoding="utf-8") as stream:
        for entry in kept:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return {
        "path": str(path),
        "before": len(entries),
        "after": len(kept),
        "removed": dict(removed),
    }


def clean_distilled_files(output_directory: Path) -> dict[str, Any]:
    """Drop degenerate (repetition-loop) rows from distilled parallel corpora."""
    results: dict[str, Any] = {}
    paths = sorted(output_directory.glob("zh-*.jsonl"))
    for path in paths:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kept = [row for row in rows if not is_degenerate_text(str(row.get("target_text", "")))]
        if len(kept) != len(rows):
            with path.open("w", encoding="utf-8") as stream:
                for row in kept:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        results[path.name] = {"before": len(rows), "after": len(kept)}
    return results


def _load_processed_passages(output_directory: Path) -> list[DomainPassage]:
    passages: list[DomainPassage] = []
    for domain in DOMAIN_CATEGORIES:
        path = output_directory / f"zh.{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing processed monolingual file: {path}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                passages.append(
                    DomainPassage(
                        domain=row.get("domain", domain),
                        title=row.get("title", ""),
                        url=row.get("url", ""),
                        timestamp=row.get("timestamp", ""),
                        text=row["text"],
                    )
                )
    return passages


def _load_zh_candidates(path: Path) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            term = str(row.get("zh") or "").strip()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append((str(row.get("domain") or "general"), term))
    return terms


def _entry_from_row(row: dict[str, Any]) -> GlossaryEntry:
    return GlossaryEntry(
        concept_id=row["concept_id"],
        terms=row["terms"],
        domain=row.get("domain", "general"),
        definition=row.get("definition", ""),
        source=row.get("source", "curated"),
        confidence=float(row.get("confidence", 1.0)),
    )


def distill_glossary(
    passages: list[DomainPassage],
    output_path: Path,
    *,
    backend: str,
    limit: int = 20_000,
    candidates: list[tuple[str, str]] | None = None,
) -> list[GlossaryEntry]:
    terms: list[tuple[str, str]] = []
    seen: set[str] = set()
    if candidates:
        for domain, term in candidates:
            if term in seen:
                continue
            seen.add(term)
            terms.append((domain, term))
    else:
        by_domain: dict[str, list[str]] = defaultdict(list)
        for passage in passages:
            by_domain[passage.domain].append(passage.text)
        per_domain = max(1, (limit * 2) // max(1, len(by_domain)))
        for domain, texts in by_domain.items():
            for term in extract_term_candidates(texts, limit=per_domain):
                if term in seen:
                    continue
                seen.add(term)
                terms.append((domain, term))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[GlossaryEntry] = []
    done_zh: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                entry = _entry_from_row(json.loads(line))
                zh_term = entry.terms.get("zh", "")
                if zh_term:
                    done_zh.add(zh_term)
                entries.append(entry)
        print(f"glossary distill: resumed {len(entries)} cached terms", flush=True)
    if len(entries) >= limit:
        print(f"glossary distill: {limit}/{limit}", flush=True)
        return entries[:limit]
    translator = _BatchTranslator(backend)
    pending = [
        (domain, term)
        for domain, term in terms
        if term not in done_zh and has_clean_term_boundaries(term) and term not in _STOP_TERMS
    ]
    print(f"glossary distill: {len(entries)}/{limit} pending={len(pending)}", flush=True)
    last_logged = (len(entries) // 1000) * 1000
    with output_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), 16):
            if len(entries) >= limit:
                break
            chunk = pending[start : start + 16]
            zh_terms = [term for _domain, term in chunk]
            english = translator.translate_many(zh_terms, Language.CHINESE, Language.ENGLISH)
            burmese_raw = translator.translate_many(zh_terms, Language.CHINESE, Language.BURMESE)
            for (domain, term), en_term, my_term in zip(chunk, english, burmese_raw, strict=True):
                if len(entries) >= limit:
                    break
                try:
                    burmese = normalize_myanmar(my_term)
                except ValueError:
                    continue
                if looks_like_zawgyi(burmese) or script_ratio(burmese, Language.BURMESE) < 0.35:
                    continue
                if script_ratio(en_term, Language.ENGLISH) < 0.45:
                    continue
                entry = GlossaryEntry(
                    concept_id=hashlib.sha256(term.encode()).hexdigest()[:16],
                    terms={"zh": term, "en": en_term.strip(), "my": burmese.strip()},
                    domain=domain,
                    definition="",
                    source=f"distill-{backend}",
                    confidence=0.6,
                )
                if glossary_entry_rejection(entry) is not None:
                    continue
                entries.append(entry)
                stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            stream.flush()
            if len(entries) >= last_logged + 1000 or len(entries) >= limit:
                last_logged = (len(entries) // 1000) * 1000
                print(f"glossary distill: {min(len(entries), limit)}/{limit}", flush=True)
    if last_logged < min(len(entries), limit):
        print(f"glossary distill: {min(len(entries), limit)}/{limit}", flush=True)
    return entries[:limit]


def write_monolingual(passages: list[DomainPassage], output_directory: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    by_domain: dict[str, list[DomainPassage]] = defaultdict(list)
    for passage in passages:
        by_domain[passage.domain].append(passage)
    for domain, items in by_domain.items():
        rows = []
        for item in items:
            record_id = item.record_id()
            rows.append(
                {
                    "id": record_id,
                    "domain": domain,
                    "language": "zh",
                    "text": item.text,
                    "title": item.title,
                    "url": item.url,
                    "timestamp": item.timestamp,
                    "license": WIKI_LICENSE,
                    "split": _split_name(record_id),
                    "cjk_chars": chinese_char_count(item.text),
                }
            )
        counts[domain] = _write_jsonl(output_directory / f"zh.{domain}.jsonl", rows)
    return counts


def _reproducibility_info(distill_backend: str | None) -> dict[str, Any]:
    """Record the environment needed to re-run a build identically."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "nllb_model": os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
        "opus_zh_en_model": os.getenv("OPUS_ZH_EN_MODEL", "Helsinki-NLP/opus-mt-zh-en"),
    }
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        info["git_commit"] = commit.stdout.strip() or "uncommitted"
    except (OSError, ValueError):
        info["git_commit"] = "unknown"
    if distill_backend == "nllb":
        try:
            import torch
            import transformers

            info["torch"] = torch.__version__
            info["transformers"] = transformers.__version__
        except ImportError:
            pass
    return info


def build_internship_corpora(
    raw_directory: Path,
    output_directory: Path,
    *,
    per_domain: int = 10_000,
    my_parallel: int = 20_000,
    glossary_terms: int = 20_000,
    distill_backend: str | None = None,
    collect_only: bool = False,
    glossary_only: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    if glossary_only:
        if not distill_backend:
            raise ValueError("--glossary-only requires --distill-backend")
        if glossary_terms <= 0:
            raise ValueError("--glossary-only requires --glossary-terms > 0")
        candidate_path = output_directory / "glossary.zh.candidates.jsonl"
        candidates = _load_zh_candidates(candidate_path) if candidate_path.exists() else None
        if candidates:
            print(f"glossary: loaded {len(candidates)} cached zh candidates", flush=True)
            passages: list[DomainPassage] = []
        else:
            passages = _load_processed_passages(output_directory)
            print(f"glossary: loaded {len(passages)} processed passages", flush=True)
        glossary = distill_glossary(
            passages,
            output_directory / "glossary.domain.jsonl",
            backend=distill_backend,
            limit=glossary_terms,
            candidates=candidates,
        )
        summary_path = output_directory / "domain.summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {"schema_version": 1, "license": WIKI_LICENSE, "cutoff": CUTOFF}
        summary["built_at"] = datetime.now(UTC).isoformat()
        summary["glossary_terms"] = len(glossary)
        summary["glossary_only"] = True
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    raw_wiki = raw_directory / "wikipedia"
    try:
        by_domain = collect_hf_wikipedia_passages(raw_wiki, target_per_domain=per_domain)
    except Exception as error:
        print(
            f"Hugging Face Wikipedia stream failed ({error}); "
            "falling back to Wikipedia API",
            flush=True,
        )
        by_domain = {
            domain: collect_domain_passages(domain, raw_wiki, target=per_domain)
            for domain in DOMAIN_CATEGORIES
        }
    all_passages: list[DomainPassage] = []
    collection = {}
    for domain, passages in by_domain.items():
        collection[domain] = len(passages)
        all_passages.extend(passages)
    monolingual = write_monolingual(all_passages, output_directory)
    traditional = {
        domain: round(
            sum(traditional_char_ratio(p.text) for p in passages) / max(1, len(passages)), 4
        )
        for domain, passages in by_domain.items()
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "license": WIKI_LICENSE,
        "cutoff": CUTOFF,
        "collected": collection,
        "monolingual_zh": monolingual,
        "traditional_char_ratio": traditional,
        "params": {
            "per_domain": per_domain,
            "my_parallel": my_parallel,
            "glossary_terms": glossary_terms,
        },
        "environment": _reproducibility_info(distill_backend),
        "note": (
            "Chinese passages are Wikipedia extracts updated since 2023; "
            "bilingual targets are LLM/NLLB distilled."
        ),
    }
    if collect_only or not distill_backend:
        summary_path = output_directory / "domain.summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    distill_stats = distill_passages(
        all_passages,
        output_directory,
        backend=distill_backend,
        targets=("en", "my"),
        my_limit=my_parallel,
        cache_dir=raw_directory / "distill",
    )
    # Keep 20k zh-my parallel rows across domains.
    my_rows: list[dict[str, Any]] = []
    for domain in DOMAIN_CATEGORIES:
        path = output_directory / f"zh-my.{domain}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            my_rows.extend(json.loads(line) for line in stream if line.strip())
    my_rows = my_rows[:my_parallel]
    _write_jsonl(output_directory / "zh-my.parallel.jsonl", my_rows)
    summary["distill"] = distill_stats
    summary["zh_my_parallel"] = len(my_rows)
    if glossary_terms > 0:
        glossary = distill_glossary(
            all_passages,
            output_directory / "glossary.domain.jsonl",
            backend=distill_backend,
            limit=glossary_terms,
        )
        summary["glossary_terms"] = len(glossary)
    else:
        summary["glossary_terms"] = 0
    summary_path = output_directory / "domain.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
