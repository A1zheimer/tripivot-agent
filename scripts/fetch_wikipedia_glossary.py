#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WIKI_API = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = "TriPivot-Agent terminology builder/0.1 (local research; contact: repo owner)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def wiki_get(params: dict[str, str], retries: int = 10) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(
        f"{WIKI_API}?{query}", headers={"User-Agent": USER_AGENT}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if "error" not in payload:
                return payload
            last_error = RuntimeError(payload["error"].get("info", "Wikipedia API error"))
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            if retry_after and str(retry_after).isdigit():
                wait = min(120, int(retry_after))
            else:
                wait = min(60, 3 * (attempt + 1))
            print(f"Wikipedia HTTP {error.code}; sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"Wikipedia API failed: {last_error}")


def fetch_langlinks(titles: list[str], language: str) -> dict[str, str]:
    result: dict[str, str] = {}
    aliases: dict[str, str] = {}
    continuation: dict[str, str] = {}
    for start in range(0, len(titles), 25):
        batch = titles[start : start + 25]
        while True:
            payload = wiki_get(
                {
                    "action": "query",
                    "prop": "langlinks",
                    "titles": "|".join(batch),
                    "lllang": language,
                    "lllimit": "50",
                    "redirects": "1",
                    **continuation,
                }
            )
            query = payload.get("query", {})
            for redirect in query.get("redirects", []):
                aliases[redirect["to"]] = redirect["from"]
            for normalized in query.get("normalized", []):
                aliases[normalized["to"]] = normalized["from"]
            for page in query.get("pages", []):
                source_title = aliases.get(page.get("title", ""), page.get("title", ""))
                links = page.get("langlinks", [])
                if links:
                    result[source_title] = links[0]["title"]
            if "continue" not in payload:
                break
            continuation = payload["continue"]
            # The continuation token is common across title batches.
            continuation = {
                key: value
                for key, value in continuation.items()
                if key not in {"titles", "continue"}
            }
            if not continuation:
                break
        continuation = {}
        aliases = {}
        time.sleep(2.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a high-confidence glossary from Chinese Wikipedia cross-language titles"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--glossary", type=Path, default=Path("data/processed/glossary.domain.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/glossary.wikipedia.jsonl")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/glossary.wikipedia.report.json"),
    )
    args = parser.parse_args()

    raw = read_jsonl(args.glossary)
    title_domains: dict[str, set[str]] = defaultdict(set)
    title_urls: dict[str, str] = {}
    for domain in ("tech", "intl", "finance"):
        for row in read_jsonl(args.data_dir / f"zh.{domain}.jsonl"):
            title = str(row.get("title", "")).strip()
            if title:
                title_domains[title].add(domain)
                title_urls.setdefault(title, str(row.get("url", "")))

    raw_by_title: dict[str, dict[str, Any]] = {}
    for row in raw:
        title = row.get("terms", {}).get("zh", "").strip()
        if title in title_domains:
            raw_by_title.setdefault(title, row)

    titles = sorted(raw_by_title)
    print(f"fetching English links for {len(titles)} exact Wikipedia titles", flush=True)
    english = fetch_langlinks(titles, "en")
    print(f"fetching Burmese links for {len(titles)} exact Wikipedia titles", flush=True)
    burmese = fetch_langlinks(titles, "my")

    rows: list[dict[str, Any]] = []
    for title in titles:
        en = english.get(title, "")
        my = burmese.get(title, "")
        domains = sorted(title_domains[title])
        raw_row = raw_by_title[title]
        rows.append(
            {
                "concept_id": f"wiki-{hashlib.sha256(title.encode()).hexdigest()[:16]}",
                "terms": {"zh": title, "en": en, "my": my},
                "domain": domains[0] if len(domains) == 1 else "multi",
                "definition": "",
                "source": "wikipedia-langlinks",
                "confidence": 0.98,
                "domains": domains,
                "url": title_urls.get(title, ""),
                "raw_distill_translation": {
                    "en": raw_row.get("terms", {}).get("en", ""),
                    "my": raw_row.get("terms", {}).get("my", ""),
                },
                "complete": bool(en and my),
            }
        )
    complete = [row for row in rows if row["complete"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in complete:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": "Chinese Wikipedia exact titles + language links",
        "raw_title_matches": len(rows),
        "complete_entries": len(complete),
        "english_links": len(english),
        "burmese_links": len(burmese),
        "domains": Counter(row["domain"] for row in complete),
        "api": WIKI_API,
        "user_agent": USER_AGENT,
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
