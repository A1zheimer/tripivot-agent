"""Document-level audit for the A/B/C agent comparison (P0-3).

Metrics per group: terminology consistency (same source term -> same target
term across the whole document), heading/structure preservation, length
ratio, and empty-chunk failures. Higher terminological consistency is the
key agent-level evidence (glossary + memory + audit act across chunks)."""
import argparse
import json
import re

# terms that repeat across sections of the demo document
TERMS = {
    "pipeline parallelism": None,
    "tensor parallelism": None,
    "KV cache": None,
    "continuous batching": None,
    "quantization": None,
    "micro-batches": None,
    "all-reduce": None,
}


def heading_count(text):
    return len(re.findall(r"^#{1,3}\s+\S", text, re.M))


def consistency_score(text):
    """Fraction of repeated source-term positions translated consistently.

    Heuristic: find English term occurrences; consistency = 1 - (distinct
    zh renderings - 1)/occurrences for each term, averaged."""
    zh_segments = re.split(r"[。；\n]", text)
    scores = []
    for term in TERMS:
        hits = [s for s in zh_segments if term.lower() in s.lower()]
        if len(hits) < 2:
            continue
        # distinct translations of the term's surrounding context can't be
        # aligned without MT-specific tooling; proxy: extract the n-gram that
        # follows the term in each hit and measure overlap
        after = []
        for s in hits:
            idx = s.lower().find(term.lower())
            after.append(re.sub(r"\s+", "", s[idx + len(term): idx + len(term) + 12]))
        distinct = len(set(after))
        scores.append(1.0 - (distinct - 1) / len(hits))
    return round(sum(scores) / len(scores), 4) if scores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/doc-agent")
    args = ap.parse_args()
    src = open("demo/doc_agent_eval.md").read()
    out = {}
    for group in ("A", "B", "C"):
        try:
            text = open(f"{args.dir}/{group}.md").read()
        except FileNotFoundError:
            continue
        out[group] = {
            "headings": heading_count(text),
            "headings_src": heading_count(src),
            "terminology_consistency": consistency_score(text),
            "length_ratio": round(len(text) / max(1, len(src)), 3),
            "empty_output": len(text.strip()) < 100,
        }
    json.dump(out, open(f"{args.dir}/report.json", "w"), indent=2, ensure_ascii=False)
    print("DOC_EVAL_DONE", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
