"""A/B/C document-level runs via the offline LLM path (no server needed).

agent=1 (groups A/B): glossary-constrained prompt per chunk + audit-retry
                      (the agent's chunk-level machinery)
agent=0 (group C):    plain prompt per chunk, no glossary, no retry
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")
ROOT = os.path.expanduser("~/tripivot-agent")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from translation_agent.agent import HierarchicalPlanner  # noqa: E402
from translation_agent.memory import MemorySystem  # noqa: E402
from translation_agent.models import Language, StyleGuide, TranslationDirection  # noqa: E402
from translation_agent.training_data import prompt_messages  # noqa: E402
from pilot_lib import load_glossary, _degenerate  # noqa: E402


def audit(hyp, pairs):
    problems = []
    missing = [f"{e}->{z}" for e, z in pairs if e.lower() in hyp.lower() and z not in hyp]
    if missing:
        problems.append("Missing terminology: " + "; ".join(missing))
    if _degenerate(hyp):
        problems.append("Repetition detected; rewrite fluently.")
    if len(hyp) < 30:
        problems.append("Too short; complete the translation.")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--agent", type=int, default=1)
    args = ap.parse_args()

    doc = open(args.input).read()
    direction = TranslationDirection(Language("en"), Language("zh"))
    style = StyleGuide(register="formal", domain="technology")
    memory = MemorySystem(Path(os.path.join(ROOT, "runs/doc-agent/_mem")))
    plan = HierarchicalPlanner().plan(doc, direction, style, memory)
    chunks = [c.source_text for c in plan.chunks if c.source_text.strip()]
    if not chunks:
        chunks = [p.strip() for p in doc.split("\n\n") if p.strip()]
    print(f"[run] {len(chunks)} chunks, agent={args.agent}", flush=True)

    glossary = load_glossary() if args.agent else None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("runs/qwen-lora/pilot_base")
    prompts, metas = [], []
    for ch in chunks:
        if args.agent:
            ex = {"source_language": "en", "target_language": "zh",
                  "domain": "technology", "source_text": ch}
            msgs = prompt_messages(ex, glossary)
        else:
            msgs = [{"role": "user", "content":
                     "Translate from English (en) to Chinese (zh). Domain: "
                     "technology. Return only the translation.\n\n" + ch}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        metas.append(ch)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    first = [o.outputs[0].text.strip() for o in outs]

    retries, retried = [], 0
    finals = list(first)
    if args.agent:
        for i, (ch, hyp) in enumerate(zip(metas, first)):
            terms = glossary.retrieve(ch, "en", "zh", domain="technology", top_k=10)
            pairs = [(t.term("en"), t.term("zh")) for t in terms]
            problems = audit(hyp, pairs)
            if problems:
                retried += 1
                user = (f"Translate from English (en) to Chinese (zh). Domain: "
                        f"technology. Return only the translation.\n\nOriginal:\n{ch}"
                        f"\n\nDraft:\n{hyp}\n\nReviewer feedback:\n" +
                        "\n".join(problems) + "\n\nOutput the corrected translation.")
                retries.append((i, tok.apply_chat_template(
                    [{"role": "user", "content": user}], tokenize=False,
                    add_generation_prompt=True)))
        if retries:
            outs2 = llm.generate([p for _, p in retries],
                                 SamplingParams(temperature=0.0, max_tokens=1024))
            for (i, _), o in zip(retries, outs2):
                if o.outputs[0].text.strip():
                    finals[i] = o.outputs[0].text.strip()

    with open(args.output, "w") as f:
        f.write("\n\n".join(finals) + "\n")
    stats = {"chunks": len(chunks), "retried": retried,
             "retry_rate": round(retried / max(1, len(chunks)), 3)}
    json.dump(stats, open(args.output + ".stats.json", "w"))
    print(f"RUN_DONE {args.output} {stats}", flush=True)


if __name__ == "__main__":
    main()
