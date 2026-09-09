"""Group C bypass: hierarchical chunking + direct per-chunk generation.

Uses the same HierarchicalPlanner as the agent, but calls the model directly
per chunk WITHOUT glossary injection, memory, reflection, or audit-retry —
isolating the contribution of the agent machinery (C vs B)."""
import argparse
import os

import torch

os.environ.setdefault("VLLM_USE_V1", "0")
ROOT = os.path.expanduser("~/tripivot-agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from translation_agent.agent import HierarchicalPlanner

    doc = open(args.input).read()
    plan = HierarchicalPlanner().plan(doc)
    chunks = [item for section in plan for item in getattr(section, "chunks", [])]
    if not chunks:  # planner API differs — fall back to simple paragraph split
        paras = [p for p in doc.split("\n\n") if p.strip()]
        chunks = paras

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85)
    prompts = []
    for ch in chunks:
        text = ch if isinstance(ch, str) else getattr(ch, "text", str(ch))
        prompts.append(
            f"Translate from English (en) to Chinese (zh). Domain: technology. "
            f"Preserve meaning, numbers, named entities, and formatting. "
            f"Return only the translation.\n\n{text}")
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    with open(args.output, "w") as f:
        for o in outs:
            f.write(o.outputs[0].text.strip() + "\n\n")
    print("BYPASS_DONE", args.output)


if __name__ == "__main__":
    main()
