"""Unified eval for the OPD vs GRPO comparison (en-zh, greedy decoding).

Scores: COMET-QE, chrF++, glossary hit, length ratio, degenerate rate.
Usage: python scripts/eval_pilot.py --tag baseline|opd|grpo [--n 400]
Writes runs/eval/{tag}.json; baseline uses pilot_base, others merge adapters.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from pilot_lib import ROOT, build_prompts, load_examples, load_glossary

PILOT_BASE = os.path.join(ROOT, "runs/qwen-lora/pilot_base")
EVAL_DIR = os.path.join(ROOT, "runs/eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    choices=["baseline", "sft", "raw", "opd", "grpo"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--direction", default="en-zh",
                    choices=["en-zh", "zh-en", "zh-my", "my-zh"])
    args = ap.parse_args()
    os.makedirs(EVAL_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(PILOT_BASE)
    glossary = load_glossary()
    exs = load_examples(args.direction, "validation", args.n, seed=7)
    if len(exs) < 50:  # no validation split -> held-out tail of train (seed-disjoint)
        exs = load_examples(args.direction, "train", args.n, seed=101)
    prompts = build_prompts(tok, exs, glossary)
    # Burmese prompts tokenize long; drop any pair that would exceed the window
    keep = [i for i, p in enumerate(prompts)
            if len(tok(p, add_special_tokens=False)["input_ids"]) <= 1400]
    if len(keep) < len(prompts):
        print(f"[eval] dropping {len(prompts) - len(keep)} over-length prompts")
        exs = [exs[i] for i in keep]
        prompts = [prompts[i] for i in keep]

    ADAPTERS = {
        "baseline": None,                     # merged pilot base (checkpoint-10772)
        # pilot_base IS the full SFT model already (LoRA merged into raw Qwen);
        # stacking final_adapter again would double-apply it
        "sft": None,
        "opd": os.path.join(ROOT, "runs/opd-r1/final_adapter"),
        "grpo": os.path.join(ROOT, "runs/grpo-r1/final_adapter"),
    }
    model_dir = PILOT_BASE
    if args.tag == "raw":
        model_dir = "Qwen/Qwen2.5-7B-Instruct"  # vanilla base, from HF cache
    elif ADAPTERS.get(args.tag):
        merged = os.path.join(EVAL_DIR, f"_merged_{args.tag}")
        m = AutoModelForCausalLM.from_pretrained(PILOT_BASE, torch_dtype=torch.bfloat16,
                                                 device_map="cpu")
        PeftModel.from_pretrained(m, ADAPTERS[args.tag]).merge_and_unload().save_pretrained(
            merged, safe_serialization=True)
        tok.save_pretrained(merged)
        del m
        model_dir = merged

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_dir, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.5)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    del llm
    torch.cuda.empty_cache()
    hyps = [o.outputs[0].text.strip() for o in outs]
    srcs = [e["source_text"] for e in exs]
    refs = [e["target_text"] for e in exs]

    from pilot_lib import score_bleu, score_chrf, score_comet, _degenerate

    comet = None
    try:
        comet = score_comet(list(zip(srcs, hyps, refs)), device="cuda:0")
    except Exception as exc:  # comet model absent — chrF/BLEU still computed
        print(f"[eval] COMET unavailable ({type(exc).__name__}); chrF/BLEU only")
    chrf = score_chrf(hyps, refs)
    bleu = score_bleu(hyps, refs)
    lens = [len(h) / max(1, len(r)) for h, r in zip(hyps, refs)]
    deg = [1 if _degenerate(h) else 0 for h in hyps]

    # keep hypotheses for later inspection / re-scoring without re-decoding
    with open(os.path.join(EVAL_DIR, f"hyps_{args.tag}_{args.direction}.jsonl"), "w") as f:
        for s, h, r in zip(srcs, hyps, refs):
            f.write(json.dumps({"src": s, "hyp": h, "ref": r}, ensure_ascii=False) + "\n")

    result = {
        "tag": args.tag,
        "direction": args.direction,
        "n": len(exs),
        "comet": round(statistics.fmean(comet), 4) if comet else None,
        "bleu": round(bleu, 4),
        "comet_qe": round(statistics.fmean(comet), 4) if comet else None,
        "chrfpp": round(statistics.fmean(chrf), 4),
        "len_ratio": round(statistics.fmean(lens), 3),
        "degenerate_rate": round(statistics.fmean(deg), 4),
    }
    out_path = os.path.join(EVAL_DIR, f"{args.tag}_{args.direction}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"EVAL_DONE {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
