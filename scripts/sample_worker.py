"""Isolated vLLM sampling worker — runs as a subprocess so GPU memory is fully
returned to the OS on exit (in-process vLLM leaks KV-cache reservations).

Usage:
  python sample_worker.py ADAPTER_DIR MERGED_OUT PROMPTS_JSON OUT_JSON K MAXLEN UTIL TEMP
"""
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PILOT_BASE = "runs/qwen-lora/pilot_base"


def main():
    adapter_dir, merged_out, prompts_json, out_json = sys.argv[1:5]
    k, max_len, util, temp = int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])

    m = AutoModelForCausalLM.from_pretrained(
        PILOT_BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    pm = PeftModel.from_pretrained(m, adapter_dir).merge_and_unload()
    pm.save_pretrained(merged_out, safe_serialization=True)
    AutoTokenizer.from_pretrained(PILOT_BASE).save_pretrained(merged_out)
    del pm, m

    from vllm import LLM, SamplingParams

    prompts = json.load(open(prompts_json))
    llm = LLM(model=merged_out, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=util)
    outs = llm.generate(prompts, SamplingParams(
        n=k, temperature=temp, top_p=0.95, max_tokens=1024))
    samples = []
    for o in outs:
        samples.extend(c.text for c in o.outputs)
    json.dump(samples, open(out_json, "w"), ensure_ascii=False)
    print("SAMPLES_DONE", len(samples), flush=True)


if __name__ == "__main__":
    main()
