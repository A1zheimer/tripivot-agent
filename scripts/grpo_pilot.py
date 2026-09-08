"""GRPO pilot: group-relative policy optimization with local rewards (en-zh).

Chunked loop: vLLM samples G=8 per prompt on GPU0 -> rewards (COMET-QE on
GPU1 + chrF + format) -> group-normalized policy-gradient update with a small
KL anchor to the frozen SFT base (adapter-disabled pass).
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from pilot_lib import ROOT, build_prompts, load_examples, load_glossary, sft_lora_kwargs, reward_bundle

PILOT_BASE = os.path.join(ROOT, "runs/qwen-lora/pilot_base")
OUT = os.path.join(ROOT, "runs/grpo-r1")


def pad(seqs, device):
    n = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, : len(s)] = 1
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--grad-acc", type=int, default=8)
    args = ap.parse_args()

    single = os.environ.get("GRPO_SINGLE_GPU") == "1"
    s_dev = "cuda:0"
    comet_dev = "cuda:0" if single else "cuda:1"
    os.makedirs(OUT, exist_ok=True)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PILOT_BASE)
    glossary = load_glossary()
    examples = load_examples("en-zh", "train", args.chunks * args.chunk_size, seed=29)

    print("[grpo] loading policy", flush=True)
    policy = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            PILOT_BASE, torch_dtype=torch.bfloat16, device_map=s_dev),
        LoraConfig(**sft_lora_kwargs()),
    )
    policy.print_trainable_parameters()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    from vllm import LLM, SamplingParams

    adapter_tmp = os.path.join(OUT, "_adapter_tmp")
    merged_tmp = os.path.join(OUT, "_merged_tmp")
    prompts_tmp = os.path.join(OUT, "_prompts.json")
    samples_tmp = os.path.join(OUT, "_samples.json")

    def sample_chunk(chunk, prompts):
        # subprocess isolation: vLLM memory is fully returned on process exit.
        # Release the training allocator's cached blocks before launching.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        policy.save_pretrained(adapter_tmp)
        json.dump(prompts, open(prompts_tmp, "w"), ensure_ascii=False)
        import subprocess
        import sys as _sys

        subprocess.run(
            [_sys.executable, os.path.join(ROOT, "scripts", "sample_worker.py"),
             adapter_tmp, merged_tmp, prompts_tmp, samples_tmp,
             str(args.group), str(args.max_len), "0.30", "1.0"],
            check=True, cwd=ROOT)
        torch.cuda.empty_cache()
        texts = []
        raw = json.load(open(samples_tmp))
        for gi in range(len(chunk)):
            texts.append([t.strip() for t in raw[gi * args.group:(gi + 1) * args.group]])
        return texts

    def token_logprobs(model, ids, mask, plens):
        logits = model(input_ids=ids, attention_mask=mask).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = torch.zeros_like(tgt, dtype=torch.bool)
        for b, pl in enumerate(plens):
            m[b, pl:mask[b].sum().item() - 1] = True
        return tok_lp, m

    step = 0
    for ci in range(args.chunks):
        chunk = examples[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        prompts = build_prompts(tok, chunk, glossary)
        print(f"[grpo] chunk {ci + 1}/{args.chunks}: rollout G={args.group}", flush=True)
        gens = sample_chunk(chunk, prompts)

        flat_src, flat_hyp, flat_ref = [], [], []
        for ex, group in zip(chunk, gens):
            for h in group:
                flat_src.append(ex["source_text"])
                flat_hyp.append(h if h else " ")
                flat_ref.append(ex["target_text"])
        print(f"[grpo] scoring {len(flat_hyp)} samples", flush=True)
        rewards = reward_bundle(flat_src, flat_hyp, flat_ref, comet_device=comet_dev)

        # group-normalized advantages
        advs = []
        for gi in range(len(chunk)):
            g = rewards[gi * args.group:(gi + 1) * args.group]
            mu, sd = statistics.fmean(g), statistics.pstdev(g)
            for r in g:
                advs.append((r - mu) / (sd + 1e-4))

        rows = []
        for gi, (ex, pr) in enumerate(zip(chunk, prompts)):
            for j, hyp in enumerate(gens[gi]):
                if not hyp:
                    continue
                ids = tok(pr + hyp, add_special_tokens=False, truncation=True,
                          max_length=args.max_len)["input_ids"]
                plen = len(tok(pr, add_special_tokens=False)["input_ids"])
                if len(ids) <= plen + 1:
                    continue
                rows.append((ids, plen, advs[gi * args.group + j]))
        print(f"[grpo] chunk {ci + 1}: {len(rows)} rows", flush=True)

        order = torch.randperm(len(rows)).tolist()
        opt.zero_grad(set_to_none=True)
        accum = 0
        n_micro = math.ceil(len(rows) / args.micro_bs)
        for mi in range(n_micro):
            batch = [rows[i] for i in order[mi * args.micro_bs:(mi + 1) * args.micro_bs]]
            ids, mask = pad([r[0] for r in batch], s_dev)
            plens = [r[1] for r in batch]
            A = torch.tensor([r[2] for r in batch], device=s_dev, dtype=torch.float32)
            # single student forward serves both PG and KL terms
            full = F.log_softmax(
                policy(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            tgt = ids[:, 1:]
            tok_lp = full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = torch.zeros_like(tgt, dtype=torch.bool)
            for b, pl in enumerate(plens):
                m[b, pl:mask[b].sum().item() - 1] = True
            pg = -(tok_lp * m).sum(1) / m.sum(1).clamp(min=1)
            loss_pg = (A * pg).mean()
            # KL anchor vs frozen base (adapter disabled) — needs the FULL
            # reference distribution, not gathered token logprobs
            with torch.no_grad():
                with policy.disable_adapter():
                    ref_full = F.log_softmax(
                        policy(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            kl = (full.exp() * (full - ref_full)).sum(-1)
            loss_kl = (kl * m).sum() / m.sum().clamp(min=1)
            loss = loss_pg + args.beta_kl * loss_kl
            (loss / args.grad_acc).backward()
            accum += 1
            if accum == args.grad_acc or mi == n_micro - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if step % 10 == 0:
                    rmean = statistics.fmean(rewards)
                    print(f"[grpo] step {step} pg={loss_pg.item():.4f} "
                          f"kl={loss_kl.item():.4f} R={rmean:.4f}", flush=True)
        policy.save_pretrained(os.path.join(OUT, "checkpoint"))
    adapter_out = os.path.join(OUT, "final_adapter")
    policy.save_pretrained(adapter_out)
    print(f"GRPO_DONE adapter={adapter_out}", flush=True)


if __name__ == "__main__":
    main()
