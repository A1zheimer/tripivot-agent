"""OPD pilot: chunked on-policy distillation with reverse KL (en-zh).

GPU0 student (chunked vLLM sampling + training); GPU1 teacher
(OPD_SINGLE_GPU=1 puts both on GPU0).
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from pilot_lib import ROOT, build_prompts, load_examples, load_glossary, sft_lora_kwargs

PILOT_BASE = os.path.join(ROOT, "runs/qwen-lora/pilot_base")
TEACHER = "Qwen/Qwen2.5-14B-Instruct"
OUT = os.path.join(ROOT, "runs/opd-r1")


def build_rows(tok, chunk, prompts, samples, k, max_len):
    """Return aligned training rows: (sample_ids, ref_ids, prompt_len)."""
    rows = []
    idx = 0
    for ex, pr in zip(chunk, prompts):
        for _ in range(k):
            hyp = samples[idx].strip()
            idx += 1
            if not hyp:
                continue
            s_ids = tok(pr + hyp, add_special_tokens=False, truncation=True,
                        max_length=max_len)["input_ids"]
            r_ids = tok(pr + ex["target_text"], add_special_tokens=False,
                        truncation=True, max_length=max_len)["input_ids"]
            p_ids = tok(pr, add_special_tokens=False)["input_ids"]
            plen = min(len(p_ids), len(s_ids) - 1)
            if len(s_ids) <= plen + 1:
                continue
            rows.append((s_ids, r_ids, plen))
    return rows


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
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--grad-acc", type=int, default=16)
    args = ap.parse_args()

    single = os.environ.get("OPD_SINGLE_GPU") == "1"
    s_dev = "cuda:0"
    t_dev = "cuda:0" if single else "cuda:1"
    os.makedirs(OUT, exist_ok=True)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PILOT_BASE)
    glossary = load_glossary()
    examples = load_examples("en-zh", "train", args.chunks * args.chunk_size, seed=13)

    print("[opd] loading student + teacher", flush=True)
    student = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            PILOT_BASE, torch_dtype=torch.bfloat16, device_map=s_dev),
        LoraConfig(**sft_lora_kwargs()),
    )
    student.print_trainable_parameters()
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map=t_dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    micro_total = math.ceil(args.chunk_size * args.k / args.micro_bs)
    total_steps = args.chunks * math.ceil(micro_total / args.grad_acc)

    def lr_lambda(step, warm=20, total=total_steps):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    from torch.optim.lr_scheduler import LambdaLR

    sched = LambdaLR(opt, lr_lambda)

    from vllm import LLM, SamplingParams

    adapter_tmp = os.path.join(OUT, "_adapter_tmp")
    merged_tmp = os.path.join(OUT, "_merged_tmp")
    prompts_tmp = os.path.join(OUT, "_prompts.json")
    samples_tmp = os.path.join(OUT, "_samples.json")

    def sample_chunk(prompt_texts):
        # subprocess isolation: vLLM memory is fully returned on process exit.
        # Critical: release the training allocator's cached blocks first, or
        # the worker OOMs against the parent's retained cache.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        student.save_pretrained(adapter_tmp)
        json.dump(prompt_texts, open(prompts_tmp, "w"), ensure_ascii=False)
        import subprocess
        import sys as _sys

        subprocess.run(
            [_sys.executable, os.path.join(ROOT, "scripts", "sample_worker.py"),
             adapter_tmp, merged_tmp, prompts_tmp, samples_tmp,
             str(args.k), str(args.max_len), "0.30", "0.7"],
            check=True, cwd=ROOT)
        torch.cuda.empty_cache()
        return json.load(open(samples_tmp))

    step = 0
    for ci in range(args.chunks):
        chunk = examples[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        prompts = build_prompts(tok, chunk, glossary)
        print(f"[opd] chunk {ci + 1}/{args.chunks}: sampling", flush=True)
        samples = sample_chunk(prompts)
        rows = build_rows(tok, chunk, prompts, samples, args.k, args.max_len)
        print(f"[opd] chunk {ci + 1}: {len(rows)} training rows", flush=True)
        order = torch.randperm(len(rows)).tolist()
        opt.zero_grad(set_to_none=True)
        accum = 0
        for mi in range(math.ceil(len(rows) / args.micro_bs)):
            batch = [rows[i] for i in order[mi * args.micro_bs:(mi + 1) * args.micro_bs]]
            s_ids, s_mask = pad([r[0] for r in batch], s_dev)
            r_ids, r_mask = pad([r[1] for r in batch], s_dev)
            resp = torch.zeros_like(s_ids, dtype=torch.bool)
            for b, (_, _, plen) in enumerate(batch):
                resp[b, plen + 1:s_mask[b].sum().item()] = True
            with torch.no_grad():
                t_logits = teacher(
                    input_ids=s_ids.to(t_dev), attention_mask=s_mask.to(t_dev)
                ).logits.to(s_dev)
            s_logits = student(input_ids=s_ids, attention_mask=s_mask).logits
            s_lp = F.log_softmax(s_logits[:, :-1].float(), dim=-1)
            t_lp = F.log_softmax(t_logits[:, :-1].float(), dim=-1)
            m = resp[:, 1:].to(s_dev)
            kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)
            loss_kl = (kl * m).sum() / m.sum().clamp(min=1)
            r_logits = student(input_ids=r_ids, attention_mask=r_mask).logits
            r_lp = F.log_softmax(r_logits[:, :-1].float(), dim=-1)
            r_tgt = r_ids[:, 1:]
            r_m = r_mask[:, 1:].bool()
            loss_nll = -(r_lp.gather(-1, r_tgt.unsqueeze(-1)).squeeze(-1) * r_m).sum() \
                / r_m.sum().clamp(min=1)
            loss = args.alpha * loss_kl + args.beta * loss_nll
            (loss / args.grad_acc).backward()
            accum += 1
            if accum == args.grad_acc or mi == math.ceil(len(rows) / args.micro_bs) - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if step % 20 == 0:
                    print(f"[opd] step {step}/{total_steps} kl={loss_kl.item():.4f} "
                          f"nll={loss_nll.item():.4f}", flush=True)
        student.save_pretrained(os.path.join(OUT, "checkpoint"))
    adapter_out = os.path.join(OUT, "final_adapter")
    student.save_pretrained(adapter_out)
    print(f"OPD_DONE adapter={adapter_out}", flush=True)


if __name__ == "__main__":
    main()
