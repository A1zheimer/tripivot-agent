"""Agentic GRPO v0: 2-turn trajectory RL with the repo's audit as environment.

Per prompt (en-zh):
  turn 1: sample G translations (T=1.0)
  audit  : glossary-missing / degenerate / over-short -> structured feedback
  turn 2: FAILING samples get one conditional revision pass
  reward : 0.5*chrF(final) + 0.2*glossary_hit + 0.2*format - 0.1*used_revision
PG update over ALL response tokens of the trajectory + KL anchor to base.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from pilot_lib import (ROOT, build_prompts, load_examples, load_glossary,
                       sft_lora_kwargs, score_chrf, _degenerate)

PILOT_BASE = os.path.join(ROOT, "runs/qwen-lora/pilot_base")
OUT = os.path.join(ROOT, "runs/agentic-r0")


def pad(seqs, device):
    n = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), n, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, : len(s)] = 1
    return ids, mask


def glossary_pairs(ex, glossary):
    terms = glossary.retrieve(ex["source_text"], "en", "zh",
                              domain=ex.get("domain", ""), top_k=10)
    return [(t.term("en"), t.term("zh")) for t in terms]


def audit(hyp, pairs):
    problems = []
    missing = [f"{e}->{z}" for e, z in pairs if e.lower() in hyp.lower() and z not in hyp]
    if missing:
        problems.append("Missing terminology (please use the specified translations): " + "; ".join(missing))
    if _degenerate(hyp):
        problems.append("The translation has repeated degradation; please rewrite it smoothly.")
    if len(hyp) < 30:
        problems.append("The translation is too short and may be missing content; please complete it.")
    return problems


def revision_prompt(tok, ex, draft, problems):
    from translation_agent.training_data import prompt_messages

    msgs = prompt_messages(ex, None)
    base = msgs[0]["content"] if msgs else "Translate the text."
    user = (base + "\n\nOriginal:\n" + ex["source_text"]
            + "\n\nDraft translation:\n" + draft
            + "\n\nReviewer feedback:\n" + "\n".join(problems)
            + "\n\nPlease output the corrected translation directly.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--micro-bs", type=int, default=1)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.makedirs(OUT, exist_ok=True)
    s_dev = "cuda:0"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PILOT_BASE)
    glossary = load_glossary()
    exs = load_examples("en-zh", "train", args.prompts, seed=41)

    policy = get_peft_model(
        AutoModelForCausalLM.from_pretrained(PILOT_BASE, torch_dtype=torch.bfloat16,
                                             device_map=s_dev),
        LoraConfig(**sft_lora_kwargs()))
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    adapter_tmp = os.path.join(OUT, "_adapter_tmp")
    merged_tmp = os.path.join(OUT, "_merged_tmp")

    def vllm_sample(prompt_list, n, temp):
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        policy.save_pretrained(adapter_tmp)
        pj, sj = os.path.join(OUT, "_p.json"), os.path.join(OUT, "_s.json")
        json.dump(prompt_list, open(pj, "w"), ensure_ascii=False)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sample_worker.py"),
                        adapter_tmp, merged_tmp, pj, sj, str(n),
                        str(args.max_len), "0.30", str(temp)], check=True, cwd=ROOT)
        torch.cuda.empty_cache()
        raw = json.load(open(sj))
        return [r.strip() for r in raw]

    prompts1 = build_prompts(tok, exs, glossary)
    print(f"[agentic] turn-1 sampling {len(prompts1)}x{args.group}", flush=True)
    first = vllm_sample(prompts1, args.group, 1.0)

    # audit -> revision prompts for failing samples only
    pairs_all = [glossary_pairs(e, glossary) for e in exs]
    rev_prompts, rev_index = [], []  # rev_index: (prompt_i, g_i)
    finals = [[] for _ in exs]
    revised = [[] for _ in exs]
    for i, ex in enumerate(exs):
        for g in range(args.group):
            hyp = first[i * args.group + g]
            problems = audit(hyp, pairs_all[i])
            if problems and hyp:
                rev_prompts.append(revision_prompt(tok, ex, hyp, problems))
                rev_index.append((i, g))
                finals[i].append(None)  # placeholder, filled after turn-2
                revised[i].append(True)
            else:
                finals[i].append(hyp)
                revised[i].append(False)
    n_rev = len(rev_prompts)
    print(f"[agentic] audit: {n_rev}/{len(exs) * args.group} samples need revision",
          flush=True)
    if n_rev:
        revs = vllm_sample(rev_prompts, 1, 1.0)
        for (i, g), r in zip(rev_index, revs):
            finals[i][g] = r if r else first[i * args.group + g]

    # rewards on finals
    flat_src, flat_hyp, flat_ref, flat_rev = [], [], [], []
    for i, ex in enumerate(exs):
        for g in range(args.group):
            h = finals[i][g]
            flat_src.append(ex["source_text"])
            flat_hyp.append(h if h else " ")
            flat_ref.append(ex["target_text"])
            flat_rev.append(1.0 if revised[i][g] else 0.0)
    chrf = score_chrf(flat_hyp, flat_ref)
    rewards = []
    for h, c, rv, pairs in zip(flat_hyp, chrf, flat_rev,
                               [p for p in pairs_all for _ in range(args.group)]):
        fmt = 1.0 if (h.strip() and not _degenerate(h)) else 0.0
        hits = sum(1 for e, z in pairs if e.lower() in h.lower() and z in h)
        tot = max(1, len(pairs))
        rewards.append(0.5 * c + 0.2 * hits / tot + 0.2 * fmt - 0.1 * rv)
    R = statistics.fmean(rewards)
    print(f"[agentic] mean reward {R:.4f} | revision rate "
          f"{statistics.fmean(flat_rev):.2f}", flush=True)

    # group-normalized advantages
    advs = []
    for i in range(len(exs)):
        g = rewards[i * args.group:(i + 1) * args.group]
        mu, sd = statistics.fmean(g), statistics.pstdev(g)
        advs.extend((r - mu) / (sd + 1e-4) for r in g)

    # trajectories: (ids, response_start, adv) — include turn-2 tokens when revised
    rows = []
    for i, (ex, pr) in enumerate(zip(exs, prompts1)):
        p1 = tok(pr, add_special_tokens=False)["input_ids"]
        for g in range(args.group):
            hyp = first[i * args.group + g]
            segs = []
            ids1 = tok(pr + hyp, add_special_tokens=False,
                       truncation=True, max_length=args.max_len)["input_ids"]
            if len(ids1) > len(p1) + 1:
                segs.append((ids1, len(p1)))
            if revised[i][g]:
                rp = rev_prompts[rev_index.index((i, g))]
                p2 = tok(rp, add_special_tokens=False)["input_ids"]
                hyp2 = finals[i][g]
                ids2 = tok(rp + hyp2, add_special_tokens=False,
                           truncation=True, max_length=args.max_len)["input_ids"]
                if len(ids2) > len(p2) + 1:
                    segs.append((ids2, len(p2)))
            for ids, plen in segs:
                rows.append((ids, plen, advs[i * args.group + g]))

    order = torch.randperm(len(rows)).tolist()
    opt.zero_grad(set_to_none=True)
    step = 0
    for mi in range(0, len(rows), args.micro_bs):
        batch = [rows[j] for j in order[mi:mi + args.micro_bs]]
        ids, mask = pad([r[0] for r in batch], s_dev)
        plens = [r[1] for r in batch]
        A = torch.tensor([r[2] for r in batch], device=s_dev, dtype=torch.float32)
        full = F.log_softmax(
            policy(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
        tgt = ids[:, 1:]
        tok_lp = full.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = torch.zeros_like(tgt, dtype=torch.bool)
        for b, pl in enumerate(plens):
            m[b, pl:mask[b].sum().item() - 1] = True
        pg = -(tok_lp * m).sum(1) / m.sum(1).clamp(min=1)
        loss_pg = (A * pg).mean()
        with torch.no_grad():
            with policy.disable_adapter():
                ref_full = F.log_softmax(
                    policy(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
        kl = (full.exp() * (full - ref_full)).sum(-1)
        loss_kl = (kl * m).sum() / m.sum().clamp(min=1)
        loss = loss_pg + args.beta_kl * loss_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0:
            print(f"[agentic] step {step} pg={loss_pg.item():.4f} "
                  f"kl={loss_kl.item():.4f}", flush=True)

    adapter_out = os.path.join(OUT, "final_adapter")
    policy.save_pretrained(adapter_out)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"mean_reward": round(R, 4),
                   "revision_rate": round(statistics.fmean(flat_rev), 3),
                   "n_rev": n_rev, "n_total": len(flat_rev)}, f, indent=2)
    print(f"AGENTIC_DONE adapter={adapter_out}", flush=True)


if __name__ == "__main__":
    main()
