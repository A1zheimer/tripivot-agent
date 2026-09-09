"""Shared helpers for the OPD vs GRPO pilot (same prompt format as SFT)."""
import json
import os
import random

# vLLM V1 engine core fails to spawn when CUDA is already initialized in the
# parent (training loop); V0 runs in-process on a single GPU.
os.environ.setdefault("VLLM_USE_V1", "0")

ROOT = os.path.expanduser("~/tripivot-agent")
DATA = os.path.join(ROOT, "runs/qwen-lora/prepared_examples.jsonl")


def load_examples(direction="en-zh", split="train", n=None, seed=13):
    rows = []
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") == direction and r.get("split", "train") == split:
                rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n] if n else rows


def build_prompts(tokenizer, examples, glossary):
    from translation_agent.training_data import prompt_messages

    prompts = []
    for ex in examples:
        msgs = prompt_messages(ex, glossary)
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def load_glossary():
    from translation_agent.memory import GlossaryMemory

    gm = GlossaryMemory()
    path = os.path.join(ROOT, "data/processed/glossary.jsonl")
    if os.path.exists(path):
        try:
            gm.load(path)
        except Exception:
            pass
    return gm


def sft_lora_kwargs():
    """Mirror the SFT adapter config (r / alpha / dropout / targets)."""
    import glob

    ckpts = sorted(glob.glob(os.path.join(ROOT, "runs/qwen-lora/checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    cfg = json.load(open(os.path.join(ckpts[-1], "adapter_config.json")))
    return {
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": cfg["target_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def latest_checkpoint():
    import glob

    ckpts = sorted(glob.glob(os.path.join(ROOT, "runs/qwen-lora/checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    return ckpts[-1]


COMET_LOCAL = os.path.join(ROOT, ".cache/comet/wmt22-comet-da")


def score_comet(triples, device="cuda:1"):
    """Reference-based COMET. triples: (src, hyp, ref). Uses the ModelScope
    evalscope mirror downloaded to .cache/comet/wmt22-comet-da."""
    if not os.path.exists(os.path.join(COMET_LOCAL, "checkpoints")):
        raise FileNotFoundError("comet model not downloaded")
    from comet import load_from_checkpoint

    ckpt = os.path.join(COMET_LOCAL, "checkpoints", "model.ckpt")
    model = load_from_checkpoint(ckpt, reload_hparams=False).to(device).eval()
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in triples]
    return model.predict(data, batch_size=32, gpus=1, progress_bar=False).scores


def score_bleu(hyps, refs, direction="en-zh"):
    """BLEU with a target-language-aware tokenizer (13a does not segment
    zh/my scripts and produces length-artifact numbers)."""
    tgt = direction.split("-")[1]
    tokenize = {"zh": "zh", "my": "char"}.get(tgt, "13a")  # char ≈ approximation for my
    bleu = BLEU(tokenize=tokenize)
    return bleu.corpus_score(hyps, [refs]).score / 100.0


def score_chrf(hyps, refs):
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    return [chrf.sentence_score(h, [r]).score / 100.0 for h, r in zip(hyps, refs)]


def reward_bundle(srcs, hyps, refs, comet_device="cuda:1"):
    """Composite reward per (src, hyp, ref). chrF-primary; COMET only when the
    reference-based model is present (path must match score_comet's)."""
    use_comet = os.path.exists(os.path.join(COMET_LOCAL, "checkpoints"))
    comet = (
        score_comet(list(zip(srcs, hyps, refs)), device=comet_device)
        if use_comet else [0.0] * len(hyps)
    )
    chrf = score_chrf(hyps, refs)
    out = []
    for s, h, c, f in zip(srcs, hyps, comet, chrf):
        fmt = 1.0 if (h.strip() and not _degenerate(h)) else 0.0
        lr = len(h) / max(1, len(s))
        len_pen = min(max(0.0, lr - 1.4), 0.4)
        if use_comet:
            out.append(0.5 * c + 0.35 * f + 0.15 * fmt - len_pen)
        else:
            out.append(0.75 * f + 0.25 * fmt - len_pen)
    return out


def _degenerate(text, n=4, max_rep=3):
    toks = text.split()
    ngrams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not ngrams:
        return False
    from collections import Counter

    return max(Counter(ngrams).values()) >= max_rep


def merge_adapter(model, adapter_dir, out_dir):
    """Merge a PEFT adapter in-memory and save; returns path usable by vLLM."""
    from peft import PeftModel

    pm = PeftModel.from_pretrained(model, adapter_dir)
    merged = pm.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    return out_dir
