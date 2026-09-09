"""Paired bootstrap significance tests for the pilot comparison (P0-2).

Uses the saved per-sentence hypotheses; no GPU or decoding needed.
Writes artifacts/eval/significance.json.
"""
import json
import random
from sacrebleu.metrics import BLEU, CHRF

N_BOOT = 1000
SEED = 13


def load_hyps(tag):
    rows = [json.loads(l) for l in open(f"artifacts/eval/hyps_{tag}_en-zh.jsonl")]
    return [r["hyp"] for r in rows], [r["ref"] for r in rows]


def sent_bleu(hyps, refs):
    # corpus BLEU is not decomposable per-sentence; use smoothed sentence BLEU
    out = []
    for h, r in zip(hyps, refs):
        out.append(BLEU(tokenize="zh", effective_order=True).sentence_score(h, [r]).score / 100.0)
    return out


def sent_chrf(hyps, refs):
    c = CHRF(word_order=2)
    return [c.sentence_score(h, [r]).score / 100.0 for h, r in zip(hyps, refs)]


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """Returns (mean_a, mean_b, p_value two-sided for mean diff == 0)."""
    rng = random.Random(seed)
    n = len(a)
    da = [x - y for x, y in zip(a, b)]
    obs = sum(da) / n
    worse = 0
    for _ in range(n_boot):
        s = sum(da[rng.randrange(n)] for _ in range(n)) / n
        if abs(s) >= abs(obs):
            worse += 1
    return sum(a) / n, sum(b) / n, worse / n_boot


def main():
    results = {}
    hyps = {}
    for tag in ("baseline", "opd", "grpo"):
        h, r = load_hyps(tag)
        hyps[tag] = (h, r)
    refs = hyps["baseline"][1]
    assert refs == hyps["opd"][1] == hyps["grpo"][1], "ref mismatch"

    for metric_name, fn in (("bleu_zh_smoothed", sent_bleu), ("chrfpp", sent_chrf)):
        vals = {t: fn(hyps[t][0], refs) for t in hyps}
        for cand in ("opd", "grpo"):
            ma, mb, p = paired_bootstrap(vals[cand], vals["baseline"])
            results[f"{metric_name}_{cand}_vs_baseline"] = {
                "mean_candidate": round(ma, 4), "mean_baseline": round(mb, 4),
                "p_value": round(p, 4), "significant_at_0.05": p < 0.05,
            }
    json.dump(results, open("artifacts/eval/significance.json", "w"), indent=2)
    for k, v in results.items():
        print(k, v)
    print("SIGNIFICANCE_DONE")


if __name__ == "__main__":
    main()
