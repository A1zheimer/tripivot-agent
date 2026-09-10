# TriPivot-Agent: Domain-Consistent Long-Document Translation for Burmese, Chinese and English

**Technical Report** · 2026-09-10 · HKUST(GZ) HPC2

---

## Abstract

This report documents a translation system for the `my`/`zh`/`en` language set, built around a
long-horizon document agent whose backend translation policy is adapted by LoRA fine-tuning and
then compared under two post-training regimes. Four findings are reported. Domain data plus LoRA
fine-tuning yields large gains on the low-resource Burmese directions (zh→my COMET +42%, chrF++
+81%, degenerate output down from 55% to 37%), while a high-resource direction regresses (zh→en
COMET −5%), which we attribute to capacity competition inside a single adapter shared across four
directions. At pilot scale, on-policy distillation (OPD) and group-relative policy optimization
(GRPO) are statistically indistinguishable from the supervised baseline (paired bootstrap, all
p > 0.46), though their behavioural signatures differ robustly. A two-turn trajectory-level GRPO
prototype runs stably but degenerates to a single turn because the audit fires on only 0.4% of
samples. Finally, at the document level, fine-tuning raises the agent's first-pass audit rate from
47% to 95% on a 19-chunk technical document, which is direct evidence that the training work serves
the agent rather than standing apart from it.

---

## 1. Scope and Deliverables

The project was specified as an internship task with both data and system targets: at least three
domains of Chinese–English terminology corpora at 10,000 entries each, a minority language
(Burmese) terminology corpus of 20,000 entries, and a translation agent combining prompting,
memory, terminology retrieval, reflection and automatic evaluation.

Delivered artifacts:

| Artifact | Location |
|---|---|
| LoRA adapter (SFT, 4 directions × 3 domains) | ModelScope `zechlei/tripivot-qwen2.5-7b-lora` |
| Agent implementation | `src/translation_agent/` |
| Data pipeline and processed corpora | `src/translation_agent/corpus.py`, `data/processed/` |
| Evaluation harness and per-system hypotheses | `scripts/eval_pilot.py`, `artifacts/eval/` |
| Post-training pilots (OPD / GRPO / trajectory v0) | `scripts/opd_pilot.py`, `scripts/grpo_pilot.py`, `scripts/agentic_grpo_v0.py` |
| Cluster automation | `scripts/*_sbatch.sh`, `scripts/overnight_orchestrator.sh` |

---

## 2. System: The Long-Horizon Translation Agent

The agent translates a document as a sequence of chunks rather than as independent sentences. Each
chunk passes through the loop: **plan → retrieve → translate → reflect → conditionally revise →
write memory**, followed by a document-level audit once all chunks complete.

**Planning.** `HierarchicalPlanner` segments Markdown by heading, paragraph, list, table and fenced
code, then splits sentences using language-specific terminators (`。！？!?`, and `။` for Burmese).
Chunks default to six sentences, balancing available context against reflection granularity. Fenced
and indented code blocks pass through untranslated.

**Memory (four layers).** L1 is a sliding window of the last four source/translation pairs, injected
as context and explicitly marked *do not retranslate*, which suppresses near-distance terminology
drift. L2 is an SQLite episode store keyed by `(document_id, chunk_id)`. L3 is the glossary,
retrieved per chunk: English matched on word boundaries, Chinese and Burmese on exact substring,
overlapping hits resolved longest-first, ranked by (domain match, term length, confidence), top 20
injected. L4 is a progress snapshot written atomically after each chunk, enabling resumption.

**Reflection and enforcement.** `TerminologyReflector` checks only the terms that were actually
injected into the prompt, so the model is never penalized for terminology it never saw. A miss
produces a structured issue carrying the expected target term; the revision request includes the
previous draft and the specific instruction, so the model performs targeted repair rather than
blind retranslation, bounded to one round. Backends that declare `revision_capable=False`
(for example NLLB) skip the loop entirely.

**Document-level audit.** After all chunks complete, `audit_document` aggregates the realized
translation of each source term across chunks and reports missing terms and *translation conflicts*
— the same term rendered differently in different chunks — as separate categories.

---

## 3. Data Pipeline

The corpus combines ALT parallel data with domain-distilled Wikipedia text across three domains
(technology, international affairs, finance).

| Split | Examples |
|---|---|
| Train | 91,852 |
| Validation | 4,970 |
| Test | 4,982 |
| **Total** | **101,804** |

By direction: en→zh 28,669, zh→en 28,669, my→zh 22,233, zh→my 22,233.

Admission rules include NFC normalization, script-ratio checks, length-ratio bounds and Zawgyi
detection (Burmese text encoded in the legacy non-Unicode font is rejected rather than silently
mis-decoded). Trilingual alignment joins `my`–`zh` and `en`–`zh` pairs on identical Chinese pivot
sentences; **the pivot sentence hash is used to partition splits**, which prevents the same Chinese
sentence from appearing in training through one language pair and in test through another. FLORES+
is reserved for evaluation and never enters training.

The glossary is tiered: 18 human-reviewed seed terms at confidence 1.0, plus domain-distilled terms
at confidence 0.6 (20,000 entries after re-distillation; tech 9,668 / intl 7,589 / finance 2,743).
Only the curated seeds are injected by default.

---

## 4. Supervised Fine-Tuning

| Setting | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Method | LoRA, r=16, α=32, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 40,370,176 (0.53% of 7.62B) |
| Epochs / steps | 2 / 10,772 |
| Effective batch | 16 (per-device 2 × grad-accum 8) |
| Learning rate | 1e-4 |
| Max sequence length | 1,536 |
| Hardware | 1 × A800-80G, ≈11 h |

Prompts at training time carry a system role, the direction written in full (`English (en)` →
`Simplified Chinese (zh)`), a `Domain:` line, and retrieved terminology as `- source => target`.
The same builder is used at inference to avoid distribution shift.

---

## 5. Evaluation Methodology

Metrics are COMET (`wmt22-comet-da`), chrF++, BLEU, length ratio and degenerate-output rate
(4-gram repetition). Decoding is greedy.

**Tokenization.** BLEU is tokenizer-sensitive in a way that matters here. sacrebleu's default `13a`
tokenizer does not segment Chinese or Burmese; since Chinese carries no whitespace, whole clauses
collapse into single tokens and matching degenerates into exact long-string comparison, which is
extremely sensitive to output length. An initial round of scoring used the default and produced
figures that were wrong in both magnitude and sign — my→zh appeared as −18% when it is +80%, and
OPD appeared to lose 57% BLEU when it in fact gains slightly. All BLEU figures in this report use
`tokenize='zh'` for Chinese targets, character-level as an approximation for Burmese (sacrebleu
provides no Burmese tokenizer), and the default for English. Superseded values are retained in the
artifacts under `bleu_tok13a`. COMET and chrF++ are unaffected, being character-based or
model-based.

**Reported chrF++** is the mean of sentence-level scores; the corpus-level value is recorded
separately as `chrfpp_corpus` and is not interchangeable.

**Significance.** Paired bootstrap resampling (1,000 draws) is applied to all system comparisons,
as pre-registered in the design document. Full hypotheses are saved per system, so any metric can
be recomputed without re-decoding.

---

## 6. Results

### 6.1 Fine-tuning gains across four directions

Raw Qwen2.5-7B-Instruct versus the SFT LoRA, held-out sets of 233–252 examples per direction.

| Direction | COMET | BLEU (corrected) | chrF++ |
|---|---|---|---|
| zh→my | 0.445 → **0.631** (+42%) | 0.244 → **0.430** (+76%, char) | 0.187 → **0.339** (+81%) |
| my→zh | 0.685 → **0.762** (+11%) | 0.092 → **0.165** (+80%) | 0.113 → **0.184** (+62%) |
| zh→en | 0.766 → 0.727 (−5%) | 0.183 → 0.213 (+16%) | 0.535 → 0.565 (+6%) |
| en→zh | 0.794 → 0.792 (≈0) | 0.229 → 0.248 (+8%) | 0.242 → 0.258 (+7%) |

The headline result is that raw Qwen2.5-7B produces **degenerate output on 55% of zh→my inputs** —
that is, the direction is not merely poorly served but unusable — and fine-tuning reduces this to
37%. Gains on the high-resource directions are modest by comparison.

**The zh→en regression is a real cost, not noise.** COMET falls 5% while the degenerate rate
doubles from 11% to 22%. We read this as capacity competition within a single LoRA shared across
four directions: the adapter reallocates capacity toward the directions with the most to gain. The
indicated remedy is per-direction adapters or direction-aware resampling.

### 6.2 Post-training pilot: OPD versus GRPO

Both branches start from the same SFT-merged base and are evaluated on the same 252-example en→zh
held-out set.

*OPD*: chunked on-policy distillation against a Qwen2.5-14B-Instruct teacher (same tokenizer, a hard
requirement for step-wise logit distillation), loss `0.6·reverseKL + 0.4·NLL(ref)`, lr 5e-6,
≈750 steps. *GRPO*: group sampling G=8, reward `0.75·chrF++ + 0.25·format − length penalty`,
group-normalized advantage, KL anchor β=0.02, lr 1e-6, ≈250 update steps.

| Metric | Baseline (SFT) | OPD-R1 | GRPO-R1 |
|---|---|---|---|
| COMET | 0.7921 | 0.7984 | 0.7927 |
| BLEU (zh) | 0.2478 | 0.2582 | 0.2603 |
| chrF++ | 0.2579 | 0.2581 | 0.2622 |
| Length ratio | 1.320 | **0.737** | 1.223 |
| Degenerate rate | 3.57% | **0.00%** | 3.17% |

**No pairwise difference is statistically significant** (paired bootstrap, 1,000 draws, all
p > 0.46; see `artifacts/eval/significance.json`). The pilot is not powered to separate the methods.

This null result is itself informative. en→zh is precisely the direction where supervised
fine-tuning gained least (COMET flat), so sentence-level headroom is close to exhausted and limited
marginal return from post-training is the expected outcome. A full-scale run should prioritize
zh→my instead.

What *is* robust is the behavioural difference. OPD collapses output length to 0.737 of the
reference while eliminating degenerate output entirely — consistent with the mode-seeking character
of reverse KL combined with style transfer from a terser teacher, and the β=0.4 NLL anchor did not
prevent it. GRPO keeps length healthy at 1.223, and its KL anchor stayed below 0.001 throughout,
indicating a stable RL process with no sign of reward hacking.

**Methodological placement.** GRPO here is a *single-step* policy gradient: one prompt, eight
sampled translations, one scalar reward each, horizon 1. It belongs to the contextual-bandit
category, alongside RLHF-style post-training, and is not agentic RL. OPD is not RL at all — it has
no reward and no policy gradient; "on-policy" refers only to the provenance of its training data.
The comparison is therefore between a dense teacher signal and a sparse metric signal.

### 6.3 Trajectory-level prototype

A two-turn agentic GRPO v0 (initial translation → audit feedback → conditional revision →
trajectory-level policy gradient) ran to completion on the first attempt: 800 steps, KL anchor
0.0007.

The informative outcome is negative. **The audit fired on only 0.4% of samples (3/800)**, so
essentially every trajectory collapsed to a single turn and the trajectory formulation contributed
nothing. The SFT policy rarely commits errors severe enough to trip a coarse audit. The conclusion
is a design requirement rather than a result: the audit must be strict enough that revision is a
genuine decision, which means soft triggers such as chrF thresholds or exact terminology matching
rather than only hard failures.

### 6.4 Document-level evidence

All metrics above are sentence-level and therefore structurally incapable of measuring what the
agent contributes. A document-level comparison was run on one 19-chunk technical document under a
fixed audit standard (missing terminology, degeneration, truncation).

| Group | Backend | Agent loop | Chunks failing audit |
|---|---|---|---|
| A | Raw Qwen2.5-7B-Instruct | full | **52.6%** (10/19) |
| B | SFT LoRA | full | **5.3%** (1/19) |

Fine-tuning raises the agent's first-pass audit rate from 47% to 95%. This is the direct evidence
that the training work serves the agent: the reflection loop can only enforce terminology on output
that is otherwise coherent, so a backend that degenerates on half its inputs leaves the enforcement
machinery idling.

*Measurement caveat.* The cross-chunk terminology consistency metric was withdrawn: the heuristic
searched for English source terms inside Chinese output, which is not well-defined. Group C
(SFT backend with the agent loop disabled), which would isolate the agent's own contribution,
depends on the repaired metric and remains outstanding.

---

## 7. Training Infrastructure

The cluster was saturated for the duration of the project, with a 14-hour window of zero queue
turnover. Three mechanisms made the experiments feasible.

*Multi-queue racing*: each job is submitted as several clones across `emergency_gpu`, `A40` and
`long_gpu`; the first to start cancels the rest. *Degradation ladder*: two GPUs → one GPU with
time-sharing → serial execution, decided automatically by the orchestrator after 75 minutes without
an allocation. *Server-side autonomy*: the upload pipeline and orchestrator run under `nohup`, so
client disconnection has no effect — the campus VPN dropped more than eight times and the login VIP
rotated four times during the project without interrupting a single training run.

Nine classes of environment failure were diagnosed and fixed, the most instructive being: PyPI's
current torch ships as a cu130 build while the cluster driver supports at most 12.8 (pinned to
`torch==2.7.1`+cu126); Slurm executes from a spool copy so `$0` does not resolve to the repository
(fixed with `SLURM_SUBMIT_DIR`); and the HuggingFace xet CDN bypasses the configured mirror and
returns 401 (`HF_HUB_DISABLE_XET=1`). The generalizable lesson is that on heterogeneous clusters,
pinned versions beat current versions, and every step of a long automation chain must be both
fault-tolerant and idempotent.

Total cost: SFT ≈11 GPU·h on one A800; OPD and GRPO pilots ≈8 GPU·h; failed retries under
15 GPU·min.

---

## 8. Artifact Verification

The published adapter was independently verified before release:

- SHA-256 matches the registry record; 161,533,192 bytes, no truncation.
- 392 tensors = 28 layers × 7 projections × (A, B); every shape matches Qwen2.5-7B geometry
  (hidden 3,584, KV projection 512 from 4 KV heads, FFN 18,944). No missing or extra keys,
  no NaN or Inf.
- The key set PEFT expects and the key set the file provides are exactly equal; all 196 target
  Linear layers are wrapped.
- **No `lora_B` matrix is zero.** PEFT initializes B to zero, so a non-zero B is direct evidence
  that the optimizer stepped on every target module.
- Per-layer update magnitude ‖BA‖_F is uniform in the range 31–41 with no collapsed or diverged
  layer; the RMS weight delta of ≈1e-3 is consistent with 10,772 steps at lr 1e-4.

Note for reuse: the weights live under the `adapter/` subdirectory of the repository, so
`PeftModel.from_pretrained` must be pointed at that subdirectory rather than the repository root.

---

## 9. Limitations

1. The OPD/GRPO comparison is an en→zh pilot at n=252, run on the direction with the least
   headroom; no inter-method difference reaches significance.
2. Reinforcement learning was applied to the **sentence-level translation policy that the agent
   calls**, not to the agent's own planning and revision decisions, which remain rule-driven.
3. Rewards use chrF rather than COMET because the COMET-QE checkpoint is gated and could not be
   fetched on the cluster; both branches use the same scale, so the comparison remains fair.
4. zh→my references are distilled rather than human-authored, which caps the achievable metric
   ceiling; Burmese BLEU is a character-level approximation.
5. Document-level evidence currently covers only first-pass audit rate on a single document
   (§6.4). Cross-chunk consistency and the agent's own net contribution are outstanding.

---

## 10. Future Work

**Per-direction adapters.** The zh→en regression is the clearest data-driven signal in the project.
Splitting the single four-direction LoRA, or resampling by direction, directly addresses the
capacity competition it exposes.

**Agentic RL at the trajectory level.** The motivation is a specific failure mode present in our own
data: *translation conflict*, where a term is rendered differently in different chunks. This is a
purely document-level property — each chunk is individually correct and the conflict appears only
under cross-chunk comparison — so a sentence-level reward cannot express it in principle. It
requires an episode-level reward with credit assignment across steps. The components already exist:
an episode is one document, state is the current chunk plus L1 context and retrieved terminology,
actions include whether to trigger revision, and `audit_document` supplies the terminal reward.
Section 6.3 establishes the prerequisite: the audit must first be made strict enough for revision to
be a real decision.

**Scaling the comparison.** Repeat OPD and GRPO on zh→my with a larger held-out set, where headroom
is greatest and the pilot's null result is least likely to recur.

**Retrieval upgrade.** The current glossary retrieval is deliberately dependency-free exact matching.
Vector recall for paraphrased terminology, and translation-memory retrieval over prior examples,
are specified in `docs/rag_module_design.md`.

---

## Appendix: Reproduction

```bash
# Supervised fine-tuning (cluster)
sbatch scripts/qwen_lora_sbatch.sh

# Evaluation, one system × one direction
python scripts/eval_pilot.py --tag sft --direction zh-my --n 400

# Full suite: 3 systems × 4 directions, aggregated
sbatch scripts/eval_suite_sbatch.sh

# Significance testing over saved hypotheses (no re-decoding)
python scripts/significance_test.py

# Document-level A/B/C comparison
bash scripts/eval_document.sh
```

Per-system hypotheses are stored in `artifacts/eval/hyps_{tag}_{direction}.jsonl`, which allows any
metric to be recomputed without re-running inference — the mechanism by which the BLEU tokenization
error in §5 was found and corrected.
