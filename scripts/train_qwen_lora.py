#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from translation_agent.models import Language  # noqa: E402
from translation_agent.preprocessing import term_occurs  # noqa: E402
from translation_agent.training_data import (  # noqa: E402
    load_glossary,
    prepare_translation_examples,
    prompt_messages,
    write_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data and train a Qwen2.5-7B-Instruct LoRA translation adapter"
    )
    parser.add_argument(
        "--model-name-or-path",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen-lora"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--domains", nargs="+", default=["tech", "intl", "finance"])
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["zh-en", "en-zh", "zh-my", "my-zh"],
        help="One or more of: zh-en en-zh zh-my my-zh",
    )
    parser.add_argument("--glossary", type=Path, default=Path("data/processed/glossary.jsonl"))
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("--glossary-top-k", type=int, default=20)

    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=1000,
        help="Stratified validation subset used for eval_loss and generation metrics.",
    )
    parser.add_argument("--preprocessing-workers", type=int, default=8)
    parser.add_argument("--dataloader-workers", type=int, default=4)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        default="auto",
        choices=("auto", "true", "false"),
        help="Use bf16 when supported. A800 defaults to true.",
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=("auto", "sdpa", "eager", "flash_attention_2"),
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--resume",
        choices=("auto", "always", "never"),
        default="auto",
        help="Automatically resume from the newest checkpoint under output-dir.",
    )
    parser.add_argument(
        "--eval-generation-samples",
        type=int,
        default=200,
        help="Validation examples generated after training; 0 disables generation metrics.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=768)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --prepare-only.")
    return parser.parse_args()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": run("status", "--porcelain"),
    }


def bool_arg(value: str) -> bool:
    return value == "true"


def resolve_bf16(value: str) -> bool:
    if value != "auto":
        return bool_arg(value)
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def cap_examples(
    examples: list[Any],
    limit: int | None,
    *,
    seed: int,
) -> list[Any]:
    if limit is None or limit <= 0 or len(examples) <= limit:
        return examples
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for example in examples:
        groups[(example.direction, example.domain)].append(example)
    for group in groups.values():
        random.Random(seed).shuffle(group)
    selected: list[Any] = []
    keys = list(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


class TranslationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            ids = feature["input_ids"]
            mask = feature.get("attention_mask", [1] * len(ids))
            current_labels = feature["labels"]
            padding = maximum - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_mask.append(mask + [0] * padding)
            labels.append(current_labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def encode_dataset(
    examples: list[Any],
    tokenizer: Any,
    *,
    glossary: Any,
    args: argparse.Namespace,
    desc: str,
) -> tuple[Dataset, dict[str, int]]:
    raw = Dataset.from_list([asdict(example) for example in examples])

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        messages = prompt_messages(
            example,
            glossary,
            top_k=args.glossary_top_k,
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = f"{example['target_text']}{tokenizer.eos_token or ''}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        if len(full_ids) > args.max_length or len(full_ids) <= len(prompt_ids):
            return {"input_ids": [], "attention_mask": [], "labels": [], "length": 0}
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "length": len(full_ids),
        }

    tokenized = raw.map(
        tokenize,
        remove_columns=raw.column_names,
        num_proc=args.preprocessing_workers or None,
        desc=desc,
    )
    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda example: len(example["input_ids"]) > 1
        and any(label != -100 for label in example["labels"]),
        num_proc=args.preprocessing_workers or None,
        desc=f"{desc}: filter valid",
    )
    stats = {
        "before": before,
        "after": len(tokenized),
        "dropped": before - len(tokenized),
    }
    return tokenized, stats


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.split("-", 1)[1]))


def load_model(args: argparse.Namespace, bf16: bool) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if bf16 else torch.float32,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    glossary: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    if args.eval_generation_samples <= 0:
        return {"enabled": False}
    import sacrebleu

    selected = cap_examples(
        examples, args.eval_generation_samples, seed=args.seed + 17
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    predictions_path = output_dir / "validation_predictions.jsonl"
    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict[str, Any]] = []
    term_hits = 0
    term_total = 0
    term_hits_by_direction: dict[str, int] = defaultdict(int)
    term_total_by_direction: dict[str, int] = defaultdict(int)

    with torch.no_grad(), predictions_path.open("w", encoding="utf-8") as stream:
        for start in range(0, len(selected), args.generation_batch_size):
            batch = selected[start : start + args.generation_batch_size]
            encoded = tokenizer(
                [
                    tokenizer.apply_chat_template(
                        prompt_messages(example, glossary, top_k=args.glossary_top_k),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for example in batch
                ],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            )
            encoded = encoded.to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            for example, output_ids in zip(batch, generated, strict=True):
                new_tokens = output_ids[encoded["input_ids"].shape[-1] :]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                reference = example.target_text.strip()
                predictions.append(prediction)
                references.append(reference)

                source_language = Language(example.source_language)
                target_language = Language(example.target_language)
                terms = glossary.retrieve(
                    example.source_text,
                    source_language,
                    target_language,
                    domain=example.domain,
                    top_k=args.glossary_top_k,
                )
                required_terms: list[dict[str, str]] = []
                for entry in terms:
                    source_term = entry.term(source_language)
                    target_term = entry.term(target_language)
                    if not source_term or not target_term:
                        continue
                    if term_occurs(example.source_text, source_term, source_language):
                        term_total += 1
                        term_total_by_direction[example.direction] += 1
                        hit = term_occurs(prediction, target_term, target_language)
                        term_hits += int(hit)
                        term_hits_by_direction[example.direction] += int(hit)
                        required_terms.append(
                            {
                                "source": source_term,
                                "target": target_term,
                                "hit": hit,
                            }
                        )
                row = {
                    "id": example.id,
                    "direction": example.direction,
                    "domain": example.domain,
                    "source_text": example.source_text,
                    "reference": reference,
                    "prediction": prediction,
                    "required_terms": required_terms,
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

    def score_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        group_predictions = [row["prediction"] for row in group_rows]
        group_references = [row["reference"] for row in group_rows]
        bleu = sacrebleu.BLEU().corpus_score(group_predictions, [group_references])
        chrf = sacrebleu.CHRF().corpus_score(group_predictions, [group_references])
        direction = group_rows[0]["direction"] if group_rows else ""
        hits = term_hits_by_direction[direction]
        total = term_total_by_direction[direction]
        return {
            "samples": len(group_rows),
            "bleu": bleu.score,
            "bleu_details": bleu.format(),
            "chrf": chrf.score,
            "chrf_details": chrf.format(),
            "required_term_total": total,
            "required_term_hits": hits,
            "required_term_accuracy": hits / total if total else None,
            "empty_predictions": sum(not prediction for prediction in group_predictions),
        }

    by_direction = {
        direction: score_group(
            [row for row in rows if row["direction"] == direction]
        )
        for direction in sorted({row["direction"] for row in rows})
    }
    bleu = sacrebleu.BLEU().corpus_score(predictions, [references])
    chrf = sacrebleu.CHRF().corpus_score(predictions, [references])
    result = {
        "enabled": True,
        "samples": len(rows),
        "predictions_path": str(predictions_path),
        "bleu": bleu.score,
        "bleu_details": bleu.format(),
        "chrf": chrf.score,
        "chrf_details": chrf.format(),
        "required_term_total": term_total,
        "required_term_hits": term_hits,
        "required_term_accuracy": term_hits / term_total if term_total else None,
        "empty_predictions": sum(not prediction for prediction in predictions),
        # BLEU/chrF are not comparable across scripts and tokenization rules.
        # Direction-level metrics are the interpretable primary report.
        "by_direction": by_direction,
    }
    return result


def main() -> int:
    args = parse_args()
    if args.dry_run:
        args.prepare_only = True
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    glossary = None if args.no_glossary else load_glossary(args.glossary)
    examples, data_stats = prepare_translation_examples(
        args.data_dir,
        domains=args.domains,
        directions=args.directions,
        glossary=glossary,
    )
    train_examples = cap_examples(
        [item for item in examples if item.split == "train"],
        args.max_train_samples,
        seed=args.seed,
    )
    validation_examples = cap_examples(
        [item for item in examples if item.split == "validation"],
        args.max_eval_samples,
        seed=args.seed + 1,
    )
    if not train_examples:
        raise RuntimeError("no training examples were loaded")
    if not validation_examples:
        raise RuntimeError("no validation examples were loaded")

    prepared_path = args.output_dir / "prepared_examples.jsonl"
    write_examples(train_examples + validation_examples, prepared_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, train_token_stats = encode_dataset(
        train_examples,
        tokenizer,
        glossary=glossary,
        args=args,
        desc="Tokenizing training set",
    )
    validation_dataset, validation_token_stats = encode_dataset(
        validation_examples,
        tokenizer,
        glossary=glossary,
        args=args,
        desc="Tokenizing validation set",
    )
    data_summary = {
        "data": data_stats,
        "prepared_path": str(prepared_path),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "tokenization": {
            "train": train_token_stats,
            "validation": validation_token_stats,
        },
        "max_length": args.max_length,
    }
    json_write(args.output_dir / "data_summary.json", data_summary)
    if args.prepare_only:
        print(json.dumps(data_summary, ensure_ascii=False, indent=2))
        return 0

    bf16 = resolve_bf16(args.bf16)
    model = load_model(args, bf16)
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_workers,
        group_by_length=True,
        length_column_name="length",
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        save_safetensors=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=TranslationCollator(tokenizer.pad_token_id),
    )
    checkpoint = None
    if args.resume != "never":
        checkpoint = latest_checkpoint(args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    generation_metrics = evaluate_generation(
        trainer.model,
        tokenizer,
        validation_examples,
        glossary,
        args,
        args.output_dir,
    )
    summary = {
        "model": args.model_name_or_path,
        "final_adapter": str(final_dir),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16": bf16,
        },
        "git": git_info(),
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "data": data_summary,
        "train_metrics": train_result.metrics,
        "trainer_state_log_history": trainer.state.log_history,
        "generation_metrics": generation_metrics,
    }
    json_write(args.output_dir / "training_summary.json", summary)
    print(
        json.dumps(
            {
                "final_adapter": str(final_dir),
                "train_metrics": train_result.metrics,
                "generation_metrics": generation_metrics,
                "summary": str(args.output_dir / "training_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
