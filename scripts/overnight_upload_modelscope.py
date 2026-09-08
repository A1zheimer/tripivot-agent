#!/usr/bin/env python3
"""Overnight pipeline: wait for LoRA training to finish, then upload to ModelScope.

Runs detached (nohup) on the HPC2 login node. Watches runs/*/final_adapter for
up to 46 hours, stages the adapter + summary + README, and pushes to
zechlei/tripivot-qwen2.5-7b-lora (created private on first upload).
"""
import datetime
import glob
import os
import shutil
import sys
import time

TOKEN = "ms-7687f981-f258-4d59-9cc8-ae83adb4dd4f"
OWNER = "zechlei"
REPO_NAME = "tripivot-qwen2.5-7b-lora"
ROOT = os.path.expanduser("~/tripivot-agent")


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def find_adapter():
    cands = glob.glob(os.path.join(ROOT, "runs", "*", "final_adapter"))
    return max(cands, key=os.path.getmtime) if cands else None


def main() -> int:
    deadline = time.time() + 46 * 3600
    adapter = None
    log("waiting for runs/*/final_adapter ...")
    while time.time() < deadline:
        adapter = find_adapter()
        if adapter:
            break
        time.sleep(600)
    if not adapter:
        log("TIMEOUT: no final_adapter after 46h")
        return 1
    run_dir = os.path.dirname(adapter)
    log(f"found adapter: {adapter}")

    stage = os.path.join(ROOT, "runs", "upload_stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    shutil.copytree(adapter, os.path.join(stage, "adapter"))
    for name in ("training_summary.json", "all_results.json", "train_results.json"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, stage)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(
            "---\n"
            "base_model: Qwen/Qwen2.5-7B-Instruct\n"
            "library_name: peft\n"
            "language:\n"
            "  - zh\n  - en\n  - my\n"
            "license: apache-2.0\n"
            "---\n"
            "# tripivot Qwen2.5-7B LoRA (my/en/zh translation)\n\n"
            "LoRA adapter trained with the tripivot-agent pipeline on HKUST(GZ) HPC2.\n\n"
            "Directions: zh-en / en-zh / zh-my / my-zh over tech / intl / finance domains\n"
            "(ALT parallel corpus + Wikipedia-distilled domain data, glossary-constrained).\n\n"
            "## Usage\n\n"
            "```python\n"
            "from peft import PeftModel\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "base = AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen2.5-7B-Instruct\", torch_dtype=\"bfloat16\")\n"
            "model = PeftModel.from_pretrained(base, \"adapter\")\n"
            "tok = AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-7B-Instruct\")\n"
            "```\n\n"
            "See adapter/ for weights, training_summary.json for metrics.\n"
        )
    log(f"staged {len(os.listdir(stage))} items in runs/upload_stage")

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(TOKEN)
    repo_id = f"{OWNER}/{REPO_NAME}"
    try:
        api.create_model(repo_id, visibility=1)  # 1 = private
        log(f"created repo {repo_id} (private)")
    except Exception as exc:
        log(f"create_model returned: {exc} (continuing; repo may exist)")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=stage,
            commit_message="LoRA adapter from HPC2 training",
        )
        log("UPLOAD_DONE")
        return 0
    except Exception as exc:
        log(f"UPLOAD_FAILED: {exc!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
