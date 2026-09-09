"""Upload a pilot adapter to ModelScope. Token comes from MS_TOKEN env var.

Usage: MS_TOKEN=ms-... python upload_adapter_modelscope.py opd|grpo
Creates zechlei/tripivot-qwen2.5-7b-lora-{opd,grpo}-r1 (private) and pushes
the adapter + eval metrics + README.
"""
import json
import os
import shutil
import sys

OWNER = "zechlei"
ROOT = os.path.expanduser("~/tripivot-agent")

TALK = {
    "opd": (
        "# tripivot Qwen2.5-7B OPD-R1 (on-policy distilled)\n\n"
        "LoRA adapter distilled on-policy from Qwen2.5-14B-Instruct "
        "(reverse KL + reference NLL), en-zh pilot.\n\n"
        "en-zh held-out: COMET 0.7984 (best), degeneration 0.0, "
        "len_ratio 0.737 — quality-first variant. See repo docs.\n"
    ),
    "grpo": (
        "# tripivot Qwen2.5-7B GRPO-R1 (metric-reward RL)\n\n"
        "LoRA adapter trained with group-relative policy optimization, "
        "chrF/format composite reward, KL-anchored, en-zh pilot.\n\n"
        "en-zh held-out: BLEU +27% over SFT baseline, chrF 0.2622 (best), "
        "stable KL anchor <0.001 — coverage-first variant. See repo docs.\n"
    ),
}


def main():
    variant = sys.argv[1]
    token = os.environ.get("MS_TOKEN")
    assert token, "MS_TOKEN env var required"
    adapter = os.path.join(ROOT, f"runs/{variant}-r1/final_adapter")
    assert os.path.isdir(adapter), adapter

    stage = os.path.join(ROOT, "runs", f"upload_{variant}")
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(adapter, os.path.join(stage, "adapter"))
    eval_src = os.path.join(ROOT, f"runs/eval/{variant}_en-zh.json")
    if os.path.exists(eval_src):
        shutil.copy(eval_src, stage)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(
            "---\nbase_model: Qwen/Qwen2.5-7B-Instruct\nlibrary_name: peft\n"
            "license: apache-2.0\nlanguage: [zh, en]\n---\n" + TALK[variant]
        )

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    repo_id = f"{OWNER}/tripivot-qwen2.5-7b-lora-{variant}-r1"
    try:
        api.create_model(repo_id, visibility=1)
    except Exception as exc:
        print(f"create_model: {exc} (continuing)")
    api.upload_folder(repo_id=repo_id, folder_path=stage,
                      commit_message=f"{variant}-r1 pilot adapter")
    print(f"UPLOAD_{variant.upper()}_DONE {repo_id}")


if __name__ == "__main__":
    main()
