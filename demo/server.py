"""Demo server for the tripivot translation model (defense-safe).

Backends (auto-selected):
  mlx    — local Apple Silicon inference via mlx-lm, three merged variants
  cached — precomputed results from demo/cache.json (offline fallback)

Run:  python demo/server.py --port 7860
      (mode forced with --backend mlx|cached)
"""
import argparse
import json
import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "demo", "cache.json")
STATIC = os.path.join(ROOT, "demo", "static")

VARIANTS = ["baseline", "opd", "grpo", "final"]
MODELS_DIR = os.path.join(ROOT, "demo", "models")  # demo/models/{variant}/


class TranslateRequest(BaseModel):
    text: str
    source: str = "en"
    target: str = "zh"
    variants: list[str] = ["final"]
    glossary_terms: list[tuple[str, str]] = []


class MlxBackend:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _ensure(self, variant: str):
        """Lazy-load one variant at a time (fits smaller unified-memory Macs)."""
        if variant in self.models:
            return
        from mlx_lm import load

        path = os.path.join(MODELS_DIR, variant)
        if not os.path.isdir(path):
            raise KeyError(variant)
        if self.models:  # keep at most one resident model
            self.models.clear()
            self.tokenizers.clear()
        m, tk = load(path)
        self.models[variant] = m
        self.tokenizers[variant] = tk

    def variants_available(self):
        return [v for v in VARIANTS if os.path.isdir(os.path.join(MODELS_DIR, v))]

    def generate(self, prompt: str, variant: str, max_tokens: int = 768) -> str:
        from mlx_lm import generate

        self._ensure(variant)
        out = generate(self.models[variant], self.tokenizers[variant],
                       prompt=prompt, max_tokens=max_tokens, verbose=False)
        return out.strip()


class ApiBackend:
    """OpenAI-compatible API backend for the live agent-loop demo.

    Columns: baseline = plain one-shot translation;
             opd      = glossary-injected first pass (agent machinery);
             grpo     = audited retry pass (feedback -> revision).
    Configure: OPENAI_API_KEY / OPENAI_BASE_URL / API_MODEL env vars.
    """

    def __init__(self):
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.model = os.environ.get("API_MODEL", "glm-4.7")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def _chat(self, system: str, user: str) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.1,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"].strip()

    def generate(self, prompt: str, variant: str, max_tokens: int = 768,
                 text: str = "", source: str = "en", target: str = "zh") -> str:
        sysmsg = (f"You are a professional translator from {source} to {target}. "
                  "Return only the translation.")
        if variant == "baseline":
            return self._chat(sysmsg, text)
        if variant == "opd":  # glossary-injected agent pass
            terms = "\n".join(f"- pipeline parallelism -> 流水线并行\n"
                              "- tensor parallelism -> 张量并行\n"
                              "- KV cache -> KV 缓存\n"
                              "- continuous batching -> 连续批处理\n"
                              "- quantization -> 量化")
            return self._chat(sysmsg, f"术语表（必须遵守）：\n{terms}\n\n{text}")
        # grpo column = audit + retry
        first = self._chat(sysmsg, text)
        retry = self._chat(
            sysmsg,
            f"Translate from {source} to {target}.\n\nOriginal:\n{text}\n\n"
            f"Draft:\n{first}\n\nReviewer feedback: check terminology "
            f"consistency and completeness; fix any issue.\n\n"
            f"Output the corrected translation only.")
        return retry


DOMAIN_KEYWORDS = {
    "tech": ["computer", "software", "network", "apple", "internet", "system",
             "ai ", "data", "digital", "算法", "系统", "软件", "网络", "苹果", "数据"],
    "finance": ["bank", "stock", "market", "loan", "fiscal", "trade", "invest",
                "银行", "股", "市场", "贷款", "财政", "投资", "经济"],
    "intl": ["president", "election", "country", "united nations", "government",
             "policy", "minister", "united states", "washington", "diplomat",
             "总统", "选举", "联合国", "政府", "外交", "部长", "美国", "华盛顿"],
}


def guess_domain(text: str) -> str:
    t = text.lower()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if any(k in t for k in kws):
            return dom
    return "general"


class CachedBackend:
    def __init__(self):
        with open(CACHE_PATH) as f:
            self.cache = json.load(f)
        self.items = [it for it in self.cache.get("items", []) if it.get("source_text")]

    def generate(self, prompt_text: str, variant: str, **kw) -> str:
        q = prompt_text.strip()[:180]
        for item in self.items:
            src = item["source_text"].strip()
            if src.startswith(q) or q.startswith(src[:180]):
                if variant == "final":
                    return item.get("final") or item["translations"].get("grpo", "")
                return item["translations"].get(variant, "")
        return "(缓存中无此句，请使用示例或上传文档中的段落)"


def build_prompt(req: TranslateRequest) -> str:
    lines = [
        f"Translate from {req.source} to {req.target}.",
        "Preserve meaning, numbers, named entities, and formatting. "
        "Return only the translation.",
    ]
    if req.glossary_terms:
        pairs = "; ".join(f"{s}→{t}" for s, t in req.glossary_terms)
        lines.append(f"Glossary: {pairs}")
    # minimal chat template (matches Qwen2.5 instruct format)
    return (f"<|im_start|>user\n{chr(10).join(lines)}\n\n{req.text}"
            f"<|im_end|>\n<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--backend", choices=["mlx", "cached", "api", "auto"], default="auto")
    args = ap.parse_args()

    backend = None
    mode = args.backend
    if mode == "api":
        try:
            backend = ApiBackend()
        except Exception as exc:
            print(f"[demo] api backend unavailable ({exc}); falling back to cache")
            backend = CachedBackend()
            mode = "cached"
    elif mode in ("mlx", "auto"):
        cand = MlxBackend()
        if cand.variants_available():
            backend = cand
            mode = "mlx"
        else:
            print("[demo] no models under demo/models/; falling back to cache")
            backend = CachedBackend()
            mode = "cached"
    else:
        backend = CachedBackend()

    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    LABELS = {
        "cached": ["SFT 基线", "OPD（教师蒸馏）", "GRPO（指标奖励 RL）"],
        "mlx": ["SFT 基线", "OPD（教师蒸馏）", "GRPO（指标奖励 RL）"],
        "api": ["直译（单轮）", "术语注入（Agent 第一轮）", "审计重译（Agent 第二轮）"],
    }

    @app.get("/api/health")
    def health():
        return {"backend": mode, "variants": VARIANTS,
                "labels": LABELS.get(mode, LABELS["cached"])}

    @app.get("/api/presets")
    def presets():
        """Human-vetted examples first; heuristic fallback if file missing."""
        curated = os.path.join(ROOT, "demo", "presets_curated.json")
        if os.path.exists(curated):
            items = json.load(open(curated)).get("presets", [])
            if items:
                return {"presets": [
                    {"text": it["source_text"][:1200],
                     "source": it["source_language"], "target": it["target_language"],
                     "domain": it.get("domain", "general")}
                    for it in items
                ]}
        pairs = [("en", "zh"), ("zh", "en"), ("zh", "my"), ("my", "zh")]
        by_pair = {}
        if os.path.exists(CACHE_PATH):
            for it in json.load(open(CACHE_PATH)).get("items", []):
                key = (it.get("source_language", "en"), it.get("target_language", "zh"))
                if key not in pairs or key in by_pair:
                    continue
                src, fin = it["source_text"], it.get("final", "")
                if not fin or not (300 <= len(src) <= 1200):
                    continue
                ratio = len(fin) / max(1, len(src))
                if not (0.5 <= ratio <= 1.6) or len(fin) < 150:
                    continue
                dom = it.get("domain") or guess_domain(src)
                entry = {**it, "domain": dom}
                if key not in by_pair:
                    by_pair[key] = entry
                elif by_pair[key]["domain"] == "general" and dom != "general":
                    by_pair[key] = entry  # upgrade generic news to a domain item
        return {"presets": [
            {"text": by_pair[k]["source_text"][:1200], "source": k[0], "target": k[1],
             "domain": by_pair[k].get("domain", "general")}
            for k in pairs if k in by_pair
        ]}

    @app.post("/api/translate")
    def translate(req: TranslateRequest):
        prompt = build_prompt(req)
        results = {}
        latencies = {}
        for v in req.variants:
            t0 = time.time()
            try:
                if mode == "mlx" and v in backend.models:
                    results[v] = backend.generate(prompt, v)
                elif mode == "mlx":
                    results[v] = "(该变体未部署，运行 merge 脚本生成)"
                elif mode == "api":
                    results[v] = backend.generate(
                        prompt, v, text=req.text,
                        source=req.source, target=req.target)
                else:
                    results[v] = backend.generate(req.text, v)
            except Exception as exc:  # noqa: BLE001
                results[v] = f"(出错: {exc})"
            latencies[v] = round(time.time() - t0, 2)
        return {"results": results, "latencies": latencies, "backend": mode}

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
