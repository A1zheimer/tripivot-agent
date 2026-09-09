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

VARIANTS = ["baseline", "opd", "grpo"]
MODELS_DIR = os.path.join(ROOT, "demo", "models")  # demo/models/{variant}/


class TranslateRequest(BaseModel):
    text: str
    source: str = "en"
    target: str = "zh"
    variants: list[str] = VARIANTS
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
                return item["translations"].get(variant, "")
        return "(缓存中无此句，请切换到 mlx 后端或补充 cache.json)"


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
    ap.add_argument("--backend", choices=["mlx", "cached", "auto"], default="auto")
    args = ap.parse_args()

    backend = None
    mode = args.backend
    if mode in ("mlx", "auto"):
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

    @app.get("/api/health")
    def health():
        return {"backend": mode, "variants": VARIANTS}

    @app.get("/api/presets")
    def presets():
        if mode == "cached" and os.path.exists(CACHE_PATH):
            items = json.load(open(CACHE_PATH)).get("items", [])
            picks = [items[i] for i in (0, len(items) // 3, 2 * len(items) // 3, -1)
                     if 0 <= i < len(items)]
            return {"presets": [{"text": p["source_text"][:180]} for p in picks]}
        return {"presets": [
            {"text": "The new inference engine reduces latency by 40%."},
            {"text": "The two countries agreed to strengthen climate cooperation."},
            {"text": "The central bank raised its policy rate by 25 basis points."},
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
