"""Single-GPU query encoding HTTP server for Qwen3-Embedding-8B.

Loads the model once on a designated GPU and serves L2-normalized query embeddings
over HTTP. Used by the dense-retrieval path in bcp_style_agent.py so all parallel
agent workers share one model instance rather than each loading 16GB on its own.

BCP query recipe (from scripts_build_index/qwen3-embed.md):
  - Prefix: "Instruct: Given a web search query, retrieve relevant passages that
    answer the query\\nQuery:"
  - EOS pool (last position after left-padding)
  - L2 normalize
  - max_length 512 (queries are short; the prefix takes ~30 tokens)
  - fp16

Endpoint:
  POST /encode   { "query": "<str>" }       -> { "vector": [float...], "dim": 4096 }
  POST /encode   { "queries": ["<str>",..] } -> { "vectors": [[float...]], "dim": 4096 }
  GET  /health                              -> { "ok": true, "device": "...", "dim": 4096 }

Run:
  CUDA_VISIBLE_DEVICES=4 python -m noisy_bench.retrieval.dense_server \\
      --model /path/to/Qwen3-Embedding-8B \\
      --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import argparse
import os
import threading
from typing import List

import numpy as np


BCP_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


class _QueryEncoder:
    def __init__(self, model_path: str, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        self.dim = self.model.config.hidden_size
        self.lock = threading.Lock()  # serialize GPU calls; uvicorn workers=1 anyway
        # Warmup: 1 forward pass so the first real call isn't paying CUDA init cost.
        with torch.no_grad():
            enc = self.tok([BCP_QUERY_PREFIX + "warmup"], padding=True,
                           truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            self.model(**enc)

    def encode(self, queries: List[str]) -> np.ndarray:
        import torch
        texts = [BCP_QUERY_PREFIX + q for q in queries]
        with self.lock:
            enc = self.tok(texts, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                hidden = self.model(**enc).last_hidden_state
                pooled = hidden[:, -1, :]                            # EOS (left-pad → last pos)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled.detach().to(torch.float32).cpu().numpy()


from pydantic import BaseModel
from typing import Optional


class EncodeReq(BaseModel):
    query: Optional[str] = None
    queries: Optional[List[str]] = None


def build_app(encoder: _QueryEncoder):
    from fastapi import FastAPI, HTTPException, Body

    app = FastAPI(title="qwen3-emb-8b query server")

    @app.get("/health")
    def health():
        return {"ok": True, "device": "cuda:0", "dim": encoder.dim}

    @app.post("/encode")
    def encode(req: EncodeReq = Body(...)):
        if req.query is None and not req.queries:
            raise HTTPException(400, "must provide 'query' or 'queries'")
        if req.queries:
            vecs = encoder.encode(req.queries)
            return {"vectors": vecs.tolist(), "dim": encoder.dim}
        vec = encoder.encode([req.query])[0]
        return {"vector": vec.tolist(), "dim": encoder.dim}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    # CUDA_VISIBLE_DEVICES is expected to be set by the launcher (e.g., =4)
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), (
        "set CUDA_VISIBLE_DEVICES before launching this server (e.g., =4)")

    print(f"Loading {args.model} on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
          flush=True)
    enc = _QueryEncoder(args.model, max_length=args.max_length)
    print(f"Ready. dim={enc.dim}. Listening on http://{args.host}:{args.port}", flush=True)

    import uvicorn
    uvicorn.run(build_app(enc), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
