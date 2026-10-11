"""Encode a corpus with Qwen3-Embedding-8B using BCP's recipe, sharded across GPUs.

BCP recipe (from scripts_build_index/qwen3-embed.md):
  - pooling = EOS (last non-pad token's hidden state)
  - L2-normalize the pooled vector
  - passage_max_len = 4096, passage_prefix = "" (no prefix for documents)
  - fp16, batch_size = 32 reference (we use smaller batches for the 8B variant)

Output layout (compatible with BCP's faiss_searcher.py which globs *.pkl):
  <out>/corpus.pkl       -> pickle of (reps: float32 (N, 4096), lookup: list[str])
  <out>/index.faiss      -> faiss.IndexFlatIP for our own retrievability checks
  <out>/docid_list.json  -> sidecar list of docids in row order

Multi-GPU strategy: workers run in parallel processes (one per GPU). Each worker
takes an equal slice of the corpus, encodes it, writes a temp shard, and the main
process merges the shards into a single (reps, lookup) pickle preserving original
row order.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


BCP_PASSAGE_PREFIX = ""  # BCP recipe: empty for documents


def _load_model_on_device(model_path: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    revision = os.environ.get("EMBEDDING_REVISION")
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left", revision=revision)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        revision=revision,
    )
    model.to(device)
    model.eval()
    return tok, model


def _eos_pool(last_hidden, attention_mask):
    """Pool the hidden state at the last non-pad position (BCP/tevatron EOS pooling).
    Because tokenizer uses left-padding, the last position is always the real EOS."""
    return last_hidden[:, -1, :]


def _encode_passages_on_device(
    texts: List[str],
    model_path: str,
    device: str,
    *,
    batch_size: int,
    max_length: int,
    prefix: str,
) -> np.ndarray:
    """Encode a list of texts on one GPU; returns float32 (N, D) IN INPUT ORDER.

    Uses length-sorted batching: tokenize all inputs once to get true lengths,
    sort indices by length ascending, encode in sorted order so each batch pads
    to a small max instead of the global max_length, then un-sort the output
    back to caller's order. Embeddings are mathematically identical to the
    naive impl — only the batching order changes, not the per-document compute.
    """
    import torch
    tok, model = _load_model_on_device(model_path, device)
    if prefix:
        texts = [prefix + t for t in texts]

    # 1) Compute true (truncated) token lengths once.
    lengths = np.empty(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        lengths[i] = len(tok(t, truncation=True, max_length=max_length,
                             add_special_tokens=True)["input_ids"])
    sorted_idx = np.argsort(lengths, kind="stable")  # ascending
    inverse = np.empty_like(sorted_idx)
    inverse[sorted_idx] = np.arange(len(sorted_idx))

    out_sorted = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    pbar = tqdm(range(0, len(texts), batch_size),
                desc=f"[{device}] encode {len(texts)} (len-sorted bs={batch_size})",
                mininterval=2.0)
    for i in pbar:
        idxs = sorted_idx[i:i + batch_size]
        batch = [texts[j] for j in idxs]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.no_grad(), torch.amp.autocast(device.split(":")[0], dtype=torch.float16):
            hidden = model(**enc).last_hidden_state
            pooled = _eos_pool(hidden, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out_sorted[i:i + len(batch)] = pooled.detach().to(torch.float32).cpu().numpy()
    # Un-sort back to caller order so shard rows align with caller's docid list.
    return out_sorted[inverse]


def _worker(args_dict):
    """One process per GPU. Encodes its assigned slice and writes a temp shard."""
    gpu_id = args_dict["gpu_id"]
    rank = args_dict["rank"]
    # Pin this process to its GPU BEFORE importing torch in a worker context.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    assert torch.cuda.is_available(), f"no cuda on worker rank={rank} gpu={gpu_id}"
    device = "cuda:0"  # always 0 after CUDA_VISIBLE_DEVICES restriction

    rows = args_dict["rows"]  # list of {docid, text}
    texts = [r.get("text", r.get("contents")) for r in rows]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Each corpus row needs a string text or contents field")
    reps = _encode_passages_on_device(
        texts, args_dict["model_path"], device,
        batch_size=args_dict["batch_size"],
        max_length=args_dict["max_length"],
        prefix=BCP_PASSAGE_PREFIX,
    )
    out_shard = Path(args_dict["shard_dir"]) / f"shard_{rank:02d}.pkl"
    with out_shard.open("wb") as f:
        pickle.dump((reps, [str(r["docid"]) for r in rows], rank), f)
    return str(out_shard)


def build_index(
    corpus_jsonl: Path,
    output_dir: Path,
    *,
    model_path: str,
    gpu_ids: List[int],
    batch_size: int,
    max_length: int,
):
    import faiss
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in Path(corpus_jsonl).open() if l.strip()]
    n = len(rows)
    n_gpus = len(gpu_ids)
    print(f"Corpus: {n} docs   GPUs: {gpu_ids}   batch={batch_size}   max_len={max_length}",
          flush=True)

    # Split rows into n_gpus contiguous chunks preserving order.
    chunks = []
    base, rem = divmod(n, n_gpus)
    start = 0
    for r in range(n_gpus):
        size = base + (1 if r < rem else 0)
        chunks.append(rows[start:start + size])
        start += size

    from multiprocessing import get_context
    ctx = get_context("spawn")
    args_list = [
        dict(rank=r, gpu_id=gpu_ids[r], rows=chunks[r],
             shard_dir=str(shard_dir), model_path=model_path,
             batch_size=batch_size, max_length=max_length)
        for r in range(n_gpus)
    ]
    t0 = time.time()
    procs = [ctx.Process(target=_worker, args=(a,)) for a in args_list]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exit={p.exitcode}")
    print(f"All {n_gpus} workers done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Merge shards in rank order.
    all_reps = []
    all_docids: List[str] = []
    for r in range(n_gpus):
        with (shard_dir / f"shard_{r:02d}.pkl").open("rb") as f:
            reps, docids, _r = pickle.load(f)
            assert _r == r
        all_reps.append(reps)
        all_docids.extend(docids)
    reps = np.concatenate(all_reps, axis=0).astype("float32")
    assert reps.shape[0] == n == len(all_docids)
    print(f"Merged: reps shape {reps.shape}, dtype {reps.dtype}", flush=True)

    with (output_dir / "corpus.pkl").open("wb") as f:
        pickle.dump((reps, all_docids), f)
    index = faiss.IndexFlatIP(reps.shape[1])
    index.add(reps)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "docid_list.json").open("w") as f:
        json.dump(all_docids, f)
    # Clean up shard files
    for r in range(n_gpus):
        (shard_dir / f"shard_{r:02d}.pkl").unlink(missing_ok=True)
    shard_dir.rmdir()
    print(f"Wrote {output_dir}/corpus.pkl, index.faiss, docid_list.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path,
                    help="JSONL with fields {docid, text}")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="per-GPU batch. 8B + max_len=4096 + L20 46GB fp16 → 16 OK with "
                         "length-sorted batching (most batches don't hit max_len)")
    ap.add_argument("--max_length", type=int, default=4096,
                    help="BCP recipe: 4096 for passages")
    args = ap.parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    build_index(args.corpus, args.output,
                model_path=args.model, gpu_ids=gpu_ids,
                batch_size=args.batch_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
