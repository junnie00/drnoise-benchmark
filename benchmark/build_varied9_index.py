#!/usr/bin/env python3
"""Build clean/noised dense indexes and shared BM25 lookup for ld_varied9."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NB = pathlib.Path(os.environ.get("DRNOISE_ROOT", HERE.parent / "main_method"))
sys.path.insert(0, str(NB))

IDX = json.loads((HERE / "tasks" / "index.json").read_text())
CONDS = [c for c in os.environ.get("CONDS", "clean,noised").split(",") if c]
FORCE = os.environ.get("FORCE_REBUILD", "0") == "1"
SKIP_DENSE = os.environ.get("SKIP_DENSE", "0") == "1"


def neut(tid: str, docid: str) -> str:
    return f"{tid}_{hashlib.md5(docid.encode()).hexdigest()[:8]}"


def task_docs(tid: str):
    out = []
    for r in json.loads((HERE / "tasks" / tid / "docs.json").read_text()):
        fp = HERE / "tasks" / tid / f"doc_{r['docid']}.txt"
        if fp.exists() and fp.read_text().strip():
            out.append((r["docid"], neut(tid, r["docid"]), r["role"], fp.read_text().strip()))
    return out


def cond_docs(tid: str, cond: str):
    docs = task_docs(tid)
    base = [(nd, txt) for _od, nd, role, txt in docs if role == "indirect"]
    main = [(nd, txt) for _od, nd, role, txt in docs if role == "main"]
    if cond == "clean":
        return base
    if cond == "noised":
        return base + main
    raise ValueError(cond)


def main() -> None:
    from noisy_bench.corpus.build_bm25_index import build_bm25_index

    from prepare_background import verify_corpus

    existing = pathlib.Path(os.environ.get("BACKGROUND_DENSE_INDEX", NB / "indexes/background/corpus.pkl"))
    drown = pathlib.Path(os.environ.get("BACKGROUND_CORPUS", NB / "data/background/corpus.jsonl"))
    manifest = json.loads((HERE / "background_manifest.json").read_text())
    verify_corpus(drown, manifest)
    model = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

    reps_all, docids_all = pickle.load(open(existing, "rb"))
    reps_all = np.asarray(reps_all, "float32")
    bg_docids = [str(d) for d in docids_all]
    if bg_docids != [d[0] for d in manifest["documents"]]:
        raise ValueError("Background dense-index IDs/order differ from the frozen manifest")
    bg_reps = reps_all
    if len(bg_reps) != len(bg_docids):
        raise ValueError("Background embeddings and document IDs are misaligned")

    every = []
    for t in IDX:
        every += [(nd, txt) for _od, nd, _role, txt in task_docs(t["id"])]
    print(f"bg {len(bg_docids)} | tasks {len(IDX)} | docs {len(every)} | conds {CONDS}")

    if not SKIP_DENSE:
        from noisy_bench.corpus.build_dense_index_qwen3_8b import _encode_passages_on_device, BCP_PASSAGE_PREFIX

        reps = _encode_passages_on_device(
            [x[1] for x in every],
            model,
            "cuda:0",
            batch_size=8,
            max_length=4096,
            prefix=BCP_PASSAGE_PREFIX,
        )
        rep_of = {d: v for (d, _txt), v in zip(every, reps)}

        for t in IDX:
            tid = t["id"]
            td = HERE / "tasks" / tid
            for cond in CONDS:
                od = td / f"idx_dense_{cond}"
                if (od / "corpus.pkl").exists() and not FORCE:
                    continue
                docs = cond_docs(tid, cond)
                add = np.stack([rep_of[d] for d, _txt in docs]).astype("float32")
                od.mkdir(parents=True, exist_ok=True)
                pickle.dump((np.vstack([bg_reps, add]), bg_docids + [d for d, _txt in docs]), open(od / "corpus.pkl", "wb"))

    tmp = HERE / "_bm25_docs.jsonl"
    with tmp.open("w") as w:
        for l in drown.open():
            l = l.strip()
            if l:
                w.write(l + "\n")
        for d, txt in every:
            w.write(json.dumps({"docid": d, "contents": txt}, ensure_ascii=False) + "\n")
    build_bm25_index(tmp, HERE / "idx_text_varied9", threads=8)
    tmp.unlink()
    print("built varied9 indexes")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
