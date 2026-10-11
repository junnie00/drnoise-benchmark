#!/usr/bin/env python3
"""Generic concurrent runner for ONE model over ONE domain x {clean,1x} x reps, sharing a dense_server.
Resume-safe. Works for ld_imo (per-task clean) and ld_oscar/ld_book/ld_oly (shared idx_dense_clean).

Usage: run_domain_par.py <domain_dir> <bm25_index> <model> <tag> <maxtok> <port> <conc> <reps>
"""
import os, sys, json, subprocess, pathlib, time
from concurrent.futures import ThreadPoolExecutor
CONDS = tuple(os.environ.get("CONDS","clean,1x").split(","))   # override for graded-noise sweeps
RETRIEVER = os.environ.get("RETRIEVER","dense")                # 'bm25' -> lexical-only (E7 retriever ablation)
NB = pathlib.Path(os.environ.get("DRNOISE_ROOT", pathlib.Path(__file__).resolve().parent))
dom, bm25, model, tag, maxtok, port, conc, reps = (
    pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))
RUNS = dom/"runs"; RUNS.mkdir(exist_ok=True)
tasks = [t["id"] for t in json.loads((dom/"tasks/index.json").read_text())]
only = set(x for x in os.environ.get("TASK_IDS","").split(",") if x)
if only:
    tasks = [t for t in tasks if t in only]
RERUN_COMPLETED = os.environ.get("RERUN_COMPLETED","0")=="1"

def dense_index(tid, cond):
    pertask = dom/"tasks"/tid/f"idx_dense_{cond}"
    if pertask.exists(): return pertask
    if (dom/"tasks"/tid/f"exclude_{cond}.json").exists():
        return dom/"tasks"/tid/"idx_dense_noised"
    return dom/f"idx_dense_{cond}"        # shared clean for oscar/book/oly

def done(tid,c,r,allow_rerun=True):
    if allow_rerun and RERUN_COMPLETED:
        return False
    f = RUNS/f"{tag}_{tid}_{c}_r{r}"/f"{tid}.json"
    if not f.exists(): return False
    try: return json.loads(f.read_text()).get("status")=="completed"
    except Exception: return False

def run(job):
    tid,c,r = job
    if done(tid,c,r): return f"skip {tag}_{tid}_{c}_r{r}"
    out = RUNS/f"{tag}_{tid}_{c}_r{r}"
    if RERUN_COMPLETED and out.exists():
        out.rename(RUNS/f"_rerun_backup_{int(time.time())}_{out.name}")
    if RETRIEVER == "bm25":
        bmi = os.environ.get(f"BM25_{c.upper()}", bm25)    # per-condition BM25 index (clean has no false doc)
        cmd = ["python","-m","noisy_bench.eval_extended.bcp_style_agent",
               "--queries",str(dom/"tasks"/tid/"query.jsonl"),
               "--bm25_index",bmi,"--condition",f"{tag}_{tid}_{c}_r{r}",
               "--output_dir",str(out),"--model",model,"--max_iterations","100","--k","5","--workers","1","--max_tokens",maxtok]
    else:
        cmd = ["python","-m","noisy_bench.eval_extended.bcp_style_agent",
               "--queries",str(dom/"tasks"/tid/"query.jsonl"),
               "--dense_index",str(dense_index(tid,c)),"--bm25_index",bm25,
               "--encode_server",f"http://127.0.0.1:{port}","--condition",f"{tag}_{tid}_{c}_r{r}",
               "--output_dir",str(out),"--model",model,"--max_iterations","100","--k","5","--workers","1","--max_tokens",maxtok]
    _iv = os.environ.get("INSTRUCTION_VARIANT","none")     # E3 ablation; unset/none -> canonical prompt
    if _iv and _iv != "none": cmd += ["--instruction_variant", _iv]
    _gate = os.environ.get("EVIDENCE_GATE", "none")
    if _gate != "none":
        cmd += ["--evidence_gate", _gate]
        obligations = dom/"tasks"/tid/"gate_obligations.json"
        if _gate == "reconciliation" and obligations.exists():
            cmd += ["--gate_obligations", str(obligations)]
        if os.environ.get("GATE_MODEL"):
            cmd += ["--gate_model", os.environ["GATE_MODEL"]]
        cmd += ["--gate_max_rejections", os.environ.get("GATE_MAX_REJECTIONS", "4")]
        cmd += ["--gate_stagnation_limit", os.environ.get("GATE_STAGNATION_LIMIT", "2")]
    exclusions = dom/"tasks"/tid/f"exclude_{c}.json"
    if exclusions.exists():
        cmd += ["--exclude_docids", str(exclusions)]
    with open(RUNS/f"log_{tag}_{tid}_{c}_r{r}.log","w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(NB)).returncode
    if rc != 0:
        return f"fail {tag}_{tid}_{c}_r{r} rc={rc}"
    return f"done {tag}_{tid}_{c}_r{r}"

jobs = [(tid,c,r) for r in range(1,reps+1) for tid in tasks for c in CONDS]
for p in range(1,4):
    todo=[j for j in jobs if not done(*j, allow_rerun=(p==1))]
    print(f"[{tag}] pass{p}: {len(todo)}/{len(jobs)} to run", flush=True)
    if not todo: break
    with ThreadPoolExecutor(conc) as ex:
        for m in ex.map(run, todo): print(m, flush=True)
print(f"[{tag}] {dom.name}_DONE {sum(1 for j in jobs if done(*j, allow_rerun=False))}/{len(jobs)}", flush=True)
