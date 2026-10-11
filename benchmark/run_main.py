"""Launch the reference clean/noisy evaluation without changing task data."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_command(model, tag, port, concurrency, repetitions, task_dir=HERE):
    protocol = json.loads((HERE / "main_protocol.json").read_text())
    profile = protocol["models"][model]
    return [sys.executable, str(ROOT / "main_method/run_domain_par.py"),
            str(task_dir), str(task_dir / "idx_text_varied9"), model, tag,
            str(profile["max_tokens"]), str(port), str(concurrency), str(repetitions)]


def main():
    protocol = json.loads((HERE / "main_protocol.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(protocol["models"]))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=protocol["repetitions"])
    parser.add_argument("--task-ids", default="", help="Comma-separated task IDs; empty means all 100")
    parser.add_argument("--dry-run", action="store_true", help="Print the launch command without any API call")
    args = parser.parse_args()
    if min(args.port, args.concurrency, args.repetitions) < 1:
        parser.error("Port, concurrency, and repetitions must be positive")
    tasks = json.loads((HERE / "tasks/index.json").read_text())
    wanted = set(args.task_ids.split(",")) if args.task_ids else {t["id"] for t in tasks}
    unknown = wanted - {t["id"] for t in tasks}
    if unknown:
        parser.error(f"Unknown task IDs: {sorted(unknown)}")
    tag = args.tag or args.model.replace("-", "")
    command = build_command(args.model, tag, args.port, args.concurrency, args.repetitions)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    missing = [HERE / "idx_text_varied9"]
    missing += [HERE / "tasks" / tid / f"idx_dense_{condition}" for tid in sorted(wanted) for condition in protocol["conditions"]]
    missing = [p for p in missing if not p.exists()]
    if missing:
        parser.error(f"Build retrieval indexes first; missing {len(missing)} paths, including {missing[0]}")
    env = os.environ.copy()
    env.update({"DRNOISE_ROOT": str(ROOT / "main_method"), "CONDS": ",".join(protocol["conditions"]), "RETRIEVER": "dense", "EVIDENCE_GATE": "none", "INSTRUCTION_VARIANT": "none", "TASK_IDS": args.task_ids})
    env_name = {"gpt-5.4": "GPT_BASE_URL", "gemini-3.5-flash": "XCODE_BASE_URL", "mimo-v2.5": "MIMO_BASE_URL", "deepseek-v4-flash": "DEEPSEEK_BASE_URL", "qwen3-32b": "LOCAL_QWEN_URL"}[args.model]
    env.setdefault(env_name, protocol["models"][args.model]["endpoint"])
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
