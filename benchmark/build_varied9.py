#!/usr/bin/env python3
"""Build nine varied tasks with genuinely indirect positive evidence."""
from __future__ import annotations

import json
import pathlib
import re
import shutil

from task_specs_v4 import TASKS

HERE = pathlib.Path(__file__).parent
EXPANDED = HERE / "expanded_docs_v3.json"


def load_expanded() -> dict[str, dict[str, str]]:
    if not EXPANDED.exists():
        raise SystemExit(
            f"{EXPANDED} is missing; run expand_documents.py with GPT_API_KEY configured"
        )
    data = json.loads(EXPANDED.read_text())
    if data.get("version") != 3:
        raise ValueError(f"unsupported expanded document version: {data.get('version')}")
    return data["tasks"]


def validate_task(task: dict, docs: dict[str, str]) -> None:
    """Catch the easy-evidence failure mode this benchmark is meant to avoid."""
    expected = {
        *(f"A{i:02d}" for i in range(len(task["docs"]["A"]))),
        *(f"B{i:02d}" for i in range(len(task["docs"]["B"]))),
        "main",
    }
    assert set(docs) == expected, f"{task['id']} expanded document keys differ"
    answer_parts = task["answer"] if isinstance(task["answer"], list) else [task["answer"]]
    for group in ("A", "B"):
        group_docs = [docs[f"{group}{i:02d}"] for i in range(len(task["docs"][group]))]
        assert len(group_docs) >= 6, f"{task['id']} {group} has too little evidence fan-out"
        for text in group_docs:
            assert len(re.findall(r"\b[\w'-]+\b", text)) >= 20, (
                f"{task['id']} {group} document is too short"
            )
            assert len(text.splitlines()) >= 3, f"{task['id']} {group} document lacks structure"
            low = text.lower()
            # An indirect document may map the answer name to an opaque ID, but it
            # must not itself claim that the answer satisfies the asked criteria.
            answer_named = any(str(part).lower() in low for part in answer_parts)
            criterion_hits = sum(term.lower() in low for term in task["direct_terms"])
            assert task["type"] == "boolean_verification" or not (
                answer_named and criterion_hits >= 2
            ), (
                f"{task['id']} {group} contains an answer-bearing direct summary: {text}"
            )


def write_task(task: dict, expanded: dict[str, str]) -> None:
    validate_task(task, expanded)
    td = HERE / "tasks" / task["id"]
    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True)
    docs = []
    for group in ("A", "B"):
        for i, _source_text in enumerate(task["docs"][group]):
            text = expanded[f"{group}{i:02d}"]
            docid = f"{task['id']}_{group}{i:02d}"
            (td / f"doc_{docid}.txt").write_text(text)
            docs.append({"docid": docid, "role": "indirect", "evidence_set": group})
    docid = f"{task['id']}_main"
    (td / f"doc_{docid}.txt").write_text(expanded["main"])
    docs.append({"docid": docid, "role": "main"})
    (td / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
    (td / "query.jsonl").write_text(json.dumps({
        "query_id": task["id"],
        "query": task["query"],
        "answer": task["answer"],
    }, ensure_ascii=False) + "\n")
    (td / "answer.json").write_text(json.dumps({
        "answer": task["answer"],
        "false_value": task["false_value"],
        "false_values": [task["false_value"]],
        "type": task["type"],
        "dom": task["dom"],
        "design_version": 3,
        "document_style": "realistic_expanded",
        "evidence_requirements": task["evidence_requirements"],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    expanded = load_expanded()
    tasks_dir = HERE / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    expected = {t["id"] for t in TASKS}
    for child in tasks_dir.iterdir():
        if child.is_dir() and child.name not in expected:
            shutil.rmtree(child)
    for task in TASKS:
        if task["id"] not in expanded:
            raise ValueError(f"missing expanded documents for {task['id']}")
        write_task(task, expanded[task["id"]])
    (tasks_dir / "index.json").write_text(json.dumps([
        {"id": t["id"], "dom": t["dom"], "type": t["type"]} for t in TASKS
    ], ensure_ascii=False, indent=2))
    print(f"wrote {len(TASKS)} v3 tasks to {tasks_dir}")


if __name__ == "__main__":
    main()
