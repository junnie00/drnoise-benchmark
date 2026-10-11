#!/usr/bin/env python3
"""Use an independent GPT-5.4 pass to audit expanded corpus documents."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import tempfile

from openai import OpenAI

from task_specs_v4 import TASKS

HERE = pathlib.Path(__file__).parent
DOCS = HERE / "expanded_docs_v3.json"
OUT = HERE / "document_audit_v3.json"

SYSTEM = """You are a strict benchmark corpus auditor. Review fictional documents
for factual fidelity and documentary realism. Do not solve the task. Judge each
document only against its canonical record and the supplied question.

For indirect documents:
- Every canonical fact must be preserved.
- Added neutral metadata is allowed, but added facts that could affect the answer
  are not.
- The document must not state the final answer or combine facts from other records.
- An endpoint lookup may contain the correct answer as one value among several
  opaque-key mappings. This is valid indirect evidence, not answer leakage, unless
  that same document also connects the value to the question's identifying
  criteria or says it is the requested conclusion.
- It must not tell a researcher which other record to retrieve next.
- It should resemble a plausible real artifact, not padded AI prose or a benchmark clue.

For the main document:
- It is intentionally false and must clearly assert the supplied false value.
- It must not reveal the correct value or admit that it is fabricated.
- It should resemble a plausible external summary.

Return JSON only."""


def canonical(task: dict) -> dict[str, str]:
    out = {}
    for group in ("A", "B"):
        for i, text in enumerate(task["docs"][group]):
            out[f"{group}{i:02d}"] = text
    out["main"] = task["docs"]["main"]
    return out


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def atomic_write(data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=HERE, delete=False, suffix=".tmp") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = pathlib.Path(f.name)
    tmp.replace(OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--task", action="append", default=[])
    args = ap.parse_args()

    key = os.environ.get("GPT_API_KEY")
    base = os.environ.get("GPT_BASE_URL", "https://xcode.best/v1")
    if not key:
        raise SystemExit("GPT_API_KEY is not set")
    client = OpenAI(api_key=key, base_url=base, timeout=300, max_retries=5)
    expanded = json.loads(DOCS.read_text())["tasks"]
    result = json.loads(OUT.read_text()) if OUT.exists() else {
        "version": 3,
        "model": args.model,
        "tasks": {},
    }
    selected = set(args.task)
    tasks = [task for task in TASKS if not selected or task["id"] in selected]
    for pos, task in enumerate(tasks, 1):
        tid = task["id"]
        if tid in result["tasks"] and not args.force:
            print(f"[{pos}/{len(tasks)}] skip {tid}", flush=True)
            continue
        payload = []
        source = canonical(task)
        for doc_key, text in expanded[tid].items():
            payload.append({
                "key": doc_key,
                "role": "main_false_summary" if doc_key == "main" else "indirect",
                "canonical_record": source[doc_key],
                "expanded_document": text,
            })
        prompt = f"""Audit this complete document set.

Question: {task['query']}
Correct answer: {json.dumps(task['answer'], ensure_ascii=False)}
Intentional false value: {json.dumps(task['false_value'], ensure_ascii=False)}

Documents:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return exactly:
{{
  "documents": [
    {{
      "key": "A00",
      "fact_faithful": true,
      "answer_leak": false,
      "cross_document_hint": false,
      "realism": "pass",
      "issue": ""
    }}
  ],
  "set_realism": "pass",
  "set_issue": ""
}}

Use realism values pass, minor, or fail. Include every key once."""
        print(f"[{pos}/{len(tasks)}] auditing {tid}", flush=True)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8000,
        )
        audit = parse_json(response.choices[0].message.content or "")
        expected = set(source)
        got = {row["key"] for row in audit["documents"]}
        if got != expected:
            raise ValueError(f"{tid}: audit keys differ")
        result["tasks"][tid] = audit
        atomic_write(result)
        severe = []
        for row in audit["documents"]:
            if row["key"] == "main":
                bad = not row["fact_faithful"] or row["realism"] == "fail"
            else:
                bad = (
                    not row["fact_faithful"] or row["answer_leak"]
                    or row["cross_document_hint"] or row["realism"] == "fail"
                )
            if bad:
                severe.append(row["key"])
        print(
            f"[{pos}/{len(tasks)}] {tid}: set={audit['set_realism']} severe={severe}",
            flush=True,
        )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
