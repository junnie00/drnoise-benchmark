# DRNoise benchmark

DRNoise contains 100 frozen tasks across 10 task types. Each task contains a
question, gold answer, designated false answer, two indirect evidence routes
(`A` and `B`), and a direct false-summary document (`main`).

## Repository contents

- `benchmark/tasks/`: the original task questions, labels, documents, and role
  mappings. The released task files are byte-identical to the experiment inputs.
- `benchmark/task_specs_v2.py`, `task_specs_v4.py`, and `build_diverse.py`:
  construction specifications and their required dependencies.
- `benchmark/expanded_docs_v3.json` and `build_varied9.py`: frozen expanded
  document content and deterministic materialization code.
- `benchmark/audit_documents.py`: the document-audit program, without previous
  audit outputs.
- `benchmark/build_varied9_index.py`: clean/noisy index construction.
- `benchmark/background_manifest.json` and `prepare_background.py`: exact
  background document selection, content verification, and ordered reconstruction.
- `benchmark/main_protocol.json` and `run_main.py`: reference evaluation settings
  and a launcher with an API-free dry-run mode.
- `benchmark/grade_varied9.py`: answer extraction and the original grading
  rules. `--rep` selects a repetition explicitly.
- `benchmark/summarize_runs.py`: complete-run checks and multi-run statistics.
- `main_method/`: the BCP-style baseline agent, concurrent runner, dense query
  server, and index utilities required to run the benchmark.

Previous experiment outputs, traces, logs, large retrieval indexes, model
weights, and API credentials are excluded. Optional exploratory SourceCut
algorithms and the separate MiroFlow agent are not included in this release.
The main baseline uses `--evidence_gate none --instruction_variant none`.

## Inspect the dataset

`benchmark/tasks/index.json` lists the tasks. Inside each task directory:

- `query.jsonl` contains the question and an evaluation label.
- `answer.json` contains the gold and designated false answers.
- `docs.json` maps document IDs to roles and evidence routes.
- `doc_<docid>.txt` contains the original document text.

Clean retrieval contains the background corpus and both indirect routes.
Noisy retrieval adds that task's `main` document. Labels and evidence-route
metadata are evaluator inputs; do not expose them to an agent. The supplied
runner passes only `query_id` and `query` to the model.

To inspect task counts without dependencies or model calls:

```bash
python -c 'import json,collections; x=json.load(open("benchmark/tasks/index.json")); print(len(x),collections.Counter(t["type"] for t in x))'
```

## Install and run

Install Python dependencies into your own environment:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e ./main_method
```

Pyserini also requires a compatible Java installation. API access and the
Qwen3-Embedding-8B checkpoint must be provided separately. See
[REPRODUCE.md](REPRODUCE.md) for the background inputs, index construction,
baseline commands, and grading.

The frozen dataset and unit tests can be inspected without downloading the
background corpus or calling a model. The background reconstruction verifies
every selected document and its order against the supplied manifest.
Proprietary model scores can vary with sampling and provider updates.
