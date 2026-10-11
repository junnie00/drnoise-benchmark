# Reproduce an evaluation

This guide reconstructs the frozen benchmark environment and launches a new
clean/noisy evaluation. Proprietary API responses and aggregate accuracy can
vary with sampling and provider updates; the configuration below is the
reference profile for new evaluations.

## 1. Environment

Use a dedicated Python environment and Java 21. The release-validation package
versions are pinned in `requirements-reproduction.txt`; Python and Java
versions are recorded in `environment.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e ./main_method
python -m unittest discover -s benchmark -p 'test_*.py' -v
```

The frozen task data and unit tests do not need an API key. CUDA and sufficient
GPU memory are required for Qwen3-Embedding-8B encoding. Select an available,
healthy GPU rather than copying a server-specific GPU assignment.

## 2. Reconstruct the background corpus

The source is the public BrowseComp-Plus corpus:

- Dataset: `Tevatron/browsecomp-plus-corpus`
- Revision: `b27b02bc3e45511b8b82a13e6f90ce761df726f6`
- Split: `train`

`benchmark/background_manifest.json` records the required document IDs in their
original dense-index order, a SHA-256 for every unchanged document text, and an
ordered-corpus checksum. The reconstruction script selects those documents
from the pinned source, verifies their text, restores the recorded order, and
refuses mismatches. It does not inject any previous synthetic documents.

```bash
python benchmark/prepare_background.py
```

This reads the upstream corpus separately (approximately 3.3 GB of text) and
writes `main_method/data/background/corpus.jsonl`. The script reuses existing
Hugging Face cache and connects directly to Hugging Face. A disk-backed staging
database keeps memory use bounded; allow disk space for both staging and the
finished corpus.

With an existing JSONL corpus, avoid another download:

```bash
python benchmark/prepare_background.py --source-jsonl /path/to/upstream.jsonl
```

An existing dataset Arrow cache can be read with `--arrow-dir /path/to/cache`.
These alternatives run the same document and order checks. To verify a
previously prepared corpus without rebuilding:

```bash
python benchmark/prepare_background.py --verify-only
```

No large background corpus or model weights are stored in this repository.

## 3. Obtain the embedding checkpoint

The release pins Qwen3-Embedding-8B. Obtain it separately or use a matching local
checkpoint. The selected revision and local checkpoint metadata hashes are in
`benchmark/embedding_manifest.json`.

```bash
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export EMBEDDING_REVISION="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
export QWEN_TOKENIZER="$EMBEDDING_MODEL"
export DRNOISE_ROOT="$PWD/main_method"
```

For an existing checkpoint, set `EMBEDDING_MODEL` to its absolute directory.
The encoder uses FP16, SDPA, left padding, last-token pooling and L2
normalization. Passage encoding uses no prefix and a 4096-token limit; query
encoding uses the included BCP instruction prefix and a 512-token limit.

## 4. Build background and task indexes

```bash
python main_method/noisy_bench/corpus/build_dense_index_qwen3_8b.py \
  --corpus main_method/data/background/corpus.jsonl \
  --output main_method/indexes/background \
  --model "$EMBEDDING_MODEL" --gpus 0 --batch_size 8 --max_length 4096

CUDA_VISIBLE_DEVICES=0 python benchmark/build_varied9_index.py
```

The second command checks the background manifest and dense document-ID
alignment, then creates condition-specific indexes beside each task. Clean
contains the frozen background and that task's A/B documents. Noisy adds only
that task's false-summary document. The shared Lucene store supplies text for
the IDs returned by dense retrieval.

Generated indexes may use hundreds of GB across all task/condition copies.
Check available disk space before this step. `BACKGROUND_CORPUS` and
`BACKGROUND_DENSE_INDEX` can point to prepared inputs stored elsewhere.

## 5. Start the query encoder

In a separate terminal using the same environment and embedding variables:

```bash
CUDA_VISIBLE_DEVICES=0 python main_method/retrieval/dense_server.py \
  --model "$EMBEDDING_MODEL" --host 127.0.0.1 --port 8765
```

## 6. Run the reference evaluation

`benchmark/main_protocol.json` defines the reference run profile:

| Model | Per-turn output ceiling | Temperature | Top-p |
| --- | ---: | ---: | ---: |
| GPT-5.4 | 32768 | 1.0 | 1.0 |
| Gemini 3.5 Flash | 32000 | 1.0 | 1.0 |
| MiMo v2.5 | 16000 | 1.0 | 0.95 |
| DeepSeek V4 Flash | 16000 | 1.0 | 1.0 |
| Qwen3-32B | 16000 | 0.6 | 0.95 |

All reference runs use one `search` tool, top-5 retrieval, 512-token snippets,
a maximum of 100 iterations, the canonical prompt, and no evidence gate.

Set your provider credential in the environment: `GPT_API_KEY`,
`XCODE_API_KEY` (Gemini), `MIMO_API_KEY`, or `DEEPSEEK_API_KEY`. Local Qwen
uses `LOCAL_QWEN_KEY` (defaults to `EMPTY`). The corresponding endpoint
variables are `GPT_BASE_URL`, `XCODE_BASE_URL`, `MIMO_BASE_URL`,
`DEEPSEEK_BASE_URL`, and `LOCAL_QWEN_URL`. The launcher supplies reference
endpoints from the profile unless you override them. Confirm that your
provider serves the requested model; do not store credentials in files here.

Print the command first; this makes no API calls:

```bash
python benchmark/run_main.py --model gpt-5.4 --dry-run
```

Then evaluate:

```bash
python benchmark/run_main.py --model gpt-5.4 --tag gpt54 --concurrency 2
```

This runs 100 tasks in clean and noisy conditions, with three repetitions.
Use `--task-ids codename_lookup_0 --repetitions 1` for a small setup check.
New outputs are written under `benchmark/runs/`, which is excluded from Git.
Only question text is passed to the model; task labels and route metadata are
not included in its prompt.

## 7. Grade and summarize

```bash
python benchmark/summarize_runs.py --tag gpt54 --repetitions 3 \
  --output benchmark/results_gpt54_summary.json
```

The summarizer requires every expected clean/noisy output file, applies the
included answer-matching rules, and reports per-repetition counts followed by
means and population standard deviations (`ddof=0`). Noncompleted runs count
as incorrect. Conditional deference uses each repetition's clean-correct set.

For one repetition's task-level verdicts:

```bash
python benchmark/grade_varied9.py --tag gpt54 --rep 1 \
  --out results_gpt54_r1.json
```

## Dataset materialization

The supplied task files are the authoritative frozen benchmark. To inspect
construction, use the included specifications and expanded documents.
`build_varied9.py` replaces its adjacent task directories: run it only in a
disposable copy, especially after creating indexes. Materialization requires
no model call. `audit_documents.py` performs new model calls and is optional.
