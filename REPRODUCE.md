# Running the benchmark

## Background corpus

The background documents come from BrowseComp-Plus. Its upstream repository is
`texttron/BrowseComp-Plus`; its corpus is published as
`Tevatron/browsecomp-plus-corpus` on Hugging Face. Follow the upstream dataset
instructions to acquire the corpus separately. This repository contains the
DRNoise task documents rather than a copy of the large background corpus.

The original index builder reuses a local `full325_scatter` background snapshot
and removes the previous injected documents before adding DRNoise documents.
Its required inputs, relative to `main_method/`, are:

```text
data/full325_scatter/corpus_drown.jsonl
data/full325_scatter/newdocs.jsonl
indexes/qwen3-emb-8b-full325-scatter/corpus.pkl
```

`corpus_drown.jsonl` supplies `{docid, contents}` records. `newdocs.jsonl`
identifies earlier injected documents to remove. `corpus.pkl` stores
`(float32_embeddings, document_ids)` aligned in the same order. The encoder is
Qwen3-Embedding-8B, with EOS pooling, L2 normalization, no passage prefix, and
4096-token passage truncation.

If starting from a corpus containing only background documents, write an empty
`newdocs.jsonl`, place the corpus at the path above, and encode it with
`main_method/noisy_bench/corpus/build_dense_index_qwen3_8b.py` (use `--help` for
its arguments). Record the source dataset revision and SHA-256 of each input.

The exact historical background snapshot is not included. Downloading the
current upstream corpus and rebuilding is a new evaluation environment unless
you independently verify that its documents match the original snapshot. This
release does not claim exact reproduction of historical rankings from the
small task package alone.

## Build indexes

Run from the repository root. Set the embedding model to a locally available
checkpoint or the model identifier:

```bash
export DRNOISE_ROOT="$PWD/main_method"
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export QWEN_TOKENIZER="$EMBEDDING_MODEL"
CUDA_VISIBLE_DEVICES=0 python benchmark/build_varied9_index.py
```

This creates per-task `idx_dense_clean` and `idx_dense_noised` directories, and
the shared `benchmark/idx_text_varied9` text lookup. Dense retrieval uses only
the documents in its condition-specific dense index; the shared BM25 store
provides text for returned IDs.

## Start the query encoder

```bash
CUDA_VISIBLE_DEVICES=0 python main_method/retrieval/dense_server.py \
  --model "$EMBEDDING_MODEL" --host 127.0.0.1 --port 8765
```

## Run the baseline

Configure your provider key and endpoint through environment variables. For
GPT models the runner reads `GPT_API_KEY` and `GPT_BASE_URL`; for MiMo it reads
`MIMO_API_KEY` and `MIMO_BASE_URL`; for DeepSeek it reads `DEEPSEEK_API_KEY`
and `DEEPSEEK_BASE_URL`; for Gemini it reads `XCODE_API_KEY` and
`XCODE_BASE_URL`; local Qwen uses `LOCAL_QWEN_URL` and `LOCAL_QWEN_KEY`.
Use the endpoint that actually serves the requested model. Do not store keys
in this repository.

Example for an already configured GPT provider, run from the repository root:

```bash
export DRNOISE_ROOT="$PWD/main_method"
export CONDS="clean,noised"
export EVIDENCE_GATE="none"
export INSTRUCTION_VARIANT="none"
python main_method/run_domain_par.py \
  "$PWD/benchmark" "$PWD/benchmark/idx_text_varied9" \
  gpt-5.4 baseline 10000 8765 2 3
```

The positional arguments are task directory, text index, model, run tag,
generation-token ceiling per turn, encoder port, concurrency, and repetitions.
This example uses the baseline defaults; match the model-specific generation
ceiling and provider used in your intended comparison. Search returns five
hits, snippets are limited to 512 tokens, and the maximum is 100 iterations.
`TASK_IDS=codename_lookup_0` limits the runner to one task for setup checks.
Model responses may differ across provider versions and sampling runs.

## Grade each repetition

```bash
python benchmark/grade_varied9.py --tag baseline --rep 1 --out results_baseline_r1.json
python benchmark/grade_varied9.py --tag baseline --rep 2 --out results_baseline_r2.json
python benchmark/grade_varied9.py --tag baseline --rep 3 --out results_baseline_r3.json
```

Accuracy is the number of correctly answered tasks divided by 100. Conditional
deference uses each repetition's clean-correct set as its denominator. Summarize
the three percentages with their mean and population standard deviation
(`ddof=0`). Before reporting a full evaluation, check that all 100 clean and
100 noisy outputs exist for each repetition, and report handling of API errors
and incomplete runs.

## Dataset materialization

The released files are the authoritative frozen inputs. `build_varied9.py`
materializes them from the included specifications and expanded document
content; it replaces the task directories next to that script. Run it only in
a disposable copy, especially after creating indexes. No model call is needed
for materialization. `audit_documents.py` performs new model calls and is not
required to inspect or run the frozen dataset.

## Packaging adaptations

The release adjusts local import paths and package dependencies for the
standalone directory layout, removes the unshipped optional SourceCut mode,
and adds explicit repetition selection to the grader. The frozen task files,
baseline prompt, default search settings, and answer-matching rules are
preserved. No previous experimental results are distributed.
