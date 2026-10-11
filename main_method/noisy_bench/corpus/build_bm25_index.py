"""Build a pyserini Lucene BM25 index from a JSONL corpus.

Wraps pyserini.index.lucene; thin layer because the BCP `bm25_searcher.py`
expects the standard pyserini format and points at the index dir with --index-path.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_bm25_index(corpus_jsonl: Path, output_dir: Path, *, threads: int = 8) -> None:
    """Reformat to pyserini's expected docs/<file>.json layout and invoke indexer."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as stage:
        stage = Path(stage)
        docs_dir = stage / "docs"
        docs_dir.mkdir()
        # pyserini expects {id, contents} JSON lines
        with (docs_dir / "corpus.json").open("w", encoding="utf-8") as f:
            for line in Path(corpus_jsonl).open("r", encoding="utf-8"):
                row = json.loads(line)
                f.write(json.dumps({
                    "id": str(row["docid"]),
                    "contents": row.get("text") or row.get("contents") or "",
                }) + "\n")

        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(docs_dir),
            "--index", str(output_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    build_bm25_index(args.corpus, args.output, threads=args.threads)


if __name__ == "__main__":
    main()
