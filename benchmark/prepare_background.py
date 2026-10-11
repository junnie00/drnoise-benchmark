"""Reconstruct the ordered historical background from pinned upstream documents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

HERE = Path(__file__).resolve().parent


def canonical_row(docid, text):
    return json.dumps({"docid": str(docid), "text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"


def verify_corpus(path, manifest):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if count >= len(manifest["documents"]):
                raise ValueError("Unexpected extra background document")
            docid, expected = manifest["documents"][count]
            text = row["text"]
            if str(row["docid"]) != docid or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
                raise ValueError(f"Background ID/order/text mismatch at row {count}: {docid}")
            digest.update(canonical_row(docid, text).encode("utf-8"))
            count += 1
    if count != manifest["document_count"] or digest.hexdigest() != manifest["ordered_corpus_sha256"]:
        raise ValueError("Background count or ordered content checksum mismatch")
    return count


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def arrow_rows(directory):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    files = sorted(Path(directory).glob("*.arrow"))
    if not files:
        raise ValueError("No dataset Arrow shards found")
    for path in files:
        with pa.memory_map(str(path), "r") as source:
            for batch in ipc.open_stream(source):
                yield from batch.to_pylist()


def reconstruct(rows, manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; use --verify-only to check it")
    expected = dict(manifest["documents"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="drnoise-background-", dir=output.parent) as stage:
        stage = Path(stage)
        database = sqlite3.connect(stage / "documents.sqlite")
        try:
            database.execute("CREATE TABLE documents (docid TEXT PRIMARY KEY, text TEXT NOT NULL)")
            for row in rows:
                docid = str(row["docid"])
                if docid not in expected:
                    continue
                text = row.get("text", row.get("contents"))
                if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected[docid]:
                    raise ValueError(f"Upstream content differs from the historical background: {docid}")
                database.execute("INSERT INTO documents VALUES (?, ?)", (docid, text))
            database.commit()
            count = database.execute("SELECT count(*) FROM documents").fetchone()[0]
            if count != manifest["document_count"]:
                raise ValueError(f"Missing background documents: found {count}/{manifest['document_count']}")
            staged = stage / "corpus.jsonl"
            with staged.open("w", encoding="utf-8", newline="\n") as target:
                for docid, _ in manifest["documents"]:
                    text = database.execute("SELECT text FROM documents WHERE docid=?", (docid,)).fetchone()[0]
                    target.write(canonical_row(docid, text))
            verify_corpus(staged, manifest)
            staged.replace(output)
        finally:
            database.close()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "background_manifest.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-jsonl", type=Path, help="Existing upstream or background corpus")
    source.add_argument("--arrow-dir", type=Path, help="Existing Hugging Face dataset Arrow cache")
    parser.add_argument("--output", type=Path, default=HERE.parent / "main_method/data/background/corpus.jsonl")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.verify_only:
        print("Verified", verify_corpus(args.output, manifest), "ordered background documents")
        return
    if args.source_jsonl:
        rows = jsonl_rows(args.source_jsonl)
    elif args.arrow_dir:
        rows = arrow_rows(args.arrow_dir)
    else:
        # This standalone process contacts Hugging Face directly.
        for key in list(os.environ):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                os.environ.pop(key)
        from datasets import load_dataset

        print(f"Reading {manifest['dataset']} at revision {manifest['revision']}; upstream is about 3.3 GB of text. Existing HF cache is reused.", flush=True)
        rows = load_dataset(manifest["dataset"], revision=manifest["revision"], split=manifest["split"], streaming=True)
    count = reconstruct(rows, manifest, args.output)
    print("Reconstructed and verified", count, "documents at", args.output)


if __name__ == "__main__":
    main()
