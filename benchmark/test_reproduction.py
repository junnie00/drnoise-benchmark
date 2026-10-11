import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import grade_varied9 as grading
import prepare_background as background
import run_main
import summarize_runs


def fixture_manifest(rows):
    return {"document_count": len(rows),
            "documents": [[str(row["docid"]), hashlib.sha256(row["text"].encode()).hexdigest()] for row in rows],
            "ordered_corpus_sha256": hashlib.sha256("".join(background.canonical_row(row["docid"], row["text"]) for row in rows).encode()).hexdigest()}


class ReproductionTest(unittest.TestCase):
    def test_background_selection_order_and_unicode(self):
        rows = [{"docid": "2", "text": "记录 β\nunchanged"}, {"docid": "1", "text": " first \n"}]
        manifest = fixture_manifest(rows)
        upstream = [{"docid": "9", "text": "unselected"}, rows[1], rows[0]]
        with tempfile.TemporaryDirectory() as stage:
            output = Path(stage) / "background.jsonl"
            background.reconstruct(iter(upstream), manifest, output)
            self.assertEqual(background.verify_corpus(output, manifest), 2)
            self.assertEqual([json.loads(line) for line in output.read_text().splitlines()], rows)
            with self.assertRaises(FileExistsError):
                background.reconstruct(iter(upstream), manifest, output)

    def test_background_content_change_rejected(self):
        rows = [{"docid": "1", "text": "original"}]
        with tempfile.TemporaryDirectory() as stage:
            output = Path(stage) / "background.jsonl"
            with self.assertRaises(ValueError):
                background.reconstruct(iter([{"docid": "1", "contents": "different"}]), fixture_manifest(rows), output)
            self.assertFalse(output.exists())

    def test_missing_background_document_rejected(self):
        rows = [{"docid": "1", "text": "first"}, {"docid": "2", "text": "second"}]
        with tempfile.TemporaryDirectory() as stage:
            with self.assertRaises(ValueError):
                background.reconstruct(iter(rows[:1]), fixture_manifest(rows), Path(stage) / "background.jsonl")

    def test_grading_selects_one_repetition(self):
        tid = "codename_lookup_0"
        answer = grading.ANS[tid]
        with tempfile.TemporaryDirectory() as stage, patch.object(grading, "HERE", Path(stage)):
            for rep, value in [(1, answer["answer"]), (2, answer["false_value"])]:
                path = Path(stage) / "runs" / f"test_tag_{tid}_clean_r{rep}"
                path.mkdir(parents=True)
                (path / f"{tid}.json").write_text(json.dumps({"status": "completed", "result": [{"type": "output_text", "output": "Exact Answer: " + value}]}))
            self.assertEqual(grading.collect("test_tag", 1)[tid, "clean"]["verdict"], "correct")
            self.assertEqual(grading.collect("test_tag", 2)[tid, "clean"]["verdict"], "fooled")
            with self.assertRaises(ValueError):
                grading.collect("test_tag")

    def test_summary_requires_every_cell(self):
        with self.assertRaises(ValueError):
            summarize_runs.metrics({}, ["task"])

    def test_summary_failure_policy_and_conditional_denominator(self):
        cells = {("one", "clean"): {"status": "completed", "verdict": "correct"},
                 ("one", "noised"): {"status": "completed", "verdict": "fooled"},
                 ("two", "clean"): {"status": "incomplete", "verdict": "correct"},
                 ("two", "noised"): {"status": "completed", "verdict": "correct"}}
        result = summarize_runs.metrics(cells, ["one", "two"])
        self.assertEqual(result["clean_accuracy"], 50)
        self.assertEqual(result["noisy_accuracy"], 50)
        self.assertEqual(result["conditional_deference"], 100)
        self.assertEqual(result["failed_or_incomplete_cells"], 1)

    def test_reference_launcher_uses_profile_budget(self):
        protocol = json.loads((Path(__file__).parent / "main_protocol.json").read_text())
        for model, profile in protocol["models"].items():
            command = run_main.build_command(model, "test", 8765, 2, 3)
            self.assertEqual(command[6], str(profile["max_tokens"]))

    def test_launcher_supplies_only_missing_lucene_placeholder(self):
        for existing in (None, "existing-test-key"):
            environment = {} if existing is None else {"OPENAI_API_KEY": existing}
            with patch.dict(os.environ, environment, clear=True), \
                 patch("sys.argv", ["run_main.py", "--model", "gpt-5.4"]), \
                 patch.object(Path, "exists", return_value=True), \
                 patch.object(run_main.subprocess, "run") as launch, \
                 patch("builtins.print"):
                run_main.main()
                self.assertEqual(launch.call_args.kwargs["env"]["OPENAI_API_KEY"], existing or "EMPTY")


if __name__ == "__main__":
    unittest.main()
