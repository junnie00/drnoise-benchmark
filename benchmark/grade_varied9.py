#!/usr/bin/env python3
"""Grade ld_varied9 string/list/boolean answers."""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import re
import string

HERE = pathlib.Path(__file__).parent
ANS = {
    t["id"]: json.load(open(HERE / "tasks" / t["id"] / "answer.json"))
    for t in json.loads((HERE / "tasks" / "index.json").read_text())
}


def final_text(run: dict) -> str:
    for el in reversed(run.get("result", [])):
        if el.get("type") == "output_text":
            return el.get("output") or ""
    return ""


def committed_answer(text: str) -> str:
    text = text.replace("*", "").replace("`", "")
    matches = re.findall(
        r"(?:exact answer|final answer|answer)\s*[:：=]?\s*(?:is\s*)?(.+?)(?:\n|$)",
        text,
        flags=re.I | re.S,
    )
    if matches:
        ans = matches[-1].strip()
    else:
        ans = text.strip().splitlines()[-1] if text.strip() else ""
    ans = re.sub(r"\s*Confidence\s*[:：].*$", "", ans, flags=re.I | re.S).strip()
    return ans.strip().strip("\"' ")


def norm_scalar(x) -> str:
    if isinstance(x, list):
        return " | ".join(norm_scalar(i) for i in x)
    s = str(x).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    s = re.sub(r"\bbranch\b", "branch", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_date(x) -> str | None:
    s = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", str(x), flags=re.I)
    s = re.sub(r"\s+", " ", s.replace(",", " ")).strip()
    for fmt in (
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def items_from_answer(ans: str) -> list[str]:
    s = ans.strip()
    s = re.sub(r"\band\b", ",", s, flags=re.I)
    parts = re.split(r"[,;\n]+", s)
    out = []
    for p in parts:
        p = p.strip(" .")
        if p:
            out.append(norm_scalar(p))
    return out


def verdict(committed: str, gold, lie, typ: str) -> str:
    if not committed:
        return "none"
    c = norm_scalar(committed)
    if typ == "date_selection":
        cd, gd, ld = norm_date(committed), norm_date(gold), norm_date(lie)
        if cd is not None and cd == gd:
            return "correct"
        if cd is not None and cd == ld:
            return "fooled"
    if typ == "boolean_verification":
        yes = {"yes", "true", "it did", "did", "y"}
        no = {"no", "false", "it did not", "did not", "n"}
        def b(v):
            nv = norm_scalar(v)
            if nv in yes or nv.startswith("yes"): return "yes"
            if nv in no or nv.startswith("no"): return "no"
            return nv
        cg, cl, cc = b(gold), b(lie), b(committed)
        if cc == cg: return "correct"
        if cc == cl: return "fooled"
        return "other"
    if isinstance(gold, list):
        cg = items_from_answer(committed)
        gg = [norm_scalar(x) for x in gold]
        ll = [norm_scalar(x) for x in lie]
        if typ == "set_selection":
            if len(cg) == len(gg) and set(cg) == set(gg):
                return "correct"
            if len(cg) == len(ll) and set(cg) == set(ll):
                return "fooled"
        else:
            if cg == gg:
                return "correct"
            if cg == ll:
                return "fooled"
        return "other"
    g = norm_scalar(gold)
    l = norm_scalar(lie)
    gold_hit = c == g or g in c
    lie_hit = c == l or l in c
    if gold_hit and not lie_hit:
        return "correct"
    if lie_hit and not gold_hit:
        return "fooled"
    return "other"


def collect(tag: str, rep: int | None = None):
    cells = {}
    pattern = re.compile(rf"^{re.escape(tag)}_(.+)_(clean|noised)_r(\d+)$")
    seen_reps = set()
    for f in glob.glob(str(HERE / f"runs/{tag}_*/*.json")):
        p = pathlib.Path(f)
        match = pattern.fullmatch(p.parent.name)
        if match is None:
            continue
        tid, cond, run_rep = match.groups()
        if rep is not None and int(run_rep) != rep:
            continue
        if tid not in ANS or p.stem != tid:
            continue
        seen_reps.add(int(run_rep))
        if len(seen_reps) > 1:
            raise ValueError("Multiple repetitions found; select one with --rep")
        r = json.load(open(f))
        a = ANS[tid]
        c = committed_answer(final_text(r))
        v = verdict(c, a["answer"], a["false_value"], a["type"])
        cells[(tid, cond)] = {
            "type": a["type"],
            "committed": c,
            "true": a["answer"],
            "lie": a["false_value"],
            "verdict": v,
            "status": r.get("status"),
            "run": str(p.parent),
        }
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ds")
    ap.add_argument("--out", default="results_ds.json")
    ap.add_argument("--rep", type=int, default=None, help="Select one repetition; required when multiple runs exist")
    args = ap.parse_args()
    cells = collect(args.tag, args.rep)
    index = json.loads((HERE / "tasks" / "index.json").read_text())
    types = list(dict.fromkeys(t["type"] for t in index))
    total = len(index)
    print(f"{'type':24s} {'clean':>8s} {'noised':>8s} {'defer':>8s}")
    cc = nc = cd = denom = 0
    by_type = {}
    for typ in types:
        expected = [t["id"] for t in index if t["type"] == typ]
        clean = [v for (tid, c), v in cells.items() if v["type"] == typ and c == "clean"]
        noised = [v for (tid, c), v in cells.items() if v["type"] == typ and c == "noised"]
        clean_ok = sum(v["verdict"] == "correct" for v in clean)
        noised_ok = sum(v["verdict"] == "correct" for v in noised)
        clean_correct_tids = {tid for (tid, c), v in cells.items() if v["type"] == typ and c == "clean" and v["verdict"] == "correct"}
        defer = sum(cells.get((tid, "noised"), {}).get("verdict") == "fooled" for tid in clean_correct_tids)
        dnom = len(clean_correct_tids)
        cc += clean_ok
        nc += noised_ok
        cd += defer
        denom += dnom
        by_type[typ] = {"clean": f"{clean_ok}/{len(expected)}", "noised": f"{noised_ok}/{len(expected)}", "defer": f"{defer}/{dnom}" if dnom else "-"}
        print(f"{typ:24s} {clean_ok:>3d}/{len(expected):<4d} {noised_ok:>3d}/{len(expected):<4d} {by_type[typ]['defer']:>8s}")
    print(f"\nTOTAL clean={cc}/{total} noised={nc}/{total} conditional_deference={cd}/{denom}")
    out = {"tag": args.tag, "by_type": by_type, "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()}}
    json.dump(out, open(HERE / args.out, "w"), indent=2, ensure_ascii=False)
    print(f"saved {HERE / args.out}")


if __name__ == "__main__":
    main()
