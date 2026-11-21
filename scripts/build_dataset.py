"""Build `src/kin_citation/data/kin_citation.jsonl` from the study's released materials.

Run from this repository with the study repository checked out beside it, using the study's
own environment (its `requirements.txt`), since this script imports the study code:

    python scripts/build_dataset.py --study-root ../kin-citation-study

It makes no API calls. It reuses the study's own selection of topics (`study.design`), its
counterbalanced orders and its style-swap pools (`study.swap`), checks that the prompts in
`src/kin_citation/prompts.py` match `study/select.py`, and writes one JSON line per topic.
The output is deterministic: running it twice gives the same bytes.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "src" / "kin_citation" / "data" / "kin_citation.jsonl"
MAIN_SHIFTS = [0, 1, 2, 3, 4, 5]
SWAP_SHIFTS = [0, 2, 4]


def load_prompts():
    spec = importlib.util.spec_from_file_location(
        "kin_prompts", HERE / "src" / "kin_citation" / "prompts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_by_key(records):
    """One record per trial key, the last one written, as `study/analyze.py` does."""
    out = {}
    for r in records:
        out[r["key"]] = r
    return list(out.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    root = args.study_root.resolve()
    sys.path.insert(0, str(root))

    from study import select, swap
    from study.config import MODELS, RAW
    from study.design import AUTHORS, SOURCES, SWAP_TOPICS, rotations, topics_in_study
    from study.store import read_jsonl

    prompts = load_prompts()
    assert prompts.SYSTEM_PROMPT == select.SYSTEM, "system prompt differs from study/select.py"
    assert prompts.USER_PROMPT_TEMPLATE == select.ANSWER, "user prompt differs from study/select.py"
    assert prompts.CITATION_SCHEMA == select.CITE_SCHEMA, "schema differs from study/select.py"
    assert list(MODELS) == AUTHORS == SOURCES[:5]

    topics = topics_in_study()
    assert len(topics) == 80, len(topics)
    in_study = {t["pageid"] for t in topics}

    main_recs = [
        r for r in latest_by_key(read_jsonl(RAW / "select_main.jsonl")) if r["pageid"] in in_study
    ]
    main_ref: dict = {}
    for r in main_recs:
        main_ref.setdefault((r["pageid"], r["shift"]), {})[r["selector"]] = r

    swap_recs = latest_by_key(read_jsonl(RAW / "select_swap.jsonl"))
    swap_ref = {(r["pageid"], r["selector"], r["shift"]): r for r in swap_recs}
    swap_jobs = swap.swap_pools(topics[:SWAP_TOPICS])  # (selector, topic, shift, order, pages)

    rows = []
    for t in topics:
        pid = t["pageid"]
        orders = {}
        reference = {}
        for k, order in rotations(pid, SOURCES, MAIN_SHIFTS):
            recs = main_ref[(pid, k)]
            assert sorted(recs) == sorted(AUTHORS), (pid, k)
            for rec in recs.values():
                assert rec["order"] == order, "recorded order differs from study.design.rotations"
            orders[str(k)] = order
            reference[str(k)] = {
                s: (recs[s]["cited"] if recs[s]["valid"] else None) for s in AUTHORS
            }
        row = {
            "pageid": pid,
            "title": t["title"],
            "revid": t["revid"],
            "question": t["question"],
            "pages": {src: t["pages"][src] for src in SOURCES},
            "orders": orders,
            "reference": reference,
            "style_swap": {},
        }
        for sel, topic, k, order, pages in swap_jobs:
            if topic["pageid"] != pid:
                continue
            entry = row["style_swap"].setdefault(
                sel,
                # Only the two rewritten slots differ from the main-arm pool, so only they are stored.
                {
                    "partner": swap.partner(sel),
                    "rewrites": {a: pages[a] for a in (sel, swap.partner(sel))},
                    "orders": {},
                    "reference": {},
                },
            )
            assert all(pages[a] == t["pages"][a] for a in SOURCES if a not in entry["rewrites"])
            rec = swap_ref.get((pid, sel, k))
            assert rec is not None, (pid, sel, k)
            assert rec["order"] == order, (pid, sel, k)
            entry["orders"][str(k)] = order
            entry["reference"][str(k)] = rec["cited"] if rec["valid"] else None
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    args.out.write_text(text, encoding="utf-8")
    n_swap = sum(len(e["orders"]) for r in rows for e in r["style_swap"].values())
    print(f"{len(rows)} topics, {6 * len(rows)} main samples, {n_swap} style-swap samples")
    print(f"sha256 {hashlib.sha256(text.encode()).hexdigest()}  {args.out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
