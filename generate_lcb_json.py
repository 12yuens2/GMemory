"""
Utility script to pull LiveCodeBench data from HuggingFace and write the
JSON files consumed by GMemory's LCB environments.

Usage
-----
# Generate all three static datasets:
python generate_lcb_json.py --scenario all

# Generate individual datasets:
python generate_lcb_json.py --scenario codegen
python generate_lcb_json.py --scenario codeexec
python generate_lcb_json.py --scenario testpred

# Date-filter code generation (reduces dataset size):
python generate_lcb_json.py --scenario codegen --start_date 2024-01-01 --end_date 2024-06-30

# Limit number of problems (useful for quick experiments):
python generate_lcb_json.py --scenario all --limit 100

Output files
------------
data/lcb/codegen.json   — list of code generation problems
data/lcb/codeexec.json  — list of code execution prediction problems
data/lcb/testpred.json  — list of test output prediction problems

Notes
-----
- Private test cases in codegen are decompressed once here so the runtime
  never has to handle compressed data.
- selfrepair.json is NOT generated here; it is derived at run time from the
  results.json written by a completed lcb_codegen run.
"""

import argparse
import json
import os
import pickle
import zlib
import base64
from datetime import datetime

OUTPUT_DIR = "data/lcb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decompress_test_cases(raw) -> list[dict]:
    try:
        tests = json.loads(raw)
    except Exception:
        tests = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        )
    return tests


def _build_input_output(public_tests: list[dict], private_tests: list[dict],
                        fn_name: str | None) -> str:
    all_tests = public_tests + private_tests
    return json.dumps({
        "inputs":  [t["input"]  for t in all_tests],
        "outputs": [t["output"] for t in all_tests],
        "fn_name": fn_name,
    })


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------

def generate_codegen(start_date: str = None, end_date: str = None,
                     limit: int = None, release_version: str = "release_v1") -> None:
    from datasets import load_dataset

    print("Loading code_generation_lite dataset …")
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release_version,
        trust_remote_code=True,
    )

    if start_date:
        p_start = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [r for r in dataset if datetime.fromisoformat(r["contest_date"]) >= p_start]
    if end_date:
        p_end = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [r for r in dataset if datetime.fromisoformat(r["contest_date"]) <= p_end]

    dataset = sorted(dataset, key=lambda r: r["question_id"])
    if limit:
        dataset = dataset[:limit]

    print(f"Processing {len(dataset)} problems …")
    rows = []
    for i, row in enumerate(dataset):
        if i % 50 == 0:
            print(f"  {i}/{len(dataset)}")
        try:
            pub  = json.loads(row["public_test_cases"])
            priv = _decompress_test_cases(row["private_test_cases"])
            meta = json.loads(row["metadata"])
            fn_name = meta.get("func_name")
            rows.append({
                "question_id":      row["question_id"],
                "question_title":   row["question_title"],
                "question_content": row["question_content"],
                "starter_code":     row.get("starter_code", ""),
                "difficulty":       row["difficulty"],
                "platform":         row["platform"],
                "contest_date":     row["contest_date"],
                "input_output":     _build_input_output(pub, priv, fn_name),
            })
        except Exception as e:
            print(f"  WARNING: skipping {row.get('question_id', '?')} — {e}")

    out_path = os.path.join(OUTPUT_DIR, "codegen.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} problems to {out_path}")


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def generate_codeexec(limit: int = None) -> None:
    from datasets import load_dataset

    print("Loading execution-v2 dataset …")
    dataset = load_dataset("livecodebench/execution-v2", split="test")
    dataset = sorted(dataset, key=lambda r: int(r["id"].split("_")[1]))
    if limit:
        dataset = dataset[:limit]

    print(f"Processing {len(dataset)} problems …")
    rows = []
    for row in dataset:
        rows.append({
            "id":              row["id"],
            "question_id":     row["question_id"],
            "function_name":   row["function_name"],
            "code":            row["code"],
            "input":           row["input"],
            "expected_output": row["output"],
        })

    out_path = os.path.join(OUTPUT_DIR, "codeexec.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} problems to {out_path}")


# ---------------------------------------------------------------------------
# Test output prediction
# ---------------------------------------------------------------------------

def generate_testpred(limit: int = None) -> None:
    from datasets import load_dataset

    print("Loading test_generation dataset …")
    dataset = load_dataset("livecodebench/test_generation", split="test")
    dataset = sorted(dataset, key=lambda r: (r["question_id"], r["test_id"]))
    if limit:
        dataset = dataset[:limit]

    print(f"Processing {len(dataset)} problems …")
    rows = []
    for row in dataset:
        try:
            tests = json.loads(row["test"])
            test_input  = tests[0]["input"]
            test_output = tests[0]["output"]
        except Exception:
            test_input  = row.get("test", "")
            test_output = ""
        rows.append({
            "question_id":      row["question_id"],
            "test_id":          row["test_id"],
            "question_title":   row["question_title"],
            "question_content": row["question_content"],
            "starter_code":     row.get("starter_code", ""),
            "function_name":    row.get("function_name", ""),
            "test_input":       test_input,
            "expected_output":  test_output,
        })

    out_path = os.path.join(OUTPUT_DIR, "testpred.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} problems to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate LCB JSON data files for GMemory.")
    parser.add_argument(
        "--scenario", type=str,
        choices=["all", "codegen", "codeexec", "testpred"],
        default="all",
    )
    parser.add_argument("--start_date", type=str, default=None,
                        help="Filter codegen by contest date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=None,
                        help="Filter codegen by contest date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of problems per scenario.")
    parser.add_argument("--release_version", type=str, default="release_v1",
                        help="LCB release version tag.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.scenario in ("all", "codegen"):
        generate_codegen(
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            release_version=args.release_version,
        )
    if args.scenario in ("all", "codeexec"):
        generate_codeexec(limit=args.limit)
    if args.scenario in ("all", "testpred"):
        generate_testpred(limit=args.limit)


if __name__ == "__main__":
    main()
