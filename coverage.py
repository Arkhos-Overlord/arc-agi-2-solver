"""DSL coverage sweep.

Runs the search across the training set to answer one question with data
instead of opinion: how much of ARC-AGI-2 does a depth-N DSL search actually
cover, and which primitives earn their place?

Outputs a coverage report. This is the evidence for whether to widen the DSL,
go deeper, or abandon the paradigm for TTT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from solver.search import search  # noqa: E402

DATA = ROOT / "data"


def _one(payload):
    name, task, depth, budget = payload
    t0 = time.time()
    try:
        r = search(task, max_depth=depth, time_budget=budget, top_k=8)
        return {
            "task": name,
            "solved": r.solved,
            "programs": [list(p.names) for p in r.programs],
            "tested": r.tested,
            "elapsed": time.time() - t0,
        }
    except Exception as e:  # noqa: BLE001
        return {"task": name, "solved": False, "programs": [],
                "tested": 0, "elapsed": 0.0, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted((DATA / args.split).glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"no tasks in {DATA / args.split}")
        return 1

    payloads = []
    for f in files:
        try:
            payloads.append((f.stem, json.loads(f.read_text(encoding="utf-8")),
                             args.depth, args.budget))
        except json.JSONDecodeError:
            continue

    print(f"coverage sweep: {len(payloads)} tasks, depth<={args.depth}, "
          f"{args.workers} workers")
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, p) for p in payloads]
        for fu in as_completed(futs):
            results.append(fu.result())
            done += 1
            if done % 100 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    solved = [r for r in results if r["solved"]]
    prim_hits = Counter()
    depth_hits = Counter()
    for r in solved:
        for p in r["programs"]:
            for step in p:
                prim_hits[step] += 1
        depth_hits[len(r["programs"][0])] += 1

    errs = [r for r in results if r.get("error")]
    pct = len(solved) / max(len(results), 1) * 100

    print()
    print("=" * 66)
    print("  DSL COVERAGE REPORT")
    print("=" * 66)
    print(f"  tasks swept            {len(results):>8}")
    print(f"  solved by search       {len(solved):>8}   ({pct:.2f}%)")
    print(f"  errors                 {len(errs):>8}")
    print(f"  wall clock             {time.time() - t0:>7.1f}s")
    print("-" * 66)
    if prim_hits:
        print("  primitives that earned a hit:")
        for name, c in prim_hits.most_common(15):
            print(f"    {name:<24} {c:>5}")
    else:
        print("  NO primitive solved any task at this depth.")
        print("  -> the DSL does not express ARC-AGI-2's rules. Widening it")
        print("     is the only way this paradigm moves at all.")
    print("=" * 66)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "split": args.split, "depth": args.depth,
            "tasks": len(results), "solved": len(solved),
            "coverage_pct": pct, "errors": len(errs),
            "elapsed_s": round(time.time() - t0, 1),
            "primitive_hits": dict(prim_hits),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
