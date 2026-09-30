"""Augmentation-stability solver.

For each task:
  1. run the DSL search on the original view, collecting ALL programs that
     reproduce every training pair
  2. for each augmented view, do the same, map its predictions back into
     original space, and vote
  3. rank test outputs by how many views agree
  4. emit the top 2, matching the competition's 2-attempt rule

Scoring uses the exact Kaggle metric so the number means what the leaderboard
means.

    python solve_aug.py --limit 20 --depth 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from solver.augment import TaskView, key, make_views  # noqa: E402
from solver.prims import dims  # noqa: E402
from solver.search import search  # noqa: E402

DATA = ROOT / "data"


def load(split: str, limit: int | None = None):
    files = sorted((DATA / split).glob("*.json"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        try:
            out.append((f.stem, json.loads(f.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return out


def exact(a, b) -> bool:
    if a is None or b is None:
        return False
    if not a or not b:
        return a == b
    return dims(a) == dims(b) and a == b


def all_programs(task: dict, max_depth: int, budget: float) -> list:
    r = search(task, max_depth=max_depth, time_budget=budget, top_k=64)
    return r.programs


def solve_task(task: dict, max_depth: int, budget: float, views: int) -> dict:
    """Return ranked candidate outputs per test input, with stability votes."""
    n_test = len(task.get("test", []))
    votes: list[Counter] = [Counter() for _ in range(n_test)]
    n_views_used = 0
    progs_found = 0

    for view in make_views(task, n=views, seed=1234):
        progs = all_programs(view.task_dict() if hasattr(view, "task_dict")
                             else {"train": view.train,
                                   "test": view.test},
                             max_depth, budget)
        if not progs:
            continue
        n_views_used += 1
        progs_found += len(progs)
        for i in range(n_test):
            local = Counter()
            for p in progs:
                try:
                    g = p(view.test[i]["input"])
                except Exception:
                    continue
                try:
                    back = view.inv(g)
                except Exception:
                    continue
                if not back:
                    continue
                local[key(back)] += 1
            # weight: a view that found MANY programs but all agree on this
            # answer is strong evidence; a view with 50 ambiguous programs
            # agreeing by chance is weak.
            if local:
                top = local.most_common(1)[0][1]
                total = sum(local.values())
                votes[i].update(
                    {k: c * (c / total) for k, c in local.items()})

    ranked = []
    for i in range(n_test):
        cands = votes[i].most_common(2)
        grids = [list(map(list, k)) for k, _ in cands]
        # pad so we always submit two attempts
        while len(grids) < 2:
            grids.append(task["test"][i]["input"])
        ranked.append(grids[:2])

    return {
        "predictions": ranked,
        "views_used": n_views_used,
        "programs_found": progs_found,
        "any_programs": progs_found > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = load(args.split, args.limit)
    if not tasks:
        print(f"no tasks in {DATA / args.split}")
        return 1

    total = correct = solved = 0
    t0 = time.time()
    per_task = []

    for name, task in tasks:
        res = solve_task(task, args.depth, args.budget, args.views)
        preds = res["predictions"]
        task_ok = True
        for t, cands in zip(task.get("test", []), preds):
            total += 1
            if any(exact(c, t.get("output")) for c in cands):
                correct += 1
            else:
                task_ok = False
        if task_ok and task.get("test"):
            solved += 1
        per_task.append({
            "task": name, "solved": task_ok,
            "programs_found": res["programs_found"],
            "views_used": res["views_used"],
        })

    el = time.time() - t0
    score = correct / total if total else 0.0
    found_any = sum(1 for p in per_task if p["programs_found"] > 0)

    print("=" * 64)
    print("  ARC-AGI-2  augmentation-stability solver")
    print("=" * 64)
    print(f"  tasks                 {len(tasks):>6}")
    print(f"  tasks w/ >=1 program  {found_any:>6}")
    print(f"  OUTPUT SCORE (Kaggle) {score * 100:>6.2f}%   ({correct}/{total})")
    print(f"  task accuracy         {solved / max(len(tasks), 1) * 100:>6.2f}%")
    print(f"  wall clock            {el:>6.1f}s  ({el / max(len(tasks), 1):.2f}s/task)")
    print("-" * 64)
    print("  official reference: human avg 66.00% | leader 83.06%")
    print("  authors' meaningful floor: 5%")
    print("=" * 64)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "score_output_metric": score,
            "tasks": len(tasks), "outputs": total, "correct": correct,
            "solved_tasks": solved, "tasks_with_programs": found_any,
            "elapsed_s": round(el, 1),
            "config": {"depth": args.depth, "views": args.views,
                       "budget": args.budget, "split": args.split},
            "per_task": per_task,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
