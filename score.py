"""Local evaluation harness for ARC-AGI-2.

Scores the solver against the 120 public evaluation tasks using the EXACT
Kaggle metric, so the number here means the same thing as the leaderboard:

  "any of the 2 predicted outputs matches the ground truth exactly, you score 1
   for that task test output, otherwise 0. The final score is the sum averaged
   of the highest score per task output divided by the total number of task
   test outputs."

Usage:
    python score.py                 # depth-2 search, 8s/task
    python score.py --depth 2 --budget 8 --limit 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver.prims import dims  # noqa: E402
from solver.search import predict, search  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(split: str, limit: int | None = None) -> list[tuple[str, dict]]:
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
    """Grids match exactly - values AND dimensions."""
    if a is None or b is None:
        return False
    if not a or not b:
        return a == b
    return dims(a) == dims(b) and a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = load(args.split, args.limit)
    if not tasks:
        print(f"no tasks found in {DATA / args.split}")
        return 1

    total_outputs = 0
    correct = 0
    solved_tasks = 0
    attempted = 0
    t_start = time.time()
    per_task = []
    failures = []

    for name, task in tasks:
        res = search(task, max_depth=args.depth,
                     time_budget=args.budget, top_k=args.attempts)
        preds = predict(task, res.programs, attempts=args.attempts)
        attempted += 1
        task_ok = True
        for t, cands in zip(task.get("test", []), preds):
            total_outputs += 1
            truth = t.get("output")
            if any(exact(c, truth) for c in cands):
                correct += 1
            else:
                task_ok = False
        if task_ok and task.get("test"):
            solved_tasks += 1
        else:
            failures.append({"task": name, "tested": res.tested,
                             "reason": res.stopped_reason,
                             "elapsed": round(res.elapsed, 2)})
        per_task.append({
            "task": name, "solved": bool(res.solved), "programs": res.programs,
            "tested": res.tested, "elapsed": round(res.elapsed, 2),
        })

    elapsed = time.time() - t_start
    score = correct / total_outputs if total_outputs else 0.0
    task_acc = solved_tasks / attempted if attempted else 0.0

    print("=" * 62)
    print(f"  ARC-AGI-2 local eval  [{args.split}]")
    print("=" * 62)
    print(f"  tasks attempted     {attempted:>8}")
    print(f"  task outputs        {total_outputs:>8}")
    print(f"  OUTPUT SCORE (Kaggle metric) {score * 100:>7.2f}%   "
          f"({correct}/{total_outputs})")
    print(f"  task-level accuracy {task_acc * 100:>7.2f}%   "
          f"({solved_tasks}/{attempted})")
    print(f"  programs found      "
          f"{sum(1 for p in per_task if p['solved']):>8}")
    print(f"  wall clock          {elapsed:>7.1f}s "
          f"({elapsed / max(attempted, 1):.2f}s/task)")
    print("-" * 62)
    print("  reference points (official):")
    print("    human average on public eval : 66.00%")
    print("    Kaggle leaderboard           : 83.06%")
    print("=" * 62)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "score_output_metric": score,
            "score_task_metric": task_acc,
            "tasks": attempted, "outputs": total_outputs,
            "correct": correct, "solved_tasks": solved_tasks,
            "elapsed_s": round(elapsed, 1),
            "config": {"depth": args.depth, "budget": args.budget,
                       "attempts": args.attempts, "split": args.split},
            "per_task": per_task, "failures": failures,
        }, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
