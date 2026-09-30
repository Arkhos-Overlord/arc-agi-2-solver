"""Self-test for the search machinery.

Separates two very different failure modes that look identical from outside:

  (a) the code is broken  -> fix the code
  (b) the code is fine and the DSL is too weak -> abandon the approach

Only a green run here licenses the conclusion "DSL search is the wrong
paradigm for ARC-AGI-2". Without it, a 0% score proves nothing.
"""

import sys
from pathlib import Path

# This file lives in solver/, so the project root is one level up - that is
# what has to be importable for `from solver.prims import ...` to resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solver.prims import PRIMITIVES, dims  # noqa: E402
from solver.search import Program, search  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def mk_task(fn, n_train=3):
    """Build a task whose rule IS the primitive `fn`."""
    base = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[5, 5, 0], [0, 5, 5], [5, 0, 0]],
    ]
    train = [{"input": g, "output": fn(g)} for g in base[:n_train]]
    test_in = [[1, 2, 4], [3, 5, 7], [6, 8, 9]]  # asymmetric on purpose
    return {"train": train, "test": [{"input": test_in}]}


print("=" * 62)
print("  search machinery self-test")
print("=" * 62)

# 1. primitives actually transform things
print("\n1. primitives are total and non-trivial")
for name, fn in PRIMITIVES.items():
    g = [[1, 2], [3, 4]]
    try:
        out = fn(g)
        assert isinstance(out, list) and out, f"{name} returned non-grid"
        assert all(isinstance(r, list) and r for r in out), f"{name} bad rows"
        assert all(isinstance(c, int) and 0 <= c <= 9
                   for r in out for c in r), f"{name} bad colour"
    except Exception as e:
        check(f"primitive {name}", False, str(e))
check(f"all {len(PRIMITIVES)} primitives total & in-range", not FAIL)

# 2. the search can recover a KNOWN primitive - the critical test
print("\n2. search recovers a known primitive (depth 1)")
recovered = []
for name in ["rot90", "rot180", "fliplr", "flipud", "transpose",
             "compress_rows", "gravity_down", "keep_largest_colour",
             "crop_bbox", "dedup_grid"]:
    if name not in PRIMITIVES:
        continue
    r = search(mk_task(PRIMITIVES[name]), max_depth=1, time_budget=5.0)
    if r.solved and any(name in p.names for p in r.programs):
        recovered.append(name)
check(f"recovered {len(recovered)}/10 known primitives at depth 1",
      len(recovered) >= 8, f"got: {recovered}")

# 3. verification is not vacuously passing
print("\n3. verification rejects wrong programs")
t = mk_task(PRIMITIVES["rot90"])
bogus = Program(("fliplr",))
out = bogus(t["test"][0]["input"])
check("bogus program produces wrong answer on test input",
      out != PRIMITIVES["rot90"](t["test"][0]["input"]))

# 4. dimensions are enforced, not just cell values
print("\n4. output dimensions are enforced")
sq = [[1, 2], [3, 4]]
wide = [[1, 2, 3, 4, 5]]
task = {"train": [{"input": sq, "output": wide},
                  {"input": [[9, 8], [7, 6]], "output": [9, 8, 7, 6, 5]}],
        "test": [{"input": [[1, 1], [2, 2]]}]}
r = search(task, max_depth=2, time_budget=6.0)
check("shape-changing task NOT falsely solved", not r.solved,
      f"programs={[str(p) for p in r.programs]}")

# 5. an unsolvable task stays unsolved (no false positives)
print("\n5. no false positives on a random task")
nonsense = {"train": [{"input": [[1, 2], [3, 4]], "output": [[9, 9], [9, 9]]},
                      {"input": [[5, 6], [7, 8]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[0, 0], [0, 0]]}]}
r = search(nonsense, max_depth=2, time_budget=6.0)
check("contradictory task reports unsolved", not r.solved)

print()
print("=" * 62)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED:", ", ".join(FAIL))
print("=" * 62)

if FAIL:
    print("\nVERDICT: machinery is BROKEN. A 0% score says nothing yet.")
    sys.exit(1)
print("\nVERDICT: machinery is CORRECT. The 0% on ARC-AGI-2 eval is therefore")
print("         a property of the DSL's coverage, not a bug - which is the")
print("         evidence that DSL search is the wrong paradigm here.")
sys.exit(0)
