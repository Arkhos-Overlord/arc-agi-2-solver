"""Diagnostic: why does TTT training return inf?

Prints the real sequence lengths the builder produces, so the max-cells
threshold is set from data instead of guessed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ttt import build_example, grid_to_cells  # noqa: E402

for split in ("evaluation", "training"):
    d = ROOT / "data" / split
    print(f"=== {split} ===")
    files = sorted(d.glob("*.json"))[:8]
    for f in files:
        task = json.loads(f.read_text(encoding="utf-8"))
        pairs = [(p["input"], p["output"]) for p in task["train"]]
        lens = []
        for k, (i_g, o_g) in enumerate(pairs):
            ctx = pairs[:k] + pairs[k + 1:]
            if not ctx:
                lens.append("no-ctx")
                continue
            s = build_example(ctx, i_g, 10 ** 6)
            if s is None:
                lens.append("None")
            else:
                lens.append(len(s) + len(grid_to_cells(o_g)))
        dims = [(len(p["input"]), len(p["input"][0]),
                 len(p["output"]), len(p["output"][0]))
                for p in task["train"][:3]]
        print(f"  {f.stem}: pairs={len(pairs)} needed={lens}")
        print(f"      in_h,in_w,out_h,out_w = {dims}")
    print()
