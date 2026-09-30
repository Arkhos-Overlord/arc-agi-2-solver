"""Search for a program that maps every training pair correctly.

Method: breadth-first over compositions of primitives. A candidate program is
accepted only if it reproduces EVERY training output exactly - including grid
dimensions. That all-pairs constraint is what makes the search tractable: it
eliminates almost everything within a second or two.

Depth is capped by the caller. Composition of two primitives already covers a
large fraction of the easy half of ARC; going deeper is where the time goes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .prims import PRIMITIVES, Grid, dims


@dataclass
class Program:
    """A named sequence of primitives."""
    names: tuple[str, ...]

    def __call__(self, g: Grid) -> Grid:
        for n in self.names:
            g = PRIMITIVES[n](g)
        return g

    def __str__(self) -> str:
        return " -> ".join(self.names) if self.names else "identity"

    @property
    def depth(self) -> int:
        return len(self.names)


@dataclass
class SearchResult:
    solved: bool = False
    programs: list = field(default_factory=list)   # verified, best-first
    tested: int = 0
    elapsed: float = 0.0
    stopped_reason: str = ""


def _fits(pairs: list[tuple[Grid, Grid]], prog: Program) -> bool:
    """True only if prog maps EVERY train input to its exact train output."""
    for src, dst in pairs:
        try:
            out = prog(src)
        except Exception:
            return False
        if not out or dims(out) != dims(dst) or out != dst:
            return False
    return True


def _distinct_outputs(prog: Program, inputs: list[Grid]) -> list[Grid]:
    outs = []
    for g in inputs:
        try:
            outs.append(prog(g))
        except Exception:
            outs.append(g)
    return outs


def search(task: dict, max_depth: int = 2, time_budget: float = 8.0,
           top_k: int = 2, collect_alternatives: bool = True) -> SearchResult:
    """Find programs consistent with a task's training pairs.

    Returns up to `top_k` verified programs. Because the competition allows 2
    attempts per test output, the caller should submit the top 2 - they are
    free accuracy.
    """
    t0 = time.time()
    res = SearchResult()
    train = task.get("train") or []
    if not train:
        res.stopped_reason = "no training pairs"
        return res

    pairs = [(p["input"], p["output"]) for p in train]
    names = list(PRIMITIVES)

    # depth 0: the empty program (input == output for all pairs)
    ident = Program(())
    res.tested += 1
    if _fits(pairs, ident):
        res.solved = True
        res.programs.append(ident)

    # depth 1
    for a in names:
        if time.time() - t0 > time_budget:
            res.stopped_reason = "time budget"
            return res
        prog = Program((a,))
        res.tested += 1
        if _fits(pairs, prog):
            res.solved = True
            res.programs.append(prog)
            if not collect_alternatives:
                break

    if max_depth >= 2 and (not res.solved or collect_alternatives):
        for a in names:
            for b in names:
                if time.time() - t0 > time_budget:
                    res.stopped_reason = "time budget"
                    res.elapsed = time.time() - t0
                    return res
                prog = Program((a, b))
                res.tested += 1
                if _fits(pairs, prog):
                    res.solved = True
                    res.programs.append(prog)

    res.programs = res.programs[:top_k]
    res.elapsed = time.time() - t0
    if not res.stopped_reason:
        res.stopped_reason = "depth exhausted" if not res.solved else "solved"
    return res


def predict(task: dict, programs: list[Program], attempts: int = 2) -> list[list[Grid]]:
    """Apply the top programs to each test input. Returns, per test input, up
    to `attempts` predicted output grids - matching the 2-attempt rule."""
    out: list[list[Grid]] = []
    for t in (task.get("test") or []):
        cands: list[Grid] = []
        seen = set()
        for p in programs[:attempts]:
            try:
                g = p(t["input"])
            except Exception:
                g = t["input"]
            key = tuple(tuple(r) for r in g)
            if key not in seen:
                seen.add(key)
                cands.append(g)
        # pad with the raw input so the submission always has 2 attempts
        while len(cands) < attempts:
            cands.append(t["input"])
        out.append(cands[:attempts])
    return out
