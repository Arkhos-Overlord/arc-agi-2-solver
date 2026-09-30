"""Augmentation and stability scoring.

The problem this solves: a task often admits MANY programs that reproduce every
training pair perfectly. Depth-1/2 search finds dozens. They disagree on the
test input, and picking arbitrarily wastes the free second attempt.

The fix is a principled selection signal. ARC grids are invariant to
dihedral transforms (8 rotations/reflections) and colour permutation. So:

  1. find all programs consistent with the training pairs
  2. generate an augmented view of the whole task
  3. re-run the search on the augmented task
  4. a candidate that survives augmentation unchanged is more likely to be the
     true rule; one that flips to a different answer is a coincidence fit

Aggregate those votes across many augmentations and rank by stability. The top
2 go into the two permitted attempts.

This is the standard ARC trick and it needs no GPU, which is what we have.
"""

from __future__ import annotations

import random
from collections import Counter

Grid = list[list[int]]

DIHEDRAL = ("identity", "rot90", "rot180", "rot270",
            "fliplr", "flipud", "transpose", "anti_transpose")


# ------------------------------------------------------------------ ops

def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def _rot180(g):
    return [row[::-1] for row in g[::-1]]


def _rot270(g):
    return [list(r) for r in zip(*g)][::-1]


def _fliplr(g):
    return [row[::-1] for row in g]


def _flipud(g):
    return g[::-1]


def _transpose(g):
    return [list(r) for r in zip(*g)]


def _anti_transpose(g):
    return [list(r) for r in zip(*g[::-1])][::-1]


DIHEDRAL_FN = {
    "identity": lambda g: [r[:] for r in g],
    "rot90": _rot90, "rot180": _rot180, "rot270": _rot270,
    "fliplr": _fliplr, "flipud": _flipud,
    "transpose": _transpose, "anti_transpose": _anti_transpose,
}


def dihedral(g: Grid, name: str) -> Grid:
    return DIHEDRAL_FN[name](g)


def colour_permutation(g: Grid, perm: dict[int, int]) -> Grid:
    return [[perm.get(c, c) for c in row] for row in g]


def random_colour_perm(rng: random.Random, colours: set[int]) -> dict[int, int]:
    """Randomly permute the non-background colours, keeping 0 as background.

    Background must stay 0: in ARC, 0 means "empty" and transformations treat
    it specially, so permuting it would change the task's meaning.
    """
    others = sorted(c for c in colours if c != 0)
    shuffled = others[:]
    rng.shuffle(shuffled)
    return dict(zip(others, shuffled))


def invert(g: Grid) -> Grid:
    """Colour inversion 1..9 -> 10-c, leaving 0 alone."""
    return [[0 if c == 0 else 10 - c for c in row] for row in g]


# ------------------------------------------------------------------ task view

class TaskView:
    """A (possibly transformed) view of a task, remembering how to undo it."""

    def __init__(self, task: dict, dih: str = "identity",
                 cperm: dict[int, int] | None = None, flip_colours: bool = False):
        self.dih = dih
        self.cperm = cperm or {}
        self.flip = flip_colours
        self.train = [
            {"input": self.fwd(p["input"]), "output": self.fwd(p["output"])}
            for p in task.get("train", [])
        ]
        self.test = [{"input": self.fwd(t["input"])} for t in task.get("test", [])]

    def _map(self, g: Grid) -> Grid:
        out = dihedral(g, self.dih)
        if self.cperm:
            out = colour_permutation(out, self.cperm)
        if self.flip:
            out = invert(out)
        return out

    def fwd(self, g: Grid) -> Grid:
        """Original -> view space."""
        return self._map(g)

    def inv(self, g: Grid) -> Grid:
        """View space -> original. Exact inverse of fwd."""
        out = g
        if self.flip:
            out = [[0 if c == 0 else 10 - c for c in row] for row in out]
        if self.cperm:
            rev = {v: k for k, v in self.cperm.items()}
            out = colour_permutation(out, rev)
        return DIHEDRAL_FN[self.dih](out)


def make_views(task: dict, n: int = 8, seed: int = 0) -> list[TaskView]:
    """Build n augmented views: the identity plus dihedral/colour variants."""
    rng = random.Random(seed)
    colours: set[int] = set()
    for p in task.get("train", []):
        for g in (p["input"], p["output"]):
            colours.update(c for row in g for c in row)

    views = [TaskView(task)]  # identity first, always
    names = list(DIHEDRAL)
    rng.shuffle(names)
    tries = 0
    while len(views) < n and tries < n * 6:
        tries += 1
        dih = rng.choice(names)
        perm = random_colour_perm(rng, colours) if rng.random() < 0.5 else {}
        flip = rng.random() < 0.15
        if dih == "identity" and not perm and not flip:
            continue
        views.append(TaskView(task, dih, perm, flip))
    return views


def key(g: Grid) -> tuple:
    """Hashable canonical form of a grid."""
    return tuple(tuple(row) for row in g)
