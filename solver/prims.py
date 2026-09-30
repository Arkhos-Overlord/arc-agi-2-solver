"""Grid primitives for ARC-AGI-2.

A "grid" is a list of lists of ints in 0..9 (0 = background).

Every primitive is total: it must return a grid for any grid. A primitive that
can crash is a primitive that will crash mid-competition, so shape-sensitive
ones fall back to returning their input rather than raising.

Primitives are grouped by the ARC concept they express, because that grouping
is what the search uses to stay tractable.
"""

from __future__ import annotations

Grid = list[list[int]]

BG = 0


# ------------------------------------------------------------ core helpers

def dims(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def empty_like(g: Grid) -> Grid:
    h, w = dims(g)
    return [[BG] * w for _ in range(h)]


def colours(g: Grid) -> set[int]:
    return {c for row in g for c in row}


def non_bg(g: Grid) -> list[int]:
    return [c for row in g for c in row if c != BG]


def bbox(g: Grid, colour: int | None = None) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) inclusive-exclusive of non-background cells."""
    h, w = dims(g)
    r0, r1, c0, c1 = h, 0, 0, 0
    found = False
    for r in range(h):
        for c in range(w):
            if g[r][c] != BG and (colour is None or g[r][c] == colour):
                found = True
                r0, r1 = min(r0, r), max(r1, r + 1)
                c0, c1 = min(c0, c), max(c1, c + 1)
    return (r0, r1, c0, c1) if found else None


def most_common(g: Grid) -> int:
    flat = [c for row in g for c in row]
    if not flat:
        return BG
    counts = {c: flat.count(c) for c in set(flat)}
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _safe(fn):
    """Wrap a primitive so a failure returns the input unchanged.

    A wrong answer costs one candidate. An exception costs the whole search.
    """
    def wrapped(g: Grid) -> Grid:
        try:
            out = fn(g)
        except Exception:
            return g
        return out if out else g
    return wrapped


# ------------------------------------------------------------ geometry

@_safe
def identity(g):
    return [row[:] for row in g]


@_safe
def rot90(g):
    return [list(r) for r in zip(*g[::-1])]


@_safe
def rot180(g):
    return [row[::-1] for row in g[::-1]]


@_safe
def rot270(g):
    return [list(r) for r in zip(*g)][::-1]


@_safe
def fliplr(g):
    return [row[::-1] for row in g]


@_safe
def flipud(g):
    return g[::-1]


@_safe
def transpose(g):
    return [list(r) for r in zip(*g)]


@_safe
def anti_transpose(g):
    return [list(r) for r in zip(*g[::-1])][::-1]


@_safe
def crop_bbox(g):
    b = bbox(g)
    if not b:
        return g
    r0, r1, c0, c1 = b
    return [row[c0:c1] for row in g[r0:r1]]


@_safe
def crop_each_colour(g):
    """Isolate the largest-colour object and crop to it."""
    cs = non_bg(g)
    if not cs:
        return g
    target = max(set(cs), key=cs.count)
    b = bbox(g, target)
    if not b:
        return g
    r0, r1, c0, c1 = b
    return [[g[r][c] if g[r][c] == target else BG for c in range(c0, c1)]
            for r in range(r0, r1)]


@_safe
def compress_rows(g):
    """Drop rows that are identical to the row above."""
    out = []
    for row in g:
        if not out or out[-1] != row:
            out.append(row[:])
    return out


@_safe
def compress_cols(g):
    t = [list(r) for r in zip(*g)]
    out = []
    for col in t:
        if not out or out[-1] != col:
            out.append(col)
    return [list(r) for r in zip(*out)]


@_safe
def dedup_grid(g):
    """Full ARC-AGI-1 'compress' op: remove duplicate rows AND columns."""
    rows, seen = [], set()
    for row in g:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            rows.append(row[:])
    cols, seen = [], set()
    for col in zip(*rows):
        if col not in seen:
            seen.add(col)
            cols.append(col)
    return [list(r) for r in zip(*cols)] if cols else rows


# ------------------------------------------------------------ scaling

@_safe
def scale2(g):
    return [[v for v in row for _ in range(2)] for row in g for _ in range(2)]


@_safe
def scale3(g):
    return [[v for v in row for _ in range(3)] for row in g for _ in range(3)]


@_safe
def downsample2(g):
    h, w = dims(g)
    if h < 2 or w < 2:
        return g
    return [[g[r][c] for r in range(0, h, 2)] for c in range(0, w, 2)]


@_safe
def upscale_to_square(g):
    """Repeat the grid until it is at least as tall as wide (or vice versa)."""
    h, w = dims(g)
    n = max(1, -(-h // w)) if h < w else max(1, -(-w // h))
    reps = max(1, n)
    return [row[:] for row in g for _ in range(reps)] if h >= w else \
           [row for row in g for _ in range(reps)]


# ------------------------------------------------------------ colour ops

@_safe
def most_common_to_bg(g):
    c = most_common(g)
    return [[BG if v == c else v for v in row] for row in g]


@_safe
def keep_largest_colour(g):
    cs = non_bg(g)
    if not cs:
        return g
    keep = max(set(cs), key=cs.count)
    return [[v if v == keep else BG for v in row] for row in g]


@_safe
def keep_smallest_colour(g):
    counts = {}
    for c in non_bg(g):
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return g
    keep = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [[v if v == keep else BG for v in row] for row in g]


@_safe
def remap_most_common(g):
    """Recolour the dominant non-background to 1, everything else by rank."""
    cs = non_bg(g)
    if not cs:
        return g
    order = sorted(set(cs), key=lambda c: (-cs.count(c), c))
    mapping = {c: i + 1 for i, c in enumerate(order)}
    return [[mapping.get(v, BG) for v in row] for row in g]


# ------------------------------------------------------------ structural

@_safe
def gravity_down(g):
    h, w = dims(g)
    out = empty_like(g)
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c] != BG]
        for i, v in enumerate(reversed(col)):
            out[h - 1 - i][c] = v
    return out


@_safe
def gravity_up(g):
    h, w = dims(g)
    out = empty_like(g)
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c] != BG]
        for i, v in enumerate(col):
            out[i][c] = v
    return out


@_safe
def gravity_left(g):
    h, w = dims(g)
    out = empty_like(g)
    for r in range(h):
        row = [v for v in g[r] if v != BG]
        for i, v in enumerate(row):
            out[r][i] = v
    return out


@_safe
def gravity_right(g):
    h, w = dims(g)
    out = empty_like(g)
    for r in range(h):
        row = [v for v in g[r] if v != BG]
        for i, v in enumerate(reversed(row)):
            out[r][w - 1 - i] = v
    return out


@_safe
def remove_noise(g):
    """Drop connected components smaller than the median size."""
    comps = components(g)
    if len(comps) < 2:
        return g
    sizes = sorted(len(c) for c in comps)
    median = sizes[len(sizes) // 2]
    keep = {id(c) for c in comps if len(c) >= median}
    out = empty_like(g)
    for comp in comps:
        if id(comp) in keep:
            for r, c, _ in comp:
                out[r][c] = g[r][c]
    return out


def components(g: Grid) -> list[list[tuple[int, int, int]]]:
    """4-connected same-colour components -> list of [(r, c, colour)]."""
    h, w = dims(g)
    seen = [[False] * w for _ in range(h)]
    out = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or g[r0][c0] == BG:
                continue
            col = g[r0][c0]
            stack = [(r0, c0)]
            seen[r0][c0] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c, col))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                            and g[nr][nc] == col):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


@_safe
def largest_object(g):
    comps = components(g)
    if not comps:
        return g
    big = max(comps, key=len)
    b = bbox_of_cells(big)
    r0, r1, c0, c1 = b
    return [[g[r][c] if g[r][c] != BG else BG for c in range(c0, c1)]
            for r in range(r0, r1)]


@_safe
def unique_object(g):
    """Keep components whose size is unique (appear exactly once)."""
    comps = components(g)
    if len(comps) < 2:
        return g
    sizes = [len(c) for c in comps]
    keep_sizes = {s for s in sizes if sizes.count(s) == 1}
    out = empty_like(g)
    for comp in comps:
        if len(comp) in keep_sizes:
            for r, c, _ in comp:
                out[r][c] = g[r][c]
    return out


def bbox_of_cells(cells):
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


@_safe
def symmetrise(g):
    """Mirror left onto right, then top onto bottom (union of content)."""
    h, w = dims(g)
    out = [row[:] for row in g]
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v != BG:
                out[r][w - 1 - c] = v
    out2 = [row[:] for row in out]
    for r in range(h):
        for c in range(w):
            v = out[r][c]
            if v != BG:
                out2[h - 1 - r][c] = v
    return out2


@_safe
def fill_holes(g):
    """Flood background from the border; anything unreached is a hole -> fill."""
    h, w = dims(g)
    seen = [[False] * w for _ in range(h)]
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not seen[r][c] and g[r][c] == BG:
                seen[r][c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not seen[r][c] and g[r][c] == BG:
                seen[r][c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                    and g[nr][nc] == BG:
                seen[nr][nc] = True
                stack.append((nr, nc))
    out = [row[:] for row in g]
    for r in range(h):
        for c in range(w):
            if not seen[r][c]:
                out[r][c] = max(colours(g) - {BG} or {1})
    return out


# ------------------------------------------------------------ the registry

PRIMITIVES = {
    # geometry
    "identity": identity,
    "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "fliplr": fliplr, "flipud": flipud,
    "transpose": transpose, "anti_transpose": anti_transpose,
    "crop_bbox": crop_bbox, "crop_each_colour": crop_each_colour,
    "compress_rows": compress_rows, "compress_cols": compress_cols,
    "dedup_grid": dedup_grid,
    # scaling
    "scale2": scale2, "scale3": scale3, "downsample2": downsample2,
    "upscale_to_square": upscale_to_square,
    # colour
    "most_common_to_bg": most_common_to_bg,
    "keep_largest_colour": keep_largest_colour,
    "keep_smallest_colour": keep_smallest_colour,
    "remap_most_common": remap_most_common,
    # structural
    "gravity_down": gravity_down, "gravity_up": gravity_up,
    "gravity_left": gravity_left, "gravity_right": gravity_right,
    "remove_noise": remove_noise, "largest_object": largest_object,
    "unique_object": unique_object, "symmetrise": symmetrise,
    "fill_holes": fill_holes,
}

__all__ = ["PRIMITIVES", "dims", "colours", "non_bg", "bbox", "most_common",
           "components", "Grid", "BG"]
