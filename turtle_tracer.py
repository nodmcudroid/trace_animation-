#!/usr/bin/env python3
"""
Animated Image -> Turtle Drawing Tracer
=======================================

Loads a black-and-white line drawing, extracts the *centre-lines* of its
strokes and lets a Python Turtle redraw the picture by moving its pen along
those coordinates, like a hand drawing in real time.

The Turtle never copies or displays the source image: everything on the
canvas is produced by ``goto`` calls along extracted coordinates.

Pipeline (one function per stage)
---------------------------------
    1. load_image()          Pillow    - open, fix EXIF rotation, flatten transparency
    2. preprocess()          OpenCV    - grayscale, resize, blur, threshold, invert
    3. extract_paths()       NumPy/cv2 - thin to 1-px skeleton, trace ordered strokes
    4. map_to_turtle()                 - image coords -> centred, y-flipped Turtle coords
    5. TurtleAnimator                  - timer-driven pen animation (pause/resume/speed)
    6. render_png()          Pillow    - save the finished drawing as a PNG

Usage
-----
    python turtle_tracer.py                       # open the UI
    python turtle_tracer.py picture.png           # open the UI with an image loaded
    python turtle_tracer.py picture.png --export out.png   # no UI: trace + save PNG
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:  # The UI needs Tk. The pipeline / --export mode works without it.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import turtle
    from PIL import ImageTk

    GUI_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover
    GUI_AVAILABLE = False
    _GUI_IMPORT_ERROR = _exc

# =============================================================================
# Configuration
# =============================================================================
CANVAS_W, CANVAS_H = 800, 620     # Turtle drawing area (pixels)
CANVAS_MARGIN = 24                # empty border kept around the drawing
MAX_UPSCALE = 2.5                 # never enlarge a small drawing more than this

WORKING_SIZE = 720                # longest image side while analysing (pixels)
BLUR_KSIZE = 3                    # Gaussian blur kernel (odd)
MIN_COMPONENT_AREA = 8            # ignore specks smaller than this many pixels
AUTO_MAX_INK = 0.15               # 'auto' mode: more ink than this -> use adaptive threshold
SMOOTH_SIGMA = 3.0                # Gaussian smoothing of traced paths (0 = off)
CORNER_ANGLE = 40.0               # turns sharper than this (deg) are kept as corners
SIMPLIFY_EPSILON = 0.15           # Douglas-Peucker tolerance in pixels (0 = off)
CONNECT_DISTANCE = 2.5            # gaps up to this size are drawn, not "jumped"

TICK_MS = 16                      # animation timer period (~60 frames/second)
PX_PER_TICK_PER_SPEED = 0.5       # speed slider 1..100 -> 0.5 .. 50 px per tick
PEN_UP_SPEED_FACTOR = 4.0         # pen-up glides are this much faster
DEFAULT_SPEED = 20

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

Pixel = Tuple[int, int]           # (row, col)


# =============================================================================
# Stage 1 - Load image (Pillow)
# =============================================================================
def load_image(path: str) -> Image.Image:
    """Open an image file and return it as an RGB Pillow image.

    Raises ``FileNotFoundError`` / ``ValueError`` with a human-friendly message
    for missing, unreadable or corrupt files.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found:\n{path}")
    name = os.path.basename(path)
    try:
        with Image.open(path) as im:
            im.load()                              # force decoding now, catch corruption
            im = ImageOps.exif_transpose(im)       # honour camera rotation flags
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"'{name}' is not a valid or readable image file.") from exc

    if im.width < 2 or im.height < 2:
        raise ValueError(f"'{name}' is too small to contain a drawing.")

    # Transparent PNGs: composite onto white so transparency becomes background.
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, rgba)
    return im.convert("RGB")


# =============================================================================
# Stage 2 - Preprocessing (OpenCV)
# =============================================================================
def to_working_gray(img: Image.Image, working_size: int = WORKING_SIZE) -> np.ndarray:
    """Grayscale + resize so the longest side is ``working_size`` (aspect kept)."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    scale = min(working_size / max(h, w), 4.0)     # cap up-scaling of tiny images
    if abs(scale - 1.0) > 1e-3:
        new_size = (max(2, round(w * scale)), max(2, round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, new_size, interpolation=interp)
    return gray


def remove_small_components(mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    """Delete connected blobs smaller than ``min_area`` pixels (scanner noise)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def binarize(gray: np.ndarray, method: str = "auto") -> np.ndarray:
    """Return a uint8 mask where the *drawing lines are 255* and background 0.

    * Dark-on-light vs light-on-dark is detected from the image border, so
      white-on-black drawings work too.
    * ``'otsu'``     - one global threshold; best for clean scans / exports.
    * ``'adaptive'`` - local threshold; copes with shadows / uneven lighting.
    * ``'auto'``     - Otsu, unless it marks an implausibly large part of the
                       image as ink (a sign of shadows), then adaptive.
    """
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if np.median(border) < 110:                    # dark background -> flip polarity
        gray = 255 - gray

    k = BLUR_KSIZE | 1                             # force odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)    # suppress noise before thresholding

    def otsu() -> np.ndarray:
        _, m = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return m

    def adaptive() -> np.ndarray:
        block = max(15, (min(gray.shape) // 15) | 1)   # odd neighbourhood size
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, 10)

    if method == "adaptive":
        mask = adaptive()
    else:
        mask = otsu()
        if method == "auto" and np.count_nonzero(mask) > mask.size * AUTO_MAX_INK:
            mask = adaptive()

    # Safety net: a line drawing is mostly background. If more than half the
    # image is "ink", the polarity guess was wrong - invert.
    if np.count_nonzero(mask) > mask.size * 0.5:
        mask = cv2.bitwise_not(mask)
    return remove_small_components(mask)


def preprocess(img: Image.Image, method: str = "auto") -> np.ndarray:
    """Stage 2 entry point: Pillow image -> clean binary mask (lines = 255)."""
    return binarize(to_working_gray(img), method)


# =============================================================================
# Stage 3 - Extract drawing paths (skeletonisation + path tracing)
# =============================================================================
def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-pixel-wide skeleton (Zhang-Suen algorithm).

    Implemented with vectorised NumPy so no ``opencv-contrib`` is required.
    A skeleton is the centre-line of every stroke, so each line is traced
    once - unlike contours, which would trace both edges of every line.
    """
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:]
            p6, p7, p8, p9 = img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]
            centre = img[1:-1, 1:-1]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9]

            neighbours = sum(n.astype(np.int16) for n in ring)           # B(P)
            transitions = np.zeros_like(neighbours)                      # A(P)
            for a, b in zip(ring, ring[1:] + ring[:1]):
                transitions += (a == 0) & (b == 1)

            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remove = (centre == 1) & (neighbours >= 2) & (neighbours <= 6) \
                & (transitions == 1) & cond
            if remove.any():
                centre[remove] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def estimate_line_width(mask: np.ndarray, skeleton: np.ndarray) -> float:
    """Typical stroke thickness (px), from the distance transform on the skeleton."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    values = dist[skeleton]
    if values.size == 0:
        return 1.0
    return float(max(1.0, 2.0 * np.median(values) - 1.0))


def build_graph(skeleton: np.ndarray) -> Dict[Pixel, List[Pixel]]:
    """Pixel adjacency graph of the skeleton (8-connectivity).

    Diagonal links that merely short-cut an L-shaped corner are dropped, so a
    staircase of pixels is a clean chain instead of a chain of tiny triangles.
    """
    padded = np.pad(skeleton, 1)
    graph: Dict[Pixel, List[Pixel]] = {}
    rows, cols = np.nonzero(skeleton)
    for r, c in zip(rows.tolist(), cols.tolist()):
        pr, pc = r + 1, c + 1
        n, s = padded[pr - 1, pc], padded[pr + 1, pc]
        w, e = padded[pr, pc - 1], padded[pr, pc + 1]
        nbrs: List[Pixel] = []
        if n: nbrs.append((r - 1, c))
        if s: nbrs.append((r + 1, c))
        if w: nbrs.append((r, c - 1))
        if e: nbrs.append((r, c + 1))
        if padded[pr - 1, pc - 1] and not (n or w): nbrs.append((r - 1, c - 1))
        if padded[pr - 1, pc + 1] and not (n or e): nbrs.append((r - 1, c + 1))
        if padded[pr + 1, pc - 1] and not (s or w): nbrs.append((r + 1, c - 1))
        if padded[pr + 1, pc + 1] and not (s or e): nbrs.append((r + 1, c + 1))
        graph[(r, c)] = nbrs
    return graph


def prune_spurs(graph: Dict[Pixel, List[Pixel]], max_len: int) -> None:
    """Remove tiny dead-end branches (<= ``max_len`` px) hanging off a junction.

    Thinning leaves little whiskers at stroke corners and ends; they are not
    part of the drawing and would make the pen scribble back and forth.
    """
    endpoints = [p for p, nb in graph.items() if len(nb) == 1]
    for ep in endpoints:
        if ep not in graph or len(graph[ep]) != 1:
            continue
        chain, prev, cur = [ep], ep, graph[ep][0]
        is_spur = False
        while len(chain) <= max_len:
            degree = len(graph[cur])
            if degree >= 3:            # reached a junction -> it was a spur
                is_spur = True
                break
            if degree == 1:            # reached another endpoint -> isolated segment
                break
            chain.append(cur)
            nxt = [n for n in graph[cur] if n != prev]
            prev, cur = cur, nxt[0]
        if is_spur and len(chain) <= max_len:
            for node in chain:
                for nb in graph[node]:
                    if nb in graph and node in graph[nb]:
                        graph[nb].remove(node)
                del graph[node]


def _edge(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    return (a, b) if a < b else (b, a)


def trace_strokes(graph: Dict[Pixel, List[Pixel]]) -> List[List[Pixel]]:
    """Cover every skeleton edge exactly once with long, ordered pixel paths.

    * Walks start at free endpoints first, then at junctions / loops.
    * At a junction the walker continues in the straightest direction, so
      curves and corners are followed naturally.
    * Edges are marked as used, so no line is traced twice.
    * Separate branches / objects become separate strokes.
    """
    visited = set()
    strokes: List[List[Pixel]] = []

    def free_edges(node: Pixel) -> List[Pixel]:
        return [n for n in graph[node] if _edge(node, n) not in visited]

    def walk(start: Pixel) -> List[Pixel]:
        path, cur = [start], start
        while True:
            options = free_edges(cur)
            if not options:
                break
            if len(options) == 1 or len(path) == 1:
                nxt = options[0]
            else:                                   # choose the straightest continuation
                back = path[max(0, len(path) - 6)]
                dy, dx = cur[0] - back[0], cur[1] - back[1]
                norm = math.hypot(dy, dx) or 1.0
                nxt, best = options[0], -2.0
                for cand in options:
                    vy, vx = cand[0] - cur[0], cand[1] - cur[1]
                    score = (dy * vy + dx * vx) / (norm * math.hypot(vy, vx))
                    if score > best:
                        nxt, best = cand, score
            visited.add(_edge(cur, nxt))
            path.append(nxt)
            cur = nxt
        return path

    for p in sorted(p for p, nb in graph.items() if len(nb) == 1):   # open ends
        if free_edges(p):
            strokes.append(walk(p))
    for p in sorted(graph):                                           # loops / junction leftovers
        while free_edges(p):
            strokes.append(walk(p))
    for p in sorted(graph):                                           # isolated dots
        if not graph[p]:
            strokes.append([p])
    return strokes


def smooth_polyline(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a pixel path to remove staircase jaggies.

    End points stay fixed; closed loops are smoothed seamlessly.
    """
    n = len(pts)
    if sigma <= 0 or n < 4:
        return pts
    sigma = min(sigma, max(0.7, n / 6.0))                    # short paths: gentler kernel
    radius = max(1, int(math.ceil(2 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    if np.allclose(pts[0], pts[-1]):                         # closed loop
        core = pts[:-1]
        if len(core) < 3:
            return pts
        padded = np.pad(core, ((radius, radius), (0, 0)), mode="wrap")
        out = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)], axis=1)
        return np.vstack([out, out[:1]])

    padded = np.pad(pts, ((radius, radius), (0, 0)), mode="reflect", reflect_type="odd")
    out = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)], axis=1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def find_corners(pts: np.ndarray, closed: bool, k: int) -> List[int]:
    """Indices where the path turns sharply (more than ``CORNER_ANGLE`` degrees).

    The turn is measured between the chords reaching ``k`` points back and
    forward, which ignores single-pixel staircase noise but catches real corners.
    """
    n = len(pts)
    if n < 2 * k + 1:
        return []
    idx = np.arange(n) if closed else np.arange(k, n - k)
    cur, prev, nxt = pts[idx], pts[(idx - k) % n], pts[(idx + k) % n]
    v1, v2 = cur - prev, nxt - cur
    cos = (v1 * v2).sum(axis=1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-9)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    corners: List[int] = []
    for j in np.nonzero(turn > CORNER_ANGLE)[0]:
        window = np.take(turn, np.arange(j - k, j + k + 1), mode="wrap" if closed else "clip")
        if turn[j] >= window.max() and (not corners or idx[j] - corners[-1] > k):
            corners.append(int(idx[j]))                         # local maximum of the turn angle
    return corners


def refine_stroke(pts: np.ndarray, k: int) -> np.ndarray:
    """Smooth a traced path strongly *without* rounding off its corners.

    A pixel skeleton is a staircase; light smoothing leaves wobbly lines and
    polygon-looking circles. So the path is cut at its sharp corners and each
    piece is smoothed heavily with its end points pinned (corners stay sharp).
    """
    if len(pts) < 4:
        return pts
    closed = bool(np.allclose(pts[0], pts[-1]))
    core = pts[:-1] if closed else pts
    corners = find_corners(core, closed, k)

    if closed:
        if not corners:
            return smooth_polyline(pts, SMOOTH_SIGMA)            # plain loop (circle, ellipse...)
        first = corners[0]                                       # re-start the loop at a corner
        core = np.roll(core, -first, axis=0)
        corners = [(c - first) % len(core) for c in corners[1:]]
        pts = np.vstack([core, core[:1]])

    bounds = [0] + sorted(set(c for c in corners if 0 < c < len(pts) - 1)) + [len(pts) - 1]
    pieces: List[np.ndarray] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = smooth_polyline(pts[a:b + 1], SMOOTH_SIGMA)
        pieces.append(piece if not pieces else piece[1:])
    return np.vstack(pieces)


def simplify_polyline(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker: keep corners, drop redundant points (fewer ``goto`` calls)."""
    if epsilon <= 0 or len(pts) < 3:
        return pts

    def dp(chain: np.ndarray) -> np.ndarray:
        approx = cv2.approxPolyDP(chain.astype(np.float32).reshape(-1, 1, 2), epsilon, False)
        return approx.reshape(-1, 2).astype(float)

    if len(pts) >= 4 and np.allclose(pts[0], pts[-1]):
        # A closed loop has a zero-length "chord" (first point == last point), which
        # breaks Douglas-Peucker. Split the loop at the point farthest from the
        # start and simplify each half on its own.
        k = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
        return np.vstack([dp(pts[:k + 1]), dp(pts[k:])[1:]])
    return dp(pts)


def order_strokes(strokes: List[np.ndarray],
                  start: Tuple[float, float] = (0.0, 0.0)) -> Tuple[List[np.ndarray], List[bool]]:
    """Order strokes like a person would draw: always continue with the nearest one.

    Strokes may be reversed so the pen enters at the closer end. Returns the
    ordered strokes plus, for each, whether it is close enough to the previous
    stroke's end that the pen can stay down (no lift, no jump).
    """
    count = len(strokes)
    starts = np.array([s[0] for s in strokes])
    ends = np.array([s[-1] for s in strokes])
    free = np.ones(count, dtype=bool)
    pos = np.asarray(start, dtype=float)
    ordered: List[np.ndarray] = []
    connected: List[bool] = []

    for k in range(count):
        d_start = np.where(free, np.hypot(*(starts - pos).T), np.inf)
        d_end = np.where(free, np.hypot(*(ends - pos).T), np.inf)
        i_s, i_e = int(np.argmin(d_start)), int(np.argmin(d_end))
        if d_start[i_s] <= d_end[i_e]:
            idx, flip, gap = i_s, False, d_start[i_s]
        else:
            idx, flip, gap = i_e, True, d_end[i_e]
        stroke = strokes[idx][::-1].copy() if flip else strokes[idx]
        ordered.append(stroke)
        connected.append(k > 0 and gap <= CONNECT_DISTANCE)
        free[idx] = False
        pos = stroke[-1]
    return ordered, connected


@dataclass
class TracedDrawing:
    """Result of the analysis stage, in *image* pixel coordinates (y points down)."""
    strokes: List[np.ndarray]     # ordered polylines
    connected: List[bool]         # pen stays down when moving from the previous stroke
    size: Tuple[int, int]         # analysed image size (width, height)
    line_width: float             # typical stroke thickness in pixels


def extract_paths(mask: np.ndarray) -> TracedDrawing:
    """Stage 3 entry point: binary mask -> ordered, smoothed stroke polylines."""
    skeleton = zhang_suen_thinning(mask)
    if not skeleton.any():
        raise ValueError("No drawing lines were found in this image.\n"
                         "Try the 'Adaptive' threshold, or use a darker line drawing.")

    line_width = estimate_line_width(mask, skeleton)
    graph = build_graph(skeleton)
    prune_spurs(graph, max_len=max(3, int(round(line_width))))

    corner_window = max(4, int(round(1.5 * line_width)) + 1)
    strokes: List[np.ndarray] = []
    for path in trace_strokes(graph):
        pts = np.array([(c, r) for r, c in path], dtype=float)     # (row,col) -> (x,y)
        if len(pts) > 1:
            pts = simplify_polyline(refine_stroke(pts, corner_window), SIMPLIFY_EPSILON)
        strokes.append(pts)
    if not strokes:
        raise ValueError("No drawing lines were found in this image.")

    ordered, connected = order_strokes(strokes)
    h, w = mask.shape
    return TracedDrawing(ordered, connected, (w, h), line_width)


def trace_image(img: Image.Image, method: str = "auto") -> TracedDrawing:
    """Stages 2 + 3 in one call (this is what the UI runs in a worker thread)."""
    return extract_paths(preprocess(img, method))


# =============================================================================
# Stage 4 - Coordinate mapping (image -> Turtle)
# =============================================================================
@dataclass
class TurtlePath:
    strokes: List[np.ndarray]     # Turtle coordinates (origin centre, y up)
    connected: List[bool]
    pen_width: float


def map_to_turtle(drawing: TracedDrawing, canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H,
                  margin: int = CANVAS_MARGIN) -> TurtlePath:
    """Centre the drawing on the canvas, keep its proportions, flip the Y axis.

    Images have their origin top-left with y growing *down*; Turtle's origin is
    the canvas centre with y growing *up*.
    """
    everything = np.vstack(drawing.strokes)
    lo, hi = everything.min(axis=0), everything.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = min((canvas_w - 2 * margin) / span[0],
                (canvas_h - 2 * margin) / span[1], MAX_UPSCALE)    # one scale -> same aspect ratio
    centre = (lo + hi) / 2.0

    strokes = []
    for s in drawing.strokes:
        t = np.empty_like(s)
        t[:, 0] = (s[:, 0] - centre[0]) * scale
        t[:, 1] = (centre[1] - s[:, 1]) * scale                    # Y inversion
        strokes.append(t)
    pen_width = float(np.clip(drawing.line_width * scale, 1.0, 14.0))
    return TurtlePath(strokes, drawing.connected, pen_width)


# =============================================================================
# Stage 5 - Animated drawing
# =============================================================================
@dataclass
class Move:
    """One pen movement. ``pts[0]`` is where the pen already is."""
    pen_down: bool
    pts: np.ndarray
    is_dot: bool = False


def build_moves(path: TurtlePath) -> List[Move]:
    """Turn ordered strokes into pen-down drawing moves and pen-up travel moves."""
    moves: List[Move] = []
    pos: Optional[np.ndarray] = None
    for i, stroke in enumerate(path.strokes):
        if len(stroke) == 1:                                       # a dot
            if pos is not None and np.hypot(*(stroke[0] - pos)) > 1e-9:
                moves.append(Move(False, np.vstack([pos, stroke[0]])))
            moves.append(Move(True, stroke, is_dot=True))
            pos = stroke[0]
            continue
        if pos is not None:
            if path.connected[i]:                                  # tiny gap: keep drawing
                stroke = np.vstack([pos, stroke])
            else:                                                  # lift pen, glide over
                moves.append(Move(False, np.vstack([pos, stroke[0]])))
        moves.append(Move(True, stroke))
        pos = stroke[-1]
    return moves


def polylines_from_moves(moves: List[Move]) -> List[np.ndarray]:
    """All pen-down polylines (consecutive drawing moves are merged)."""
    polys: List[np.ndarray] = []
    run: List[np.ndarray] = []
    for mv in moves:
        if mv.pen_down and not mv.is_dot:
            run.append(mv.pts if not run else mv.pts[1:])
            continue
        if run:
            polys.append(np.vstack(run))
            run = []
        if mv.is_dot:
            polys.append(mv.pts)
    if run:
        polys.append(np.vstack(run))
    return polys


def _moves_length(moves: List[Move]) -> float:
    total = 0.0
    for mv in moves:
        if len(mv.pts) > 1:
            total += float(np.hypot(*np.diff(mv.pts, axis=0).T).sum())
    return total


class TurtleAnimator:
    """Moves a Turtle along pre-computed moves, a fixed *distance* per call.

    ``advance(budget)`` is meant to be called from a timer. Because the budget
    is a distance (not a point count) the pen moves at an even pace along both
    long straight lines and dense curves. It never blocks the event loop.
    """

    def __init__(self, screen, pen, moves: List[Move], start_pos: np.ndarray, pen_width: float):
        self.screen, self.pen = screen, pen
        self.moves, self.start_pos, self.pen_width = moves, np.asarray(start_pos, float), pen_width
        self.total_length = max(_moves_length(moves), 1e-9)
        self.reset()

    # ---- state ----------------------------------------------------------
    def reset(self) -> None:
        """Clear the canvas and put the pen at the start of the drawing."""
        self.move_i, self.pt_i = 0, 1
        self.pos = self.start_pos.copy()
        self.done_length = 0.0
        self.polylines: List[np.ndarray] = []          # finished pen-down polylines
        self._current: List[np.ndarray] = []           # polyline being drawn
        self._pen_down = False
        self._heading_vec: Optional[Tuple[float, float]] = None

        pen = self.pen
        pen.clear()
        pen.penup()
        pen.goto(float(self.pos[0]), float(self.pos[1]))
        pen.setheading(90)
        pen.pensize(self.pen_width)
        pen.showturtle()
        self.screen.update()

    @property
    def finished(self) -> bool:
        return self.move_i >= len(self.moves)

    @property
    def progress(self) -> float:
        return 1.0 if self.finished else min(self.done_length / self.total_length, 1.0)

    def snapshot(self) -> List[np.ndarray]:
        """Everything drawn so far (used for saving a partial drawing)."""
        polys = list(self.polylines)
        if len(self._current) > 1:
            polys.append(np.array(self._current))
        return polys

    # ---- pen helpers ------------------------------------------------------
    def _set_pen(self, down: bool) -> None:
        if down == self._pen_down:
            return
        if down:
            self._current = [self.pos.copy()]
            self.pen.pendown()
        else:
            if len(self._current) > 1:
                self.polylines.append(np.array(self._current))
            self._current = []
            self.pen.penup()
        self._pen_down = down

    def _goto(self, target: np.ndarray) -> None:
        d = target - self.pos
        if abs(d[0]) + abs(d[1]) > 1e-9:
            self._heading_vec = (float(d[0]), float(d[1]))
        self.pen.goto(float(target[0]), float(target[1]))
        self.pos = np.array(target, dtype=float)
        if self._pen_down:
            self._current.append(self.pos.copy())

    # ---- the animation step ---------------------------------------------------
    def advance(self, budget: float) -> None:
        """Draw for ``budget`` pixels of pen travel (pen-up glides cost less)."""
        while budget > 1e-9 and not self.finished:
            mv = self.moves[self.move_i]
            self._set_pen(mv.pen_down)

            if mv.is_dot:
                self.pen.dot(max(self.pen_width, 1.0))
                self.polylines.append(mv.pts.copy())
                self.move_i, self.pt_i = self.move_i + 1, 1
                budget -= 1.0
                continue

            speedup = 1.0 if mv.pen_down else PEN_UP_SPEED_FACTOR
            while budget > 1e-9 and self.pt_i < len(mv.pts):
                target = mv.pts[self.pt_i]
                delta = target - self.pos
                dist = math.hypot(delta[0], delta[1])
                cost = dist / speedup
                if cost <= budget:                               # reach the next vertex
                    self._goto(target)
                    budget -= cost
                    self.done_length += dist
                    self.pt_i += 1
                else:                                            # stop part-way along the segment
                    frac = budget * speedup / dist
                    self._goto(self.pos + delta * frac)
                    self.done_length += dist * frac
                    budget = 0.0
            if self.pt_i >= len(mv.pts):
                self.move_i, self.pt_i = self.move_i + 1, 1

        if self.finished:
            self._set_pen(False)
            self.pen.hideturtle()
        elif self._heading_vec is not None:                      # cursor faces travel direction
            self.pen.setheading(math.degrees(math.atan2(self._heading_vec[1], self._heading_vec[0])))


# =============================================================================
# Stage 6 - Save the finished drawing as PNG (Pillow)
# =============================================================================
def render_png(polylines: List[np.ndarray], canvas_size: Tuple[int, int], pen_width: float,
               out_path: str, scale: int = 2) -> None:
    """Rasterise the traced polylines to a PNG, matching the Turtle canvas.

    Drawn at 2x extra resolution and down-sampled for smooth anti-aliased lines,
    with round caps/joins like the Turtle's own pen.
    """
    cw, ch = canvas_size
    f = scale * 2                                               # internal supersampling
    img = Image.new("RGB", (cw * f, ch * f), "white")
    draw = ImageDraw.Draw(img)
    width = max(1, int(round(pen_width * f)))
    r = width / 2.0

    def to_px(p):
        return ((p[0] + cw / 2.0) * f, (ch / 2.0 - p[1]) * f)   # Turtle -> pixel coordinates

    for poly in polylines:
        pts = [to_px(p) for p in poly]
        if len(pts) > 1:
            draw.line(pts, fill="black", width=width, joint="curve")
        for (x, y) in (pts[0], pts[-1]):                         # round caps
            draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
    img = img.resize((cw * scale, ch * scale), Image.LANCZOS)
    img.save(out_path)


# =============================================================================
# User interface (Tk panel + embedded Turtle canvas)
# =============================================================================
class TracerApp:
    EMPTY, ANALYZING, READY, RUNNING, PAUSED, DONE = range(6)

    def __init__(self, root: "tk.Tk", initial_image: Optional[str] = None, method: str = "auto"):
        self.root = root
        self.state = self.EMPTY
        self.token = 0                         # invalidates stale timer callbacks
        self.job_id = 0                        # invalidates stale analysis results
        self.closed = False
        self.image_path: Optional[str] = None
        self.animator: Optional[TurtleAnimator] = None
        self.results: "queue.Queue" = queue.Queue()
        self.preview_photo = None
        self.pen_width = 3.0

        root.title("Turtle Image Tracer")
        root.resizable(False, False)
        self.method_var = tk.StringVar(value=method.capitalize())
        self.speed_var = tk.IntVar(value=DEFAULT_SPEED)
        self._build_ui()
        self._build_turtle()
        self._refresh_controls()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if initial_image:
            root.after(150, lambda: self.load(initial_image))

    # ---- construction -------------------------------------------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        panel = ttk.Frame(main)
        panel.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(panel, text="Turtle Image Tracer", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Line drawing  ->  animated Turtle", foreground="#666").pack(anchor="w", pady=(0, 8))

        self.btn_select = ttk.Button(panel, text="Select image...", command=self.on_select)
        self.btn_select.pack(fill="x")

        frame = ttk.LabelFrame(panel, text="Original image (preview)")
        frame.pack(fill="x", pady=8)
        self.preview = tk.Canvas(frame, width=240, height=170, bg="#f3f3f3", highlightthickness=0)
        self.preview.pack(padx=4, pady=4)
        self.preview.create_text(120, 85, text="No image selected", fill="#999", tags="hint")

        ttk.Label(panel, text="Line detection:").pack(anchor="w")
        self.cmb_method = ttk.Combobox(panel, textvariable=self.method_var, state="readonly",
                                       values=("Auto", "Otsu", "Adaptive"))
        self.cmb_method.pack(fill="x", pady=(0, 8))
        self.cmb_method.bind("<<ComboboxSelected>>", lambda _e: self.on_method_changed())

        self.btn_start = ttk.Button(panel, text="Start drawing", command=self.on_start)
        self.btn_start.pack(fill="x", pady=2)
        self.btn_pause = ttk.Button(panel, text="Pause", command=self.on_pause_resume)
        self.btn_pause.pack(fill="x", pady=2)
        self.btn_restart = ttk.Button(panel, text="Restart animation", command=self.on_restart)
        self.btn_restart.pack(fill="x", pady=2)

        self.scale = tk.Scale(panel, from_=1, to=100, orient="horizontal", variable=self.speed_var,
                              label="Drawing speed", showvalue=True, length=240)
        self.scale.pack(fill="x", pady=(8, 2))

        self.btn_save = ttk.Button(panel, text="Save drawing as PNG...", command=self.on_save)
        self.btn_save.pack(fill="x", pady=(6, 2))

        self.progress = ttk.Progressbar(panel, maximum=100)
        self.progress.pack(fill="x", pady=(10, 2))
        self.status = ttk.Label(panel, text="", wraplength=240, justify="left")
        self.status.pack(anchor="w", pady=4)

        border = tk.Frame(main, bg="#aaaaaa", padx=1, pady=1)       # thin outline around the canvas
        border.pack(side="left")
        self.canvas = tk.Canvas(border, width=CANVAS_W, height=CANVAS_H, bg="white",
                                highlightthickness=0, bd=0)         # no border: keeps (0,0) centred
        self.canvas.pack()

    def _build_turtle(self) -> None:
        self.screen = turtle.TurtleScreen(self.canvas)       # Turtle drawing on a plain Tk canvas
        self.screen.bgcolor("white")
        self.screen.tracer(0)                  # we redraw once per timer tick (smooth + fast)
        self.pen = turtle.RawTurtle(self.screen)
        self.pen.setundobuffer(None)           # don't remember every move (saves memory)
        self.pen.speed(0)
        self.pen.shape("turtle")
        self.pen.shapesize(1.1, 1.1, 1)
        self.pen.color("black", "#2fa36b")     # black ink, green turtle body
        self.pen.penup()
        self.pen.hideturtle()
        self.hint_id = self.canvas.create_text(0, 0, text="Select a black-and-white line drawing to begin",
                                               fill="#999", font=("TkDefaultFont", 13))

    # ---- state / controls --------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status.config(text=text)

    def _refresh_controls(self) -> None:
        s = self.state
        busy = s == self.ANALYZING
        on = lambda flag: "normal" if flag else "disabled"
        self.btn_select.config(state=on(not busy))
        self.cmb_method.config(state="disabled" if busy else "readonly")
        self.btn_start.config(state=on(s == self.READY))
        self.btn_pause.config(state=on(s in (self.RUNNING, self.PAUSED)),
                              text="Resume" if s == self.PAUSED else "Pause")
        self.btn_restart.config(state=on(s in (self.RUNNING, self.PAUSED, self.DONE)))
        self.btn_save.config(state=on(s in (self.RUNNING, self.PAUSED, self.DONE)))

    # ---- loading + analysis (worker thread keeps the UI responsive) -------------------------
    def on_select(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a black-and-white line drawing",
            filetypes=[("Images", " ".join("*" + e for e in ALLOWED_EXTENSIONS)), ("All files", "*.*")])
        if path:
            self.load(path)

    def on_method_changed(self) -> None:
        if self.image_path:
            self.load(self.image_path)

    def load(self, path: str) -> None:
        try:                                   # validate first: a bad file must not disturb
            pil = load_image(path)             # whatever is currently running
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return
        self._stop_animation()
        self.image_path = path
        self._show_preview(pil)
        self.pen.clear()
        self.pen.hideturtle()
        self.canvas.itemconfig(self.hint_id, text="Analysing image...")
        self.animator = None
        self.progress["value"] = 0
        self.state = self.ANALYZING
        self._refresh_controls()
        self._set_status(f"Analysing {os.path.basename(path)} ...")

        self.job_id += 1
        job, method = self.job_id, self.method_var.get().lower()

        def work():
            try:
                self.results.put((job, "ok", trace_image(pil, method)))
            except Exception as exc:                          # report any failure in the UI
                self.results.put((job, "error", exc))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(50, self._poll_results)

    def _poll_results(self) -> None:
        if self.closed:
            return
        try:
            job, kind, payload = self.results.get_nowait()
        except queue.Empty:
            if self.state == self.ANALYZING:
                self.root.after(50, self._poll_results)
            return
        if job != self.job_id:                                # stale result, ignore
            self.root.after(50, self._poll_results)
            return

        if kind == "error":
            self.state = self.EMPTY
            self.canvas.itemconfig(self.hint_id, text="Select a black-and-white line drawing to begin")
            self._refresh_controls()
            self._set_status("Could not analyse the image.")
            messagebox.showerror("No drawing found", str(payload))
            return

        drawing: TracedDrawing = payload
        path = map_to_turtle(drawing)
        moves = build_moves(path)
        self.animator = TurtleAnimator(self.screen, self.pen, moves, path.strokes[0][0], path.pen_width)
        self.pen.hideturtle()
        self.canvas.itemconfig(self.hint_id, text="")
        self.pen_width = path.pen_width
        self.state = self.READY
        self._refresh_controls()
        points = sum(len(s) for s in path.strokes)
        self._set_status(f"Ready: {len(path.strokes)} strokes, {points} points.\nPress 'Start drawing'.")

    def _show_preview(self, pil: Image.Image) -> None:
        thumb = pil.copy()
        thumb.thumbnail((240, 170))
        self.preview_photo = ImageTk.PhotoImage(thumb)             # keep a reference!
        self.preview.delete("all")
        self.preview.create_image(120, 85, image=self.preview_photo)

    # ---- animation control -------------------------------------------------------------------
    def _stop_animation(self) -> None:
        self.token += 1                                           # cancels any pending tick

    def _begin(self) -> None:
        self._stop_animation()
        self.animator.reset()
        self.progress["value"] = 0
        self.state = self.RUNNING
        self._refresh_controls()
        self._set_status("Drawing...")
        self._schedule(self.token, 0)

    def _schedule(self, token: int, delay: int) -> None:
        self.screen.ontimer(lambda: self._tick(token), max(1, delay))

    def _tick(self, token: int) -> None:
        if self.closed or token != self.token or self.state != self.RUNNING:
            return
        started = time.perf_counter()
        try:
            self.animator.advance(self.speed_var.get() * PX_PER_TICK_PER_SPEED)
            self.screen.update()                                  # one redraw per tick
        except (tk.TclError, turtle.Terminator):
            return                                                # window was closed
        self.progress["value"] = self.animator.progress * 100
        if self.animator.finished:
            self.state = self.DONE
            self._refresh_controls()
            self._set_status("Drawing complete. You can save it as a PNG.")
            return
        elapsed = int((time.perf_counter() - started) * 1000)
        self._schedule(token, TICK_MS - elapsed)                  # keep a steady frame rate

    def on_start(self) -> None:
        if self.animator and self.state == self.READY:
            self._begin()

    def on_restart(self) -> None:
        if self.animator:
            self._begin()

    def on_pause_resume(self) -> None:
        if self.state == self.RUNNING:
            self.state = self.PAUSED
            self._stop_animation()
            self._set_status("Paused.")
        elif self.state == self.PAUSED:
            self.state = self.RUNNING
            self._stop_animation()
            self._set_status("Drawing...")
            self._schedule(self.token, 0)
        self._refresh_controls()

    # ---- saving ----------------------------------------------------------------------------------
    def on_save(self) -> None:
        if not self.animator:
            return
        polylines = self.animator.snapshot()
        if not polylines:
            messagebox.showinfo("Nothing to save", "Nothing has been drawn yet.")
            return
        stem = os.path.splitext(os.path.basename(self.image_path or "drawing"))[0]
        out = filedialog.asksaveasfilename(title="Save drawing", defaultextension=".png",
                                           initialfile=f"{stem}_traced.png",
                                           filetypes=[("PNG image", "*.png")])
        if not out:
            return
        try:
            render_png(polylines, (CANVAS_W, CANVAS_H), self.pen_width, out)
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._set_status(f"Saved: {os.path.basename(out)}")

    def on_close(self) -> None:
        self.closed = True
        self.token += 1
        self.root.destroy()


# =============================================================================
# Command line
# =============================================================================
def run_headless(image: str, method: str, out_png: str) -> int:
    """Trace an image and save the result without opening any window."""
    try:
        drawing = trace_image(load_image(image), method)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    path = map_to_turtle(drawing)
    moves = build_moves(path)
    render_png(polylines_from_moves(moves), (CANVAS_W, CANVAS_H), path.pen_width, out_png)
    points = sum(len(s) for s in path.strokes)
    print(f"Traced {len(path.strokes)} strokes / {points} points "
          f"(pen width {path.pen_width:.1f}) -> {out_png}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Trace a line-drawing image with an animated Turtle.")
    parser.add_argument("image", nargs="?", help="image to load on start (PNG, JPG, JPEG, BMP)")
    parser.add_argument("--threshold", choices=("auto", "otsu", "adaptive"), default="auto",
                        help="line detection method (default: auto)")
    parser.add_argument("--export", metavar="OUT.png",
                        help="no UI: trace IMAGE and save the finished drawing to OUT.png")
    args = parser.parse_args(argv)

    if args.export:
        if not args.image:
            parser.error("--export needs an input image")
        return run_headless(args.image, args.threshold, args.export)

    if not GUI_AVAILABLE:
        print(f"Tkinter/turtle is not available ({_GUI_IMPORT_ERROR}).\n"
              "Linux: sudo apt install python3-tk     macOS (Homebrew): brew install python-tk",
              file=sys.stderr)
        return 1
    root = tk.Tk()
    TracerApp(root, args.image, args.threshold)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
