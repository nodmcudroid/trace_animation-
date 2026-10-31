#!/usr/bin/env python3
"""
Generate ``sample_input.png`` - a black-and-white line drawing for testing
the Turtle tracer.

The scene deliberately mixes straight lines, circles, curves, a spiral,
junctions (rays touching a circle, the roof meeting the walls) and several
disconnected objects, so it exercises every stage of the tracing pipeline.

Usage:
    python create_sample_image.py            # writes sample_input.png
    python create_sample_image.py out.png    # custom output name
"""
import math
import sys

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 900, 650      # final image size
SS = 3                        # supersampling factor (gives smooth, anti-aliased lines)
LINE = 3                      # line thickness in final pixels


def make_sample(path: str = "sample_input.png") -> None:
    img = Image.new("L", (WIDTH * SS, HEIGHT * SS), 255)
    d = ImageDraw.Draw(img)
    w = LINE * SS

    def P(x, y):                       # scale a point to the supersampled canvas
        return (x * SS, y * SS)

    def line(*pts):
        d.line([P(*p) for p in pts], fill=0, width=w, joint="curve")

    def circle(cx, cy, r):
        d.ellipse([P(cx - r, cy - r), P(cx + r, cy + r)], outline=0, width=w)

    def rect(x0, y0, x1, y1):
        d.rectangle([P(x0, y0), P(x1, y1)], outline=0, width=w)

    # --- ground ---------------------------------------------------------
    line((30, 520), (870, 520))

    # --- house ----------------------------------------------------------
    rect(300, 300, 560, 520)                                  # walls
    line((275, 305), (430, 180), (585, 305), (275, 305))      # roof (closed triangle)
    line((500, 222), (500, 175), (535, 175), (535, 245))      # chimney
    rect(400, 410, 462, 520)                                  # door
    circle(447, 468, 4)                                       # door knob
    for x0 in (330, 475):                                     # two windows with crosses
        rect(x0, 340, x0 + 60, 400)
        line((x0 + 30, 340), (x0 + 30, 400))
        line((x0, 370), (x0 + 60, 370))

    # --- smoke curls from the chimney -----------------------------------
    smoke = [(518 + 14 * math.sin(t / 9.0), 165 - t) for t in range(0, 95, 3)]
    line(*smoke)

    # --- path to the door -----------------------------------------------
    line((410, 520), (370, 600), (30, 600))
    line((455, 520), (500, 600), (870, 600))

    # --- sun with rays ---------------------------------------------------
    sx, sy, sr = 130, 120, 45
    circle(sx, sy, sr)
    for k in range(12):
        a = k * math.pi / 6
        line((sx + (sr + 10) * math.cos(a), sy + (sr + 10) * math.sin(a)),
             (sx + (sr + 38) * math.cos(a), sy + (sr + 38) * math.sin(a)))

    # --- tree ------------------------------------------------------------
    line((650, 520), (650, 410))
    line((680, 520), (680, 410))
    for cx, cy, r in ((665, 350, 70), (610, 395, 45), (720, 395, 45)):
        circle(cx, cy, r)

    # --- spiral (top right) ----------------------------------------------
    spiral = []
    for i in range(0, 400):
        a = i * 0.045
        r = 3 + 3.1 * a
        spiral.append((775 + r * math.cos(a), 110 + r * math.sin(a)))
    line(*spiral)

    # --- cloud (arcs) -----------------------------------------------------
    for cx, cy, r in ((300, 90, 30), (340, 75, 38), (385, 90, 30)):
        d.arc([P(cx - r, cy - r), P(cx + r, cy + r)], 180, 360, fill=0, width=w)
    line((270, 90), (415, 90))

    # --- two little birds ---------------------------------------------------
    for bx, by in ((590, 55), (640, 90)):
        line(*[(bx + t, by - 14 * math.sin(t / 22.0 * math.pi)) for t in range(0, 23, 2)])
        line(*[(bx + 22 + t, by - 14 * math.sin(t / 22.0 * math.pi)) for t in range(0, 23, 2)])

    # --- a flower on the left -------------------------------------------------
    fx, fy = 110, 380
    line((fx, fy + 20), (fx, 520))
    line((fx, 480), (fx - 30, 455), (fx - 8, 475))            # leaf
    circle(fx, fy, 10)
    for k in range(6):
        a = k * math.pi / 3
        circle(fx + 24 * math.cos(a), fy + 24 * math.sin(a), 14)

    # --- grass tufts -------------------------------------------------------------
    for gx in range(190, 280, 22):
        line((gx - 8, 520), (gx, 495), (gx + 8, 520))

    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    img.save(path)
    print(f"Saved {path} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    make_sample(sys.argv[1] if len(sys.argv) > 1 else "sample_input.png")
