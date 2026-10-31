# Animated Image → Turtle Drawing Tracer

Pick a black-and-white line drawing and watch a Python **Turtle** redraw it stroke by
stroke, like a hand drawing in real time. The Turtle never shows or copies the source
image: every line on the canvas comes from `goto` calls along coordinates extracted
from the picture.

| File | Purpose |
|---|---|
| `turtle_tracer.py` | The application (pipeline + animation + UI) |
| `requirements.txt` | Python dependencies |
| `sample_input.png` | Example line drawing to try |
| `sample_output.png` | What the traced result looks like |
| `create_sample_image.py` | Regenerates `sample_input.png` |

---

## Installation

Requires **Python 3.8+** with Tk (the toolkit behind `turtle`).

```bash
# 1. (optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt
```

`turtle` and `tkinter` are part of the standard library, but some systems ship Tk separately:

| System | If `import tkinter` fails |
|---|---|
| Windows / macOS (python.org installer) | Already included |
| Ubuntu / Debian | `sudo apt install python3-tk` |
| Fedora | `sudo dnf install python3-tkinter` |
| macOS (Homebrew Python) | `brew install python-tk` |

## Running

```bash
python turtle_tracer.py                      # open the app, then click "Select image..."
python turtle_tracer.py sample_input.png     # open the app with an image already loaded
python turtle_tracer.py sample_input.png --export result.png   # no window: trace and save PNG
```

Options: `--threshold auto|otsu|adaptive` (default `auto`).

### Using the app

| Control | What it does |
|---|---|
| **Select image…** | Choose a PNG / JPG / JPEG / BMP. A preview appears and the image is analysed in the background (the window stays responsive). |
| **Start drawing** | Begins the animation. |
| **Pause / Resume** | Freezes and continues the animation exactly where it stopped. |
| **Drawing speed** | Slider 1–100, adjustable at any time, even mid-drawing. |
| **Restart animation** | Clears the canvas and draws again from the beginning. |
| **Save drawing as PNG…** | Saves what has been drawn so far (the complete picture once it finishes) at 2× resolution. |
| **Line detection** | `Auto` (default), `Otsu` (clean scans/exports) or `Adaptive` (photos with shadows or uneven lighting). Changing it re-analyses the image. |

---

## How the image-processing pipeline works

```
 image file ─► 1 Load ─► 2 Preprocess ─► 3 Extract paths ─► 4 Map coordinates ─► 5 Animate ─► 6 Save PNG
              Pillow      OpenCV          skeleton + trace    image → Turtle       timer-driven    Pillow
```

### 1. Load (`load_image`)
Pillow opens the file, applies EXIF rotation, and flattens transparent PNGs onto white
(so black lines on a transparent background work). Missing, corrupt or unreadable files
raise a clear error that the UI shows in a dialog.

### 2. Preprocess (`preprocess`)
1. Convert to **grayscale** and **resize** so the longest side is 720 px (aspect ratio kept).
2. **Gaussian blur** (3×3) to suppress noise and JPEG artefacts.
3. **Threshold** to a binary mask:
   * *Otsu* picks one global threshold automatically.
   * *Adaptive* thresholds each pixel against its neighbourhood (robust to shadows).
   * *Auto* uses Otsu, but switches to Adaptive if Otsu marks more than 15 % of the image as ink.
4. **Invert if necessary.** The border brightness tells whether lines are dark-on-light or
   light-on-dark, so the result always has *lines = white, background = black*.
5. **Remove specks** (connected components under 8 px).

### 3. Extract drawing paths (`extract_paths`)
This is the core of the project.

* **Skeletonisation instead of contours.** A contour of a thick line runs along *both*
  edges, so the Turtle would trace every line twice. Thinning reduces each stroke to its
  1-pixel centre-line (Zhang–Suen algorithm, vectorised in NumPy, so no `opencv-contrib`
  is needed). Each line is drawn once.
* **Pixel graph.** The skeleton becomes a graph of connected pixels. Tiny diagonal
  short-cuts are removed, and little whiskers (≤ one line-width long) that thinning leaves
  at corners are pruned.
* **Ordered tracing.** Paths start at free line ends and walk along the skeleton. At a
  junction the walker continues in the *straightest* direction, so curves and crossings
  flow naturally. Every edge is marked as used and is never traversed twice. Disconnected
  parts become separate strokes; isolated dots are kept as dots.
* **Corner-aware smoothing.** A pixel skeleton is a staircase. Paths are split at sharp
  corners (turns > 40°) and each piece is Gaussian-smoothed with its end points pinned, so
  circles and curves come out smooth while corners stay sharp. Douglas–Peucker
  simplification then removes redundant points (0.15 px tolerance).
* **Human-like order.** Strokes are chained by always continuing with the *nearest* one
  (reversing it if its far end is closer), starting at the top-left. If the next stroke
  begins within 2.5 px of the previous end, the pen stays down, so there is no pointless
  lift.
* **Pen width.** The distance transform measures the original line thickness, and the
  Turtle's pen size is set to match.

### 4. Coordinate mapping (`map_to_turtle`)
The drawing's bounding box is scaled with a **single factor** (so proportions are kept),
centred on the canvas, and **Y is flipped**, because images have their origin at the
top-left with y pointing down while Turtle's origin is the centre with y pointing up.

### 5. Animated drawing (`TurtleAnimator` + `TracerApp`)
* The Turtle runs with `tracer(0)`, and the screen is redrawn **once per timer tick**
  (~60 fps) via `screen.ontimer`. `mainloop` is never blocked, so buttons stay live.
* Each tick moves the pen a fixed **distance** (set by the speed slider), interpolating
  along the path. Speed is therefore even on long straight lines and dense curves alike.
* Moving between disconnected strokes lifts the pen (`penup`) and glides at 4× speed.
* The turtle cursor is visible and rotates to face its direction of travel.
* **Pause** and **Restart** use a token counter that invalidates already-scheduled timer
  callbacks, so a stale tick can never draw after you have paused or restarted.

### 6. Final output (`render_png`)
When finished, the cursor hides and the full drawing stays on screen. **Save** re-renders
the same polylines with Pillow, 2× supersampled with round caps and joins, into a PNG that
matches the Turtle canvas. Unlike a Tk canvas export, this needs no Ghostscript.

---

## Tuning

Constants at the top of `turtle_tracer.py`:

| Constant | Default | Effect |
|---|---|---|
| `CANVAS_W`, `CANVAS_H` | 800 × 620 | Size of the drawing area |
| `WORKING_SIZE` | 720 | Analysis resolution; higher keeps finer detail but is slower |
| `SMOOTH_SIGMA` | 3.0 | Path smoothing strength (0 = raw skeleton) |
| `CORNER_ANGLE` | 40° | Turns sharper than this are kept as corners |
| `SIMPLIFY_EPSILON` | 0.15 | Point-reduction tolerance (lower = more points) |
| `MIN_COMPONENT_AREA` | 8 | Specks smaller than this are ignored |
| `CONNECT_DISTANCE` | 2.5 | Gaps up to this size are drawn rather than jumped |
| `PX_PER_TICK_PER_SPEED` | 0.5 | Speed slider scale |
| `AUTO_MAX_INK` | 0.15 | Auto mode's switch-over to Adaptive |

## Tips and known limitations

* Works best with **clean, dark lines on a light background**. Light-on-dark is detected automatically.
* **Filled black areas** become a single centre-line. The tracer is built for outlines and
  line art, not solid shapes or shaded photos.
* Where many **small circles overlap** (like the flower in the sample), arcs get slightly
  angular at the crossings: a general limit of skeletonising crossing lines.
* Specks smaller than 8 pixels and dangling stubs shorter than one line-width are treated as noise and dropped.
* Very detailed drawings animate slowly at low speeds; raise the slider.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'tkinter'` | Install Tk (table above) |
| "No drawing found" dialog | Try `Adaptive`, or use an image with darker lines |
| Photo has shadows and the result is messy | Choose `Adaptive` |
| Lines look too thick or thin | Use a higher-resolution source, or adjust `WORKING_SIZE` |
| Window too large for a small screen | Reduce `CANVAS_W` / `CANVAS_H` |
