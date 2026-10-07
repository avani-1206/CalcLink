# CalcInk: Architecture & Design

This document explains how CalcInk turns pen strokes into answers on the page, and why it was built this way.

## 1. Pipeline at a glance

```
 pointer events                       main thread                                  worker thread
 ─────────────┐   ┌──────────────────────────────────────────────────────┐   ┌────────────────────────┐
 mouse/pen/   │   │ canvas.js        strokes (vectors, immutable list)   │   │ inference-worker.js    │
 touch  ──────┼──►│   draw @60 FPS ─ undo/redo ─ erasers ─ scratch       │   │                        │
              │   │        │ change                                      │   │ ONNX Runtime Web       │
              │   │        ▼ (350 ms pause, pen up)                      │   │ (WASM)                 │
              │   │ calcink-core.segment()  strokes → lines → symbols    │   │                        │
              │   │ symbol vectors (points + boxes) ─────────────────────┼──►│ rasterizeSymbol()      │
              │   │                                                      │   │  (OffscreenCanvas)     │
              │   │                                        probabilities◄┼───│ pre-trained + bundled  │
              │   │ calcink-core.combinePredictions()  → characters      │   └────────────────────────┘
              │   │ calcink-core.evaluateDocument()   Shunting Yard      │
              │   │ projection.render()  answers / graphs next to "="    │
              │   └──────────────────────────────────────────────────────┘
```

All of this happens on the device. No request ever leaves the browser except the first download of the app's own files.

## 2. Key decisions

### 2.1 Vector strokes are the source of truth

The canvas bitmap is only a *view*. The model is `strokes`: an immutable array of `{id, kind, pts, w, color, alpha, box}`.

- **Undo/redo** keeps references to previous arrays. Stroke objects are shared, so 300 history steps cost kilobytes, not the 8–30 MB per step a full-screen `ImageData` snapshot would cost.
- **The pixel eraser is geometric.** It removes the points under the eraser and splits each stroke into the surviving pieces. What you see is exactly what the recogniser reads. A "transparent eraser stroke" would leave the hidden ink in the data, and a later stroke-erase could make it reappear.
- **Resizing and DPR changes** re-draw the vectors into a backing store of `cssSize × devicePixelRatio`, so lines stay crisp on Retina screens and nothing is lost when the window shrinks.
- **Dark mode** recolours the theme-aware `ink` colour by re-drawing.

### 2.2 Two canvases for 60 FPS drawing

`committed` (offscreen) holds finished strokes. On each `pointermove` the visible canvas does one `drawImage(committed)` plus the single live stroke. The cost per event is constant, no matter how much ink is on the page. `getCoalescedEvents()` gives every sub-frame sample of fast pen movements.

### 2.3 Inference never runs on the main thread

The PS requires a steady 60 FPS while recognition is running. So:

1. Recognition is **debounced** (350 ms after the last change) and **never starts while a pointer is down**.
2. The neural networks run in a **Web Worker**, and so does turning symbols into images: the main thread sends only the stroke points, and the worker draws the model inputs on an `OffscreenCanvas`. If a browser lacks `OffscreenCanvas` in workers, the main thread draws the images and *transfers* them (zero-copy, `postMessage(..., [buffer])`).
3. Only one recognition is in flight at a time. Changes made meanwhile queue exactly one re-run, so the newest state wins.

The e2e test draws continuously and forces a recognition at the same time. It watches two things: frame gaps (`requestAnimationFrame`) and the browser's **Long Tasks** API, which reports any main-thread job longer than 50 ms. Result: zero long tasks, and 100 % of about 390 frames within 20 ms (worst 16.8 ms, one frame at 60 Hz).

### 2.4 From strokes to characters: segmentation

Implemented in `calcink-core.js`, with no DOM, so it is unit-tested.

- **Lines:** strokes whose vertical ranges overlap are joined into one line (union-find).
- **Symbols:** within a line, two strokes form one symbol when the centre of the narrower stroke lies inside the horizontal span of the wider one. This keeps the bars of `=`, the dots of `÷`, both lines of `+`/`×`, and the bar of `5`/`7` together, while neighbouring digits stay apart.
- **Dots:** a tiny blob only joins a neighbour when it floats above or below it (the dots of `÷`). A dot level with a digit stays separate, so `12.5` written with no gaps is not misread as `125` (a bug found by stress-testing; regression test added).
- **Decimal point:** a blob smaller than 15 % of the line height is classified by geometry. A CNN cannot tell a scaled-up dot from an `o`.
- **Side-by-side sums:** a horizontal gap bigger than 1.5 × the line height starts a new expression, so two sums on one row each get an answer.
- **Powers (BODMAS "O"):** after recognition, `lineChars()` checks each symbol against the one before it. A symbol that is clearly smaller (≤ 75 % height), centred in the upper 30 % of its base, and with its bottom above the base's middle is an exponent. `^` is inserted before it, and a raised group like `²⁺¹` gets brackets. The base must be a number, `)` or a variable, so a normal mid-height minus is never mistaken for a power.

### 2.5 Rasterisation must match training

Each symbol is redrawn *from its vectors*, not cropped from the screen, into exactly the format its model was trained on: longest side scaled to `fit` px, centred, fixed pen width, polarity, 4× supersampling and box-filter downsampling. Redrawing makes recognition independent of the user's pen colour, width, opacity and zoom. The shared function `CalcCore.rasterizeSymbol(ctx, sym, spec)` takes any 2D context, so the same code runs in the worker (`OffscreenCanvas`) and on the main thread. The per-model contract (`size`, `fit`, `fitIncludesPen`, `penWidth`, `inkIsWhite`, `layout`) lives in `recognizer.js` (bundled) and in `models/pretrained.json` (pre-trained).

### 2.6 Combining two models

`combinePredictions()` trusts the **pre-trained** model for every symbol. The single exception is when the **bundled** model is ≥ 85 % sure the symbol is one the pre-trained model cannot output (`(`, `)`, `y`). This keeps the PS vocabulary on the stronger model and still allows parentheses, variables and plots.

### 2.7 Evaluation: Shunting Yard (no `eval`)

```
chars ─resolveX─► tokens ─implicit ×─► Shunting Yard ─► RPN ─► stack evaluator ─► {ok | undefined | error}
```

- **`resolveX`:** a handwritten `x` between two operands (`3x5`, `(2)x4`) means multiply; anywhere else it is the variable `x`.
- **Precedence (BODMAS):** `^` 4 (right-associative, so `2^3^2 = 2^9`), unary minus 3, `× ÷` 2, `+ −` 1, with `×÷` and `+−` evaluated left to right. Brackets are handled by the algorithm itself.
- **Unary minus:** a `-` at the start or after an operator/`(` becomes a `neg` operator, right-associative and below `^`, so `−2² = −4` (standard maths) while `2×-3` and `--2` still work.
- **Steps:** while evaluating the RPN, every operation is recorded with its BODMAS letter (`B` if it sits inside brackets). The Steps view shows exactly the order the engine used, e.g. `[B] 2 + 3 = 5 → [O] 5 ^ 2 = 25 → [M] 25 × 2 = 50`. Because it records what really happened, it doubles as a correctness check for judges.
- **Implicit multiplication:** inserted between operand-ending and operand-starting tokens (`2(3)`, `3x`, `xy`).
- **Safety:** every failure path is a `CalcError` that becomes `{status:'error'}`. Division by zero and overflow return `{status:'undefined'}`, shown as **Undefined**. `evaluate()` never throws; a 5,000-case fuzz test checks this.
- **Formatting:** results are rounded to 12 significant digits, so `0.1+0.2` shows `0.3`. Negatives use a real minus sign (−).
- **Lines** are classified as *result* (`…=`), *assignment* (`x=5`, remembered for the lines **below**), *assignment + result* (`x=2+3=`), *plot* (`y=f(x)` while `x` is unknown), *incomplete* (no `=` yet), or *ignored* (`2+2=4`, the user wrote their own answer).

### 2.8 Projection and reactivity

`resultAnchor(eqBox, lineBox)` places the answer `0.35 × fontSize` to the right of the `=`, vertically centred on it, with `fontSize = clamp(0.95 × lineHeight, 18, 72)`. Answers are DOM elements in an overlay, so they get real text rendering, CSS animation and tooltips, and the canvas pixels stay pure user ink.

Each answer is keyed by the **stroke ids of its `=`**. After an edit, only answers whose text changed re-animate. When an `=` is erased, its answer is removed immediately, before the model even runs again.

### 2.9 Scratch-to-erase

`isScratch(pts)` counts direction reversals along x and y, ignoring wiggles under 25 % of the box size, and compares path length with the box diagonal. A scribble needs ≥ 5 reversals **and** a length ≥ 3.5 × the diagonal. Normal glyphs (`8`, `M`, `W`, lines) stay below these thresholds; this is unit-tested. The stroke turns red while you scribble. On release, every stroke with at least half its points under the scribble fades out and is removed, as one undoable step. A scribble on empty paper stays as ink.

### 2.10 Ink and paper

- **Pen ink** is drawn as a filled outline rather than a fixed-width line. The outline's half-width at each point = width × end-taper (sine ease over the first and last ~3 × width px) × pressure factor (0.7–1.3 from `PointerEvent.pressure`, stylus only). Left and right edges are smoothed with quadratic curves and joined by round caps, then filled once, so opacity stays uniform.
- **Paper:** an SVG `feTurbulence` grain, a red margin line and 36 px ruling are layered CSS backgrounds (no images, no network). PNG export redraws the rules and margin.
- **Micro-interactions:** answers "write themselves in" (clip-path reveal and de-blur), erased ink fades out over 300 ms, a scribble turns red before it erases, a haptic tick fires when an answer appears (`navigator.vibrate`, where supported), and there is an optional WebAudio "pop" (off by default; nodes are disconnected after use so nothing leaks). Preferences (dark mode, steps, sound) are stored in `localStorage` inside try/catch.

### 2.11 Offline

- `sw.js` pre-caches the app shell and models, and also caches ONNX Runtime, whether it was loaded from `vendor/ort/` or the CDN. Requests are served cache-first.
- ONNX Runtime is **vendored** by `npm install` (`scripts/vendor-ort.js`), so a deployed site never depends on a CDN.
- If ONNX Runtime cannot load at all, the worker falls back to a small built-in **ONNX interpreter** (~150 lines of JS). It decodes the `.onnx` protobuf (nodes + weights) and runs the ten operators our two models use (Transpose, Conv, Relu, BatchNormalization, MaxPool, GlobalAveragePool, Flatten, Gemm, Softmax, Identity). Its outputs match onnxruntime to within 1e-6 (`tests/engine.test.js`). It is slower (about 130 ms per symbol for the pre-trained CNN), but it runs in the worker, so drawing is unaffected.
- `dist/calcink-standalone.html` inlines CSS, scripts, the worker source and the models (base64), so it works when double-clicked from disk.

## 3. Model choice

The PS asks for an existing open-source pre-trained model that runs client-side. Requirements: covers digits and `+ − × ÷ =`; small enough for instant load on a phone; permissive licence; convertible to ONNX.

| Option | Verdict |
|---|---|
| **altynbk/handwritten-math-recognition CNN** (chosen) | MIT, exactly the PS vocabulary (15 classes), trained on about 8k real handwritten symbols with augmentation, about 390k parameters (≈1.5 MB), single-symbol classifier that fits our segmentation pipeline, Keras, so it converts to ONNX with `tf2onnx` |
| ONNX Model Zoo MNIST (`mnist-8`) | Apache-2.0 and tiny, but **digits only**: no operators |
| Full expression recognisers (CROHME / MathWriting encoder-decoders, TrOCR-style) | Handle 2-D layout, but are tens to hundreds of MB, need autoregressive decoding (slow on CPU/WASM), and are overkill for linear arithmetic |
| Cloud OCR (Mathpix, Google Vision) | Forbidden by the PS |

**Conversion without TensorFlow.** The model ships as a Keras 3 `.keras` file (a zip of `config.json` + `model.weights.h5`). TensorFlow does not support Python 3.13, so `tools/convert_pretrained.py` avoids it: `tools/minih5.py` reads the HDF5 weights directly (superblock → B-tree → symbol nodes → datasets), the converter rebuilds the layer graph from `config.json` as ONNX, and it verifies the result against an independent NumPy implementation of the Keras layers. On rendered test glyphs the converted model scores 98.8 % with the repo's own preprocessing, and 99.6 % through CalcInk's rasteriser with pen width 4 (chosen by a sweep over 2–5).

The **bundled model** exists because the pre-trained model cannot output `( ) y`, which parentheses, variables and plots need. It was trained on synthetic strokes (`tools/bundled-model/synth.py`) with a NumPy CNN (`train.py`) and exported by a tiny hand-written ONNX writer (`onnx_writer.py`). Its outputs were cross-checked against onnxruntime to within 1e-7.

## 4. Testing strategy

| Layer | Tool | Covers |
|---|---|---|
| Parser & evaluator | `node --test` | Full BODMAS table incl. powers, right-associative `^`, `−2² = −4`, decimals, negatives, ÷0 → Undefined, malformed input, 5,000-case fuzz (never throws), implicit ×, x/× disambiguation, variables, formatting, step trace |
| Geometry ("coordinate conversion") | `node --test` | Line/symbol segmentation (`=`, `÷`, `+` grouping), squeezed decimal points, side-by-side sums, superscript → power detection, decimal detection, answer anchor coordinates, font clamping |
| Gestures & plots | `node --test` | Scratch detection vs `8`/`M`/lines, scratch targets, reversal counting, plot sampling with gaps, model combining |
| Inference engine | `node --test` | Built-in ONNX interpreter vs onnxruntime outputs for both models; pre-trained model metadata |
| Build | `node --test` | Site output, standalone has no external scripts, SW pre-cache list matches `index.html`, no `eval`/`new Function` in app code |
| Whole app | Playwright (`tests/e2e/browser_e2e.py`, 20 checks) | Real drawing → answers, reactive edit (18+4×3 → 18+4×5 = 38), zero long tasks + frame budget while recognising, undo/redo, both erasers, clear, scratch-to-erase, plotting, handwritten power 2³+1, side-by-side sums, Steps view, **heap stable over 480 extra strokes**, offline reload, **touch input on a DPR-2 tablet**, standalone file |

## 5. Performance notes

- Drawing: O(1) work per pointer event (one blit + one live stroke).
- Recognition of about 40 symbols: segmentation takes under 1 ms on the main thread; rasterisation and inference (5–30 ms, depending on the engine) run in the worker.
- Memory: history shares stroke objects and is capped at 300 steps; answer, plot and steps elements are removed with their `=`; worker requests are deleted from the pending map when answered; audio nodes are disconnected after each sound. Measured: the JS heap stays flat (3.5 MB → 3.5 MB) after 480 more strokes.
- Robustness found by stress tests: varied handwriting sizes (22–56 px), slant (−0.15 to 0.25) and near-zero spacing. 8/8 mixed lines read correctly after the squeezed-decimal fix.
- Recognition re-reads the whole page each time. This is simple and correct (variables flow between lines), and it is cheap at notebook scale. Per-line caching would be the next optimisation.
