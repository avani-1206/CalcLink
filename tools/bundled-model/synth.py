"""
Synthetic handwritten-symbol generator for CalcInk.

Why synthetic?  The sandbox this was built in has no access to Hugging Face /
Zenodo (where HASYv2 / CROHME live), so instead we "fake" handwriting:
every symbol has a few hand-designed stroke templates, and each training
sample is that template bent, slanted, wobbled and re-drawn with a random pen
width.  Because the browser also renders the user's *strokes* (not a photo),
the training images look very much like what the app will feed the model.

The rendering here MUST match `rasterizeSymbol()` in calcink.html:
  - fit the symbol's bounding box so its longest side is 20 px
  - centre it in a 28x28 image, white ink on black, values 0..1
"""
import math
import numpy as np
import cv2

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '+', '-', 'x', '÷', '=', '(', ')', 'y']

SIZE, FIT = 28, 20


# ── tiny helpers to describe strokes ────────────────────────────────────────
def arc(cx, cy, rx, ry, a0, a1, n=24):
    """Points on an ellipse from angle a0 to a1 (degrees, 0 = right, 90 = down)."""
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def line(*pts):
    return np.array(pts, dtype=float)


def cat(*parts):
    return np.concatenate(parts, 0)


def fig8(cx, cy, rx, ry, n=60):
    t = np.linspace(0, 2 * math.pi, n)
    return np.stack([cx + rx * np.sin(2 * t) * 0.9, cy - ry * np.cos(t)], 1)


def dot(x, y):
    return np.array([[x, y]], float)


# Each template = list of strokes; each stroke = (N,2) array, y grows downward.
def templates(c, r):
    """Return one randomly chosen template for class c (r = np.random.Generator)."""
    u = r.uniform
    if c == '0':
        return [[arc(.3, .5, .27, .5, -90 + u(-20, 20), 270 + u(-10, 30))]]
    if c == '1':
        v = r.integers(3)
        stem = line((.3, 0), (.3 + u(-.05, .05), 1))
        if v == 0:
            return [[stem]]
        flag = line((.3 - u(.12, .22), u(.12, .25)), (.3, 0))
        if v == 1:
            return [[cat(flag, stem[1:])]]
        return [[cat(flag, stem[1:])], [line((.12, 1), (.48, 1))]]
    if c == '2':
        if r.integers(2):
            return [[cat(arc(.3, .27, .26, .26, 190, 385), line((.53, .38), (.04, 1), (.58, 1)))]]
        return [[cat(arc(.3, .25, .25, .24, 180, 340), line((.52, .2), (.05, 1), (.6, 1)))]]
    if c == '3':
        if r.integers(2):
            return [[cat(arc(.28, .25, .25, .25, 200, 450), arc(.28, .75, .28, .25, 270, 520))]]
        return [[cat(line((.05, 0), (.52, 0), (.22, .4)), arc(.27, .7, .28, .3, 260, 520))]]
    if c == '4':
        v = r.integers(3)
        if v == 0:
            return [[line((.42, 0), (0, .66), (.6, .66))], [line((.44, .15), (.44, 1))]]
        if v == 1:
            return [[line((.42, 1), (.42, 0), (0, .66), (.6, .66))]]
        return [[line((.05, 0), (.03, .6), (.6, .6))], [line((.45, 0), (.45, 1))]]
    if c == '5':
        body = cat(line((.1, 0), (.07, .45)), arc(.27, .7, .28, .28, 230, 480))
        if r.integers(2):
            return [[body], [line((.1, 0), (.58, 0))]]
        return [[cat(line((.58, 0), (.1, 0)), body[1:])]]
    if c == '6':
        return [[cat(arc(.55, .6, .5, .6, 255, 180), arc(.3, .72, .25, .27, 180, 540))]]
    if c == '7':
        s = [line((0, 0), (.6, 0), (.18 + u(-.05, .05), 1))]
        if r.integers(3) == 0:
            return [s, [line((.15, .5), (.55, .5))]]
        return [s]
    if c == '8':
        if r.integers(3) == 0:
            return [[arc(.3, .24, .22, .24, 0, 360)], [arc(.3, .74, .28, .26, -90, 270)]]
        return [[fig8(.3, .5, .3, .5)]]
    if c == '9':
        loop = arc(.3, .27, .25, .27, 0, -360)
        if r.integers(2):
            return [[cat(loop, line((.55, .27), (.5, 1)))]]
        return [[cat(loop, line((.55, .27), (.55, .8)), arc(.4, .8, .15, .2, 0, 120))]]
    if c == '+':
        h, v = line((0, .5), (1, .5)), line((.5, 0), (.5, 1))
        return [[h], [v]] if r.integers(2) else [[v], [h]]
    if c == '-':
        return [[line((0, .5), (1, .5 + u(-.04, .04)))]]
    if c == 'x':
        if r.integers(4) == 0:  # cursive x: ")(" back to back
            return [[arc(.15, .5, .35, .5, -70, 70)], [arc(.85, .5, .35, .5, 110, 250)]]
        w = u(.7, 1.0)
        return [[line((0, 0), (w, 1))], [line((w, 0), (0, 1))]]
    if c == '÷':
        g = u(.25, .4)
        return [[line((0, .5), (1, .5))], [dot(.5, .5 - g)], [dot(.5, .5 + g)]]
    if c == '=':
        g = u(.26, .5)
        return [[line((0, .5 - g / 2), (1, .5 - g / 2))],
                [line((u(-.1, .1), .5 + g / 2), (1 + u(-.1, .1), .5 + g / 2))]]
    if c == '(':
        return [[arc(.6, .5, .6 * u(.6, 1.2), .5, 245, 115)]]
    if c == ')':
        return [[arc(-.0, .5, .6 * u(.6, 1.2), .5, -65, 65)]]
    if c == 'y':
        if r.integers(2):
            return [[line((0, 0), (.3, .5))], [line((.6, 0), (.05, 1))]]
        return [[cat(line((0, 0), (0, .3)), arc(.2, .3, .2, .2, 180, 0)[1:], line((.4, 0), (.4, .8)),
                     arc(.2, .8, .2, .2, 0, 150))]]
    raise ValueError(c)


# ── augmentation ────────────────────────────────────────────────────────────
def resample(s, step=0.03):
    if len(s) < 2:
        return s
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(s, axis=0).T))]
    if d[-1] < 1e-9:
        return s[:1]
    n = max(2, int(d[-1] / step) + 1)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])], 1)


def wobble(s, r, amp):
    """Smooth low-frequency noise — imitates a shaky hand."""
    if len(s) < 3:
        return s + r.normal(0, amp / 2, s.shape)
    n = r.normal(0, amp, s.shape)
    k = max(3, len(s) // 4) | 1
    ker = np.ones(k) / k
    n = np.stack([np.convolve(n[:, i], ker, 'same') for i in range(2)], 1) * math.sqrt(k)
    return s + n


def augment(strokes, r):
    strokes = [resample(np.asarray(s[0] if isinstance(s, list) else s, float)) for s in strokes]
    rot = math.radians(r.uniform(-12, 12))
    shear = r.uniform(-.3, .3)
    sx, sy = r.uniform(.8, 1.2), r.uniform(.8, 1.2)
    M = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]) @ \
        np.array([[1, shear], [0, 1]]) @ np.diag([sx, sy])
    amp = r.uniform(0, .02)
    out = []
    for s in strokes:
        s = wobble(s, r, amp) @ M.T
        s = s + r.normal(0, .012, (1, 2))  # each stroke lands a bit off
        out.append(s)
    return out


# ── rasterise (same maths as the JS side) ───────────────────────────────────
def render(strokes, width=2.2, ss=4):
    allp = np.concatenate(strokes, 0)
    mn, mx = allp.min(0), allp.max(0)
    span = max((mx - mn).max(), 1e-6)
    scale = FIT / span
    ctr = (mn + mx) / 2
    img = np.zeros((SIZE * ss, SIZE * ss), np.uint8)
    th = max(1, int(round(width * ss)))
    for s in strokes:
        p = ((s - ctr) * scale + SIZE / 2) * ss
        if len(p) == 1:
            cv2.circle(img, tuple(np.round(p[0]).astype(int)), th // 2 + 1, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [np.round(p).astype(np.int32)], False, 255, th, cv2.LINE_AA)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.


def make_dataset(n_per_class, seed):
    r = np.random.default_rng(seed)
    X = np.zeros((n_per_class * len(CLASSES), 1, SIZE, SIZE), np.float32)
    Y = np.zeros(n_per_class * len(CLASSES), np.int64)
    i = 0
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X[i, 0] = render(augment(templates(c, r), r), width=r.uniform(1.3, 3.2))
            Y[i] = ci
            i += 1
    return X, Y


if __name__ == '__main__':
    X, Y = make_dataset(10, 0)
    grid = X[:, 0].reshape(len(CLASSES), 10, SIZE, SIZE).transpose(0, 2, 1, 3).reshape(len(CLASSES) * SIZE, 10 * SIZE)
    cv2.imwrite('samples.png', cv2.resize((grid * 255).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print('ok', X.shape)
