"""
Install the open-source pre-trained recogniser into CalcInk (one-time step).
NO TensorFlow needed — only Python 3 + NumPy (+ onnxruntime, optional, for checks).

Model : "CNN (augmented)" from altynbk/handwritten-math-recognition (MIT licence)
        https://github.com/altynbk/handwritten-math-recognition
        Keras CNN: 4 × [Conv3×3+ReLU → BatchNorm → MaxPool] (32/64/128/256 filters)
        → GlobalAveragePooling → Dropout → Dense(15, softmax)
        Input 64×64×1, light strokes on dark background, values 0..1,
        symbol cropped to its ink bounding box + 15 % margin, padded square.
        Classes: 0-9, add, div, eq, mul, sub  (folder names, sorted)

Usage
    python tools/convert_pretrained.py                       # looks for the zip in ~/Downloads, or downloads it
    python tools/convert_pretrained.py path/to/handwritten-math-recognition-main.zip

What it does
  1. opens the repository zip (GitHub → Code → Download ZIP)
  2. reads models/cnn_aug.keras  (a zip: config.json + model.weights.h5)
  3. reads the weights with tools/minih5.py (tiny pure-Python HDF5 reader)
  4. writes the same network as ONNX            → models/pretrained.onnx
  5. checks ONNX output == a NumPy re-implementation of the Keras layers
  6. measures accuracy on the repository's own images (if the zip has data/)
  7. writes the input contract + class mapping  → models/pretrained.json
"""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'bundled-model'))
from minih5 import MiniH5                    # noqa: E402
import onnx_writer as ow                     # noqa: E402

REPO = 'https://github.com/altynbk/handwritten-math-recognition'
ZIP_URL = 'https://codeload.github.com/altynbk/handwritten-math-recognition/zip/refs/heads/main'
OUT = HERE.parent / 'models'
CHAR = {'add': '+', 'sub': '-', 'mul': 'x', 'div': '÷', 'eq': '='}   # 'x' = "×" (CalcCore.resolveX decides)


# ── 1. get the repository zip ──────────────────────────────────────────────
def open_repo_zip(arg):
    candidates = [Path(arg)] if arg else [
        Path.home() / 'Downloads' / 'handwritten-math-recognition-main.zip',
        Path('handwritten-math-recognition-main.zip'),
    ]
    for c in candidates:
        if c.exists():
            print('using', c)
            return zipfile.ZipFile(c)
    print('downloading', ZIP_URL)
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZIP_URL, timeout=120).read()))


def find(z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if not names:
        sys.exit(f'{suffix} not found in the zip')
    return names[0]


# ── 2+3. read the Keras model ──────────────────────────────────────────────
def load_keras(keras_bytes):
    k = zipfile.ZipFile(io.BytesIO(keras_bytes))
    config = json.loads(k.read('config.json'))
    weights = dict(MiniH5(k.read('model.weights.h5')).walk())
    layers = config['config']['layers']
    stored = {}
    for path, arr in weights.items():                # e.g. "layers/conv2d_1/vars/0"
        parts = path.split('/')
        if parts[0] == 'layers' and 'vars' in parts:
            name = parts[parts.index('vars') - 1]
            stored.setdefault(name, {})[int(parts[-1])] = arr
    # Keras 3 stores weights under fresh per-type names (conv2d, conv2d_1, …) in
    # layer order, which can differ from the names in config.json (conv2d_4, …).
    # So match the k-th layer of each type in config to the k-th stored group.
    snake = lambda c: {'Conv2D': 'conv2d', 'BatchNormalization': 'batch_normalization', 'Dense': 'dense'}.get(c)
    def order(name):
        base, _, n = name.rpartition('_')
        return (base, int(n)) if n.isdigit() and base else (name, 0)
    groups = {}
    for name in stored:
        base, n = order(name)
        groups.setdefault(base, []).append((n, name))
    by_layer, seen = {}, {}
    for L in layers:
        base = snake(L['class_name'])
        if not base:
            continue
        k = seen.get(base, 0); seen[base] = k + 1
        candidates = sorted(groups.get(base, []))
        if k >= len(candidates):
            sys.exit(f'no stored weights for layer {L["config"]["name"]}')
        by_layer[L['config']['name']] = stored[candidates[k][1]]
    return layers, by_layer, weights


# ── 4. build the ONNX graph layer by layer ─────────────────────────────────
def attr_float(name, v):
    return ow.f_bytes(1, name) + ow.f_float(2, v) + ow.f_int(20, 1)


def to_onnx(layers, W, size, n_classes, path):
    nodes, inits = [], []
    cur = 'nhwc_input'
    nodes.append(ow.node('Transpose', [cur], ['x0'], [ow.attr_ints('perm', [0, 3, 1, 2])]))
    cur, i = 'x0', 0
    plan = []                                       # for the NumPy reference
    for L in layers:
        cls, cfg, name = L['class_name'], L['config'], L['config']['name']
        v = W.get(name, {})
        i += 1
        out = f'x{i}'
        if cls == 'InputLayer' or cls == 'Dropout':
            i -= 1; continue
        if cls == 'Conv2D':
            k = v[0]; b = v[1] if cfg.get('use_bias', True) else np.zeros(k.shape[-1], np.float32)
            kh, kw = k.shape[:2]
            if tuple(cfg.get('strides', (1, 1))) != (1, 1):
                sys.exit('only stride-1 convolutions are supported')
            pads = [(kh - 1) // 2, (kw - 1) // 2, kh - 1 - (kh - 1) // 2, kw - 1 - (kw - 1) // 2] if cfg['padding'] == 'same' else [0, 0, 0, 0]
            inits += [(f'{name}_W', k.transpose(3, 2, 0, 1)), (f'{name}_b', b)]
            nodes.append(ow.node('Conv', [cur, f'{name}_W', f'{name}_b'], [out],
                                 [ow.attr_ints('kernel_shape', [kh, kw]), ow.attr_ints('pads', pads)]))
            plan.append(('conv', k, b, cfg['padding']))
            cur = out
            act = cfg.get('activation', 'linear')
        elif cls == 'BatchNormalization':
            n = v[max(v)].shape[0]
            idx = 0
            gamma = v[idx] if cfg.get('scale', True) else np.ones(n, np.float32); idx += cfg.get('scale', True)
            beta = v[idx] if cfg.get('center', True) else np.zeros(n, np.float32); idx += cfg.get('center', True)
            mean, var = v[idx], v[idx + 1]
            eps = float(cfg.get('epsilon', 1e-3))
            inits += [(f'{name}_g', gamma), (f'{name}_be', beta), (f'{name}_m', mean), (f'{name}_v', var)]
            nodes.append(ow.node('BatchNormalization', [cur, f'{name}_g', f'{name}_be', f'{name}_m', f'{name}_v'], [out],
                                 [attr_float('epsilon', eps)]))
            plan.append(('bn', gamma, beta, mean, var, eps))
            cur, act = out, 'linear'
        elif cls == 'MaxPooling2D':
            p = cfg.get('pool_size', (2, 2)); s = cfg.get('strides') or p
            if cfg.get('padding', 'valid') != 'valid':
                sys.exit('only valid max-pooling is supported')
            nodes.append(ow.node('MaxPool', [cur], [out], [ow.attr_ints('kernel_shape', list(p)), ow.attr_ints('strides', list(s))]))
            plan.append(('pool', tuple(p)))
            cur, act = out, 'linear'
        elif cls == 'GlobalAveragePooling2D':
            nodes.append(ow.node('GlobalAveragePool', [cur], [out + 'g']))
            nodes.append(ow.node('Flatten', [out + 'g'], [out], [ow.attr_int('axis', 1)]))
            plan.append(('gap',))
            cur, act = out, 'linear'
        elif cls == 'Dense':
            k = v[0]; b = v[1] if cfg.get('use_bias', True) else np.zeros(k.shape[1], np.float32)
            inits += [(f'{name}_W', k), (f'{name}_b', b)]
            nodes.append(ow.node('Gemm', [cur, f'{name}_W', f'{name}_b'], [out]))
            plan.append(('dense', k, b))
            cur = out
            act = cfg.get('activation', 'linear')
        else:
            sys.exit(f'layer type {cls} is not supported by this converter')
        if act == 'relu':
            i += 1; nodes.append(ow.node('Relu', [cur], [f'x{i}'])); cur = f'x{i}'; plan.append(('relu',))
        elif act == 'softmax':
            i += 1; nodes.append(ow.node('Softmax', [cur], [f'x{i}'], [ow.attr_int('axis', 1)])); cur = f'x{i}'; plan.append(('softmax',))
        elif act != 'linear':
            sys.exit(f'activation {act} not supported')
    nodes.append(ow.node('Identity', [cur], ['probs']))

    graph = b''.join(ow.f_bytes(1, n) for n in nodes) + ow.f_bytes(2, 'altynbk_cnn_aug')
    graph += b''.join(ow.f_bytes(5, ow.tensor(n, a)) for n, a in inits)
    graph += ow.f_bytes(11, ow.value_info('nhwc_input', ['N', size, size, 1]))
    graph += ow.f_bytes(12, ow.value_info('probs', ['N', n_classes]))
    model = ow.f_int(1, 7) + ow.f_bytes(2, 'calcink-convert_pretrained') + ow.f_bytes(7, graph)
    model += ow.f_bytes(8, ow.f_bytes(1, '') + ow.f_int(2, 13))
    path.write_bytes(model)
    return plan


# ── 5. independent NumPy forward pass (NHWC, Keras semantics) ───────────────
def numpy_forward(plan, x):
    from numpy.lib.stride_tricks import sliding_window_view as win
    for step in plan:
        kind = step[0]
        if kind == 'conv':
            _, k, b, pad = step
            kh, kw = k.shape[:2]
            if pad == 'same':
                x = np.pad(x, ((0, 0), ((kh - 1) // 2, kh - 1 - (kh - 1) // 2), ((kw - 1) // 2, kw - 1 - (kw - 1) // 2), (0, 0)))
            w = win(x, (kh, kw), axis=(1, 2))                       # N,H,W,C,kh,kw
            x = np.einsum('nhwcij,ijco->nhwo', w, k, optimize=True) + b
        elif kind == 'relu':
            x = np.maximum(x, 0)
        elif kind == 'bn':
            _, g, be, m, v, eps = step
            x = (x - m) / np.sqrt(v + eps) * g + be
        elif kind == 'pool':
            ph, pw = step[1]
            n, h, w_, c = x.shape
            x = x[:, :h - h % ph, :w_ - w_ % pw].reshape(n, h // ph, ph, w_ // pw, pw, c).max(axis=(2, 4))
        elif kind == 'gap':
            x = x.mean(axis=(1, 2))
        elif kind == 'dense':
            x = x @ step[1] + step[2]
        elif kind == 'softmax':
            e = np.exp(x - x.max(1, keepdims=True)); x = e / e.sum(1, keepdims=True)
    return x


# ── 6. the repository's own preprocessing (src/data.py), for the accuracy check
def preprocess(path_bytes, size):
    import cv2
    img = cv2.imdecode(np.frombuffer(path_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:                         # transparency → white
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = img.astype(np.float32) / 255.0
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border) > 0.5:
        arr = 1.0 - arr
    ys, xs = np.where(arr > 0.25)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0 + 1, x1 - x0 + 1
        side = int(max(h, w) * 1.3)
        canvas = np.zeros((side, side), np.float32)
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = arr[y0:y1 + 1, x0:x1 + 1]
        arr = canvas
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def main():
    z = open_repo_zip(sys.argv[1] if len(sys.argv) > 1 else None)
    meta = json.loads(z.read(find(z, 'models/meta.json')))
    classes = meta['class_names']
    size = int(meta.get('img_size', [64, 64])[0])
    layers, W, raw = load_keras(z.read(find(z, 'models/cnn_aug.keras')))
    print(f'read {len(raw)} weight arrays for layers: {", ".join(W)}')

    OUT.mkdir(exist_ok=True)
    onnx_path = OUT / 'pretrained.onnx'
    plan = to_onnx(layers, W, size, len(classes), onnx_path)
    print(f'wrote {onnx_path} ({onnx_path.stat().st_size // 1024} KB)')

    rng = np.random.default_rng(0)
    x = rng.random((4, size, size, 1), dtype=np.float32)
    ref = numpy_forward(plan, x)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        run = lambda a: sess.run(None, {'nhwc_input': a.astype(np.float32)})[0]
        diff = np.abs(run(x) - ref).max()
        print(f'ONNX vs NumPy reference: max difference {diff:.2e}')
        assert diff < 1e-4, 'conversion mismatch'
    except ImportError:
        print('(onnxruntime not installed — skipping the ONNX check)')
        run = lambda a: numpy_forward(plan, a)

    # accuracy on the repository's own images (whatever the zip contains)
    imgs = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg')) and '/data/' in n]
    if imgs:
        sample = imgs[::max(1, len(imgs) // 1500)]
        X, Y = [], []
        for n in sample:
            label = n.split('/')[-2]
            if label not in classes:
                continue
            a = preprocess(z.read(n), size)
            if a is not None:
                X.append(a); Y.append(classes.index(label))
        if X:
            pred = np.concatenate([run(np.array(X[i:i + 256])[..., None]).argmax(1) for i in range(0, len(X), 256)])
            print(f'accuracy on {len(X)} images from the repo: {(pred == np.array(Y)).mean():.3f}')
    else:
        print('(no data/ images in the zip — skipping the accuracy check)')

    info = {
        'name': 'altynbk CNN (augmented), MIT',
        'source': REPO,
        'license': 'MIT',
        'classes': [CHAR.get(c, c) for c in classes],
        'size': size,
        'fit': round(size / 1.3, 2),     # crop_to_content(): side = 1.3 × longest side
        'fitIncludesPen': True,          # their crop is measured on inked pixels
        'penWidth': 4,                   # tuned: matches their training strokes (99.6 % on test glyphs)
        'inkIsWhite': True,              # normalize_polarity(): light strokes on dark
        'layout': 'NHWC',                # Keras: [n, 64, 64, 1]
    }
    (OUT / 'pretrained.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('wrote models/pretrained.json, classes:', ' '.join(info['classes']))


if __name__ == '__main__':
    main()
