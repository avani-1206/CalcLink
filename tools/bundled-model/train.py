"""
Train a tiny CNN (pure NumPy) on the synthetic symbols and export it to ONNX.

Network (≈210k parameters, ~0.8 MB):
  input [N,1,28,28]
  Conv 3x3 → 16 ch → ReLU → MaxPool 2      → [N,16,14,14]
  Conv 3x3 → 32 ch → ReLU → MaxPool 2      → [N,32,7,7]
  Flatten → Dense 128 → ReLU → Dense 18 → Softmax
"""
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from synth import make_dataset, CLASSES
from onnx_writer import save_onnx

rng = np.random.default_rng(42)
NC = len(CLASSES)


def he(shape, fan_in):
    return (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)


P = {
    'W1': he((16, 1, 3, 3), 9), 'b1': np.zeros(16, np.float32),
    'W2': he((32, 16, 3, 3), 144), 'b2': np.zeros(32, np.float32),
    'W3': he((32 * 7 * 7, 128), 1568), 'b3': np.zeros(128, np.float32),
    'W4': he((128, NC), 128), 'b4': np.zeros(NC, np.float32),
}


def conv_fwd(x, W, b):
    N, C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = sliding_window_view(xp, (3, 3), axis=(2, 3))            # N,C,H,W,3,3
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
    out = cols @ W.reshape(W.shape[0], -1).T + b                    # NHW,F
    return out.reshape(N, H, Wd, -1).transpose(0, 3, 1, 2), cols


def conv_bwd(dout, cols, x, W):
    N, C, H, Wd = x.shape
    F = W.shape[0]
    d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    dW = (d.T @ cols).reshape(W.shape)
    db = d.sum(0)
    dcols = (d @ W.reshape(F, -1)).reshape(N, H, Wd, C, 3, 3)
    dxp = np.zeros((N, C, H + 2, Wd + 2), np.float32)
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + Wd] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dxp[:, :, 1:-1, 1:-1], dW, db


def pool_fwd(x):
    N, C, H, W = x.shape
    r = x.reshape(N, C, H // 2, 2, W // 2, 2)
    out = r.max(axis=(3, 5))
    mask = (r == out[:, :, :, None, :, None])
    return out, mask


def pool_bwd(dout, mask):
    d = mask * dout[:, :, :, None, :, None]
    N, C, h, _, w, _ = d.shape
    return d.reshape(N, C, h * 2, w * 2)


def forward(x, train=False):
    c1, k1 = conv_fwd(x, P['W1'], P['b1']); r1 = np.maximum(c1, 0); p1, m1 = pool_fwd(r1)
    c2, k2 = conv_fwd(p1, P['W2'], P['b2']); r2 = np.maximum(c2, 0); p2, m2 = pool_fwd(r2)
    f = p2.reshape(len(x), -1)
    h = np.maximum(f @ P['W3'] + P['b3'], 0)
    logits = h @ P['W4'] + P['b4']
    cache = (x, c1, k1, p1, m1, c2, k2, m2, f, h, p2.shape)
    return logits, cache


def backward(dlog, cache):
    x, c1, k1, p1, m1, c2, k2, m2, f, h, p2s = cache
    g = {}
    g['W4'] = h.T @ dlog; g['b4'] = dlog.sum(0)
    dh = (dlog @ P['W4'].T) * (h > 0)
    g['W3'] = f.T @ dh; g['b3'] = dh.sum(0)
    dp2 = (dh @ P['W3'].T).reshape(p2s)
    dc2 = pool_bwd(dp2, m2) * (c2 > 0)
    dp1, g['W2'], g['b2'] = conv_bwd(dc2, k2, p1, P['W2'])
    dc1 = pool_bwd(dp1, m1) * (c1 > 0)
    _, g['W1'], g['b1'] = conv_bwd(dc1, k1, x, P['W1'])
    return g


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def accuracy(X, Y, bs=512):
    pred = np.concatenate([forward(X[i:i + bs])[0].argmax(1) for i in range(0, len(X), bs)])
    return (pred == Y).mean(), pred


if __name__ == '__main__':
    import sys
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    Xv, Yv = make_dataset(300, seed=999)
    print(f'val set ready {time.time() - t0:.0f}s', flush=True)

    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    step, lr, bs = 0, 1e-3, 128
    for ep in range(epochs):
        # fresh synthetic data every epoch = endless variety, no overfitting
        X, Y = make_dataset(per, seed=ep)
        idx = rng.permutation(len(X))
        lr_ep = lr * (0.5 if ep >= epochs * 0.6 else 1) * (0.5 if ep >= epochs * 0.85 else 1)
        loss_sum = 0
        for i in range(0, len(X), bs):
            b = idx[i:i + bs]
            xb = X[b]
            # small random shift (±1px) for robustness
            if rng.random() < .5:
                xb = np.roll(xb, rng.integers(-1, 2, 2), axis=(2, 3))
            logits, cache = forward(xb, True)
            p = softmax(logits)
            loss_sum += -np.log(p[np.arange(len(b)), Y[b]] + 1e-9).sum()
            p[np.arange(len(b)), Y[b]] -= 1
            g = backward(p / len(b), cache)
            step += 1
            for k in P:
                g[k] += 1e-4 * P[k] if k[0] == 'W' else 0
                m[k] = .9 * m[k] + .1 * g[k]
                v[k] = .999 * v[k] + .001 * g[k] ** 2
                mh = m[k] / (1 - .9 ** step); vh = v[k] / (1 - .999 ** step)
                P[k] -= (lr_ep * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        acc, _ = accuracy(Xv, Yv)
        print(f'epoch {ep + 1}/{epochs} loss {loss_sum / len(X):.4f} val_acc {acc:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    acc, pred = accuracy(Xv, Yv)
    conf = np.zeros((NC, NC), int)
    for a, b in zip(Yv, pred):
        conf[a, b] += 1
    print('per-class acc:', {CLASSES[i]: round(conf[i, i] / conf[i].sum(), 3) for i in range(NC)})
    np.savez('weights.npz', **P)
    save_onnx(P, 'calcink_symbols.onnx', NC)
    print('saved calcink_symbols.onnx')
