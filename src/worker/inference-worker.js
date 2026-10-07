/*
 * inference-worker.js — runs the neural networks on a separate thread.
 * ─────────────────────────────────────────────────────────────────────────
 * Why a worker? JavaScript has one main thread that also paints the page.
 * If inference ran there, every recognition would freeze the pen for a few
 * frames. In a worker it runs in parallel and drawing stays at 60 FPS.
 *
 * Messages
 *   in : {type:'init', ortUrls:[...], models:[{id, bytes, size, layout, numClasses}]}
 *   out: {type:'ready', engine, models:[ids that loaded]}  |  {type:'fatal', message}
 *   in : {type:'run', id, jobs:[{model, n, batch?:Float32Array, spec?}], symbols?:[{box, strokes}]}
 *        (with `symbols`, the worker draws the input images itself on an
 *         OffscreenCanvas — so not even rasterisation touches the main thread)
 *   out: {type:'result', id, outputs:[Float32Array per job]}  |  {type:'error', id, message}
 *
 * Engines, tried in order:
 *   1. ONNX Runtime Web from vendor/ort/ (copied by `npm install`, works offline)
 *   2. ONNX Runtime Web from the jsDelivr CDN
 *   3. a small built-in ONNX interpreter (plain JS) that supports exactly the
 *      layers our two CNNs use, so the app still works with no ONNX Runtime
 */
'use strict';
// shared geometry/rasteriser code (already inlined in the standalone build)
if (typeof CalcCore === 'undefined') importScripts('../core/calcink-core.js');
const canDraw = typeof OffscreenCanvas !== 'undefined';
const rasterCtxs = {};
function rasterBatch(symbols, spec) {
  const W = spec.size * CalcCore.RASTER_SS;
  if (!rasterCtxs[W]) rasterCtxs[W] = new OffscreenCanvas(W, W).getContext('2d', { willReadFrequently: true });
  const S = spec.size * spec.size, batch = new Float32Array(symbols.length * S);
  symbols.forEach((sym, i) => batch.set(CalcCore.rasterizeSymbol(rasterCtxs[W], sym, spec), i * S));
  return batch;
}

const runners = {};       // model id → async (batch, n) => probabilities
const meta = {};          // model id → {size, layout, numClasses}

self.onmessage = async ({ data: m }) => {
  if (m.type === 'init') {
    for (const x of m.models) meta[x.id] = x;
    let engine = null;
    for (const url of m.ortUrls) {
      try {
        importScripts(url + 'ort.min.js');
        ort.env.wasm.wasmPaths = url;
        ort.env.wasm.numThreads = 1;          // threads need cross-origin isolation headers
        engine = 'onnxruntime-web' + (url.includes('jsdelivr') ? ' (CDN)' : ' (local)');
        break;
      } catch (e) { /* try the next location */ }
    }
    if (engine) {
      for (const x of m.models) {
        try {
          const session = await ort.InferenceSession.create(x.bytes.slice(), { executionProviders: ['wasm'] });
          runners[x.id] = makeOrtRunner(session, x);
        } catch (e) { console.warn('[worker] could not load model', x.id, e); }
      }
    }
    // models ORT couldn't run (or no ORT at all): try the built-in engine
    let usedFallback = false;
    for (const x of m.models) {
      if (runners[x.id]) continue;
      try {
        const g = parseOnnx(x.bytes);                 // throws if a layer type is unsupported
        runners[x.id] = async (batch, n) => runGraph(g, batch, n, x);
        usedFallback = true;
      } catch (e) { console.warn('[worker] built-in engine cannot run', x.id, e.message); }
    }
    if (usedFallback) engine = engine ? engine + ' + built-in JS' : 'built-in JS engine (offline fallback)';
    const ids = m.models.map(x => x.id).filter(id => runners[id]);
    if (!ids.length) return postMessage({ type: 'fatal', message: 'no model could be loaded' });
    postMessage({ type: 'ready', engine, models: ids, offscreen: canDraw });
  } else if (m.type === 'run') {
    try {
      const outputs = [];
      for (const job of m.jobs) {
        const batch = job.batch || rasterBatch(m.symbols, job.spec);
        const probs = toProbabilities(Float32Array.from(await runners[job.model](batch, job.n)), meta[job.model].numClasses);
        outputs.push(probs);
      }
      postMessage({ type: 'result', id: m.id, outputs }, outputs.map(o => o.buffer));
    } catch (err) {
      postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) });
    }
  }
};

function makeOrtRunner(session, x) {
  const input = session.inputNames[0], output = session.outputNames[0];
  return async (batch, n) => {
    const dims = x.layout === 'NHWC' ? [n, x.size, x.size, 1] : [n, 1, x.size, x.size];
    const out = await session.run({ [input]: new ort.Tensor('float32', batch, dims) });
    return out[output].data;
  };
}

// Some models output probabilities, others raw scores ("logits"). If a row
// isn't already a probability distribution, apply softmax to it.
function toProbabilities(x, k) {
  for (let r = 0; r < x.length; r += k) {
    let sum = 0, neg = false;
    for (let j = 0; j < k; j++) { sum += x[r + j]; if (x[r + j] < 0) neg = true; }
    if (!neg && Math.abs(sum - 1) < 1e-3) continue;
    let m = -Infinity, z = 0;
    for (let j = 0; j < k; j++) m = Math.max(m, x[r + j]);
    for (let j = 0; j < k; j++) { x[r + j] = Math.exp(x[r + j] - m); z += x[r + j]; }
    for (let j = 0; j < k; j++) x[r + j] /= z;
  }
  return x;
}

/* ── built-in engine: a tiny ONNX interpreter ──────────────────────────── */
// An .onnx file is a Protocol Buffers message:
//   ModelProto.graph(7) → node(1)  {input(1), output(2), op_type(4), attribute(5)}
//                       → initializer(5) {dims(1), name(8), raw_data(9)}   (the weights)
// We decode just those fields, then run the nodes in order.
const SUPPORTED = new Set(['Transpose', 'Conv', 'Relu', 'BatchNormalization', 'MaxPool',
  'GlobalAveragePool', 'Flatten', 'Gemm', 'Softmax', 'Identity']);

function pbFields(b, cb) {
  let i = 0;
  const varint = () => { let v = 0, mul = 1, byte; do { byte = b[i++]; v += (byte & 127) * mul; mul *= 128; } while (byte & 128); return v; };
  while (i < b.length) {
    const key = varint(), f = Math.floor(key / 8), wt = key & 7;
    if (wt === 0) cb(f, varint(), 0);
    else if (wt === 2) { const n = varint(); cb(f, b.subarray(i, i + n), 2); i += n; }
    else if (wt === 5) { cb(f, new DataView(b.buffer, b.byteOffset + i, 4).getFloat32(0, true), 5); i += 4; }
    else if (wt === 1) i += 8;
    else throw new Error('bad protobuf');
  }
}
const utf8 = v => new TextDecoder().decode(v);

function parseOnnx(buf) {
  const g = { nodes: [], weights: {}, input: null, output: null };
  pbFields(buf, (f, graph, wt) => {
    if (f !== 7 || wt !== 2) return;
    pbFields(graph, (k, v, w) => {
      if (w !== 2) return;
      if (k === 1) {                                   // NodeProto
        const node = { inputs: [], outputs: [], op: '', attrs: {} };
        pbFields(v, (a, val, aw) => {
          if (a === 1) node.inputs.push(utf8(val));
          else if (a === 2) node.outputs.push(utf8(val));
          else if (a === 4) node.op = utf8(val);
          else if (a === 5) {                          // AttributeProto
            let name = '', ints = [], value;
            pbFields(val, (t, x, tw) => {
              if (t === 1) name = utf8(x);
              else if (t === 2) value = x;             // float
              else if (t === 3) value = x;             // int
              else if (t === 8) { if (tw === 2) { ints = ints.concat(unpackVarints(x)); } else ints.push(x); }
            });
            node.attrs[name] = ints.length ? ints : value;
          }
        });
        if (!SUPPORTED.has(node.op)) throw new Error(`layer ${node.op} not supported`);
        g.nodes.push(node);
      } else if (k === 5) {                            // TensorProto (weights)
        let name = '', raw = null; const dims = [];
        pbFields(v, (t, x) => { if (t === 1) dims.push(x); if (t === 8) name = utf8(x); if (t === 9) raw = x; });
        g.weights[name] = { data: new Float32Array(raw.slice().buffer), dims };
      } else if (k === 11 && !g.input) {
        pbFields(v, (t, x) => { if (t === 1) g.input = utf8(x); });
      } else if (k === 12 && !g.output) {
        pbFields(v, (t, x) => { if (t === 1) g.output = utf8(x); });
      }
    });
  });
  return g;
}
function unpackVarints(b) {
  const out = []; let i = 0;
  while (i < b.length) { let v = 0, mul = 1, byte; do { byte = b[i++]; v += (byte & 127) * mul; mul *= 128; } while (byte & 128); out.push(v); }
  return out;
}

// Tensors are {data: Float32Array, dims: [...]}; 4-D tensors are NCHW.
function runGraph(g, batch, n, x) {
  const env = Object.assign({}, g.weights);
  env[g.input] = { data: batch, dims: x.layout === 'NHWC' ? [n, x.size, x.size, 1] : [n, 1, x.size, x.size] };
  for (const node of g.nodes) {
    const ins = node.inputs.map(name => env[name]);
    env[node.outputs[0]] = OPS[node.op](ins, node.attrs);
  }
  return env[g.output].data;
}

const OPS = {
  Identity: ([a]) => a,
  Relu: ([a]) => ({ data: a.data.map(v => (v > 0 ? v : 0)), dims: a.dims }),
  Flatten: ([a]) => ({ data: a.data, dims: [a.dims[0], a.data.length / a.dims[0]] }),
  Transpose([a], { perm }) {                         // only NHWC→NCHW (perm 0,3,1,2) is needed
    if (perm.join() !== '0,3,1,2') throw new Error('unsupported transpose');
    const [N, H, W, C] = a.dims, out = new Float32Array(a.data.length);
    for (let n = 0; n < N; n++) for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) for (let c = 0; c < C; c++)
      out[((n * C + c) * H + h) * W + w] = a.data[((n * H + h) * W + w) * C + c];
    return { data: out, dims: [N, C, H, W] };
  },
  Conv([a, k, b], { pads = [0, 0, 0, 0] }) {        // stride 1; loops ordered for contiguous inner rows
    const [N, C, H, W] = a.dims, [F, , KH, KW] = k.dims;
    const OH = H + pads[0] + pads[2] - KH + 1, OW = W + pads[1] + pads[3] - KW + 1;
    const out = new Float32Array(N * F * OH * OW);
    for (let n = 0; n < N; n++) for (let f = 0; f < F; f++) {
      const o = (n * F + f) * OH * OW;
      out.fill(b ? b.data[f] : 0, o, o + OH * OW);
      for (let c = 0; c < C; c++) {
        const src = (n * C + c) * H * W;
        for (let i = 0; i < KH; i++) for (let j = 0; j < KW; j++) {
          const wv = k.data[((f * C + c) * KH + i) * KW + j];
          if (wv === 0) continue;
          for (let y = 0; y < OH; y++) {
            const iy = y + i - pads[0]; if (iy < 0 || iy >= H) continue;
            const x0 = Math.max(0, pads[1] - j), x1 = Math.min(OW, W + pads[1] - j);
            const row = src + iy * W - pads[1] + j, orow = o + y * OW;
            for (let x = x0; x < x1; x++) out[orow + x] += wv * a.data[row + x];
          }
        }
      }
    }
    return { data: out, dims: [N, F, OH, OW] };
  },
  BatchNormalization([a, s, b, m, v], { epsilon = 1e-5 }) {
    const [N, C] = a.dims, hw = a.data.length / (N * C), out = new Float32Array(a.data.length);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
      const k = s.data[c] / Math.sqrt(v.data[c] + epsilon), off = b.data[c] - m.data[c] * k, base = (n * C + c) * hw;
      for (let i = 0; i < hw; i++) out[base + i] = a.data[base + i] * k + off;
    }
    return { data: out, dims: a.dims };
  },
  MaxPool([a], { kernel_shape: [kh, kw], strides }) {
    const [sh, sw] = strides || [kh, kw], [N, C, H, W] = a.dims;
    const OH = Math.floor((H - kh) / sh) + 1, OW = Math.floor((W - kw) / sw) + 1, out = new Float32Array(N * C * OH * OW);
    for (let p = 0; p < N * C; p++) for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
      let best = -Infinity;
      for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) best = Math.max(best, a.data[(p * H + y * sh + i) * W + x * sw + j]);
      out[(p * OH + y) * OW + x] = best;
    }
    return { data: out, dims: [N, C, OH, OW] };
  },
  GlobalAveragePool([a]) {
    const [N, C] = a.dims, hw = a.data.length / (N * C), out = new Float32Array(N * C);
    for (let p = 0; p < N * C; p++) { let s = 0; for (let i = 0; i < hw; i++) s += a.data[p * hw + i]; out[p] = s / hw; }
    return { data: out, dims: [N, C, 1, 1] };
  },
  Gemm([a, w, b], { transA = 0, transB = 0, alpha = 1, beta = 1 }) {
    if (transA) throw new Error('Gemm transA not supported');
    const [N, K] = a.dims, M = transB ? w.dims[0] : w.dims[1], out = new Float32Array(N * M);
    for (let n = 0; n < N; n++) {
      for (let m = 0; m < M; m++) out[n * M + m] = b ? beta * b.data[b.data.length === M ? m : 0] : 0;
      for (let k = 0; k < K; k++) {
        const v = alpha * a.data[n * K + k]; if (!v) continue;
        for (let m = 0; m < M; m++) out[n * M + m] += v * (transB ? w.data[m * K + k] : w.data[k * M + m]);
      }
    }
    return { data: out, dims: [N, M] };
  },
  Softmax([a]) {
    const [N, M] = a.dims, out = new Float32Array(a.data.length);
    for (let n = 0; n < N; n++) {
      let mx = -Infinity, z = 0;
      for (let m = 0; m < M; m++) mx = Math.max(mx, a.data[n * M + m]);
      for (let m = 0; m < M; m++) { out[n * M + m] = Math.exp(a.data[n * M + m] - mx); z += out[n * M + m]; }
      for (let m = 0; m < M; m++) out[n * M + m] /= z;
    }
    return { data: out, dims: a.dims };
  },
};

// Node.js (unit tests): expose the interpreter
if (typeof module !== 'undefined' && module.exports) module.exports = { parseOnnx, runGraph };
