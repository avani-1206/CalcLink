/*
 * recognizer.js — PHASE 2: handwriting recognition (main-thread side)
 * ─────────────────────────────────────────────────────────────────────────
 * Pipeline (runs after you lift the pen, never while drawing):
 *   1. CalcCore.segment(): pen strokes → lines → symbols
 *   2. each symbol is drawn into a small grayscale image the way its model
 *      expects (size, margins, pen width, polarity)            → rasterize()
 *   3. all symbols go to the Web Worker in ONE batch per model; the worker
 *      runs ONNX Runtime Web on its own thread, so drawing stays at 60 FPS
 *   4. CalcCore.combinePredictions() merges the models' answers
 *
 * Models (see README → "Model attribution"):
 *   • pretrained  – models/pretrained.onnx + .json: the open-source altynbk CNN
 *                   (MIT), converted from Keras by tools/convert_pretrained.py.
 *                   Primary model: digits and + − × ÷ =.
 *   • bundled     – models/calcink_symbols.onnx, a tiny CNN shipped with the app.
 *                   Adds ( ) y for variables/plots (the pre-trained model has no
 *                   such classes), and stands in if pretrained.onnx is missing.
 *
 * Public API (window.CalcInk.recognizer):
 *   load()                 start the worker and load the models
 *   recognize(strokes)     → Promise<lines>, each symbol gets .char and .conf
 *   state / engine / models,  onState(fn)
 */
(function () {
  'use strict';
  const CI = (window.CalcInk = window.CalcInk || {});
  const Core = window.CalcCore;

  const ORT_VER = '1.20.1';
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  const ORT_LOCAL = new URL('vendor/ort/', document.baseURI).href;   // filled by `npm install`

  /* ── what each model expects as input ──────────────────────────────────── */
  const BUNDLED = {
    id: 'bundled',
    name: 'CalcInk symbols CNN (bundled)',
    url: 'models/calcink_symbols.onnx',
    classes: ['0','1','2','3','4','5','6','7','8','9','+','-','x','÷','=','(',')','y'],
    size: 28,             // input is size × size
    fit: 20,              // longest side of the symbol inside the image (px)
    fitIncludesPen: false,// measure `fit` on the centre-line (false) or on the inked pixels (true)
    penWidth: 2.2,        // pen width used when drawing the symbol image (px, at `size`)
    inkIsWhite: true,     // white ink on black background
    layout: 'NCHW',       // tensor layout: [n,1,H,W]  (NHWC = [n,H,W,1], Keras style)
  };
  let PRETRAINED = null;  // filled from models/pretrained.json if it exists

  const specs = [];       // models that are loaded, primary first
  const state = { value: 'loading', engine: '', models: [] };
  const stateListeners = [];
  function setState(v, extra = {}) {
    Object.assign(state, { value: v }, extra);
    stateListeners.forEach(fn => fn(state));
  }

  /* ── rasterise one symbol for one model (main-thread fallback) ─────────── */
  // Normally the WORKER draws the symbol images (OffscreenCanvas), so even
  // this step stays off the main thread. Browsers without OffscreenCanvas in
  // workers fall back to drawing here.
  const rasterCanvases = new Map();
  function rasterCtx(px) {
    if (!rasterCanvases.has(px)) {
      const c = document.createElement('canvas');
      c.width = c.height = px;
      rasterCanvases.set(px, c.getContext('2d', { willReadFrequently: true }));
    }
    return rasterCanvases.get(px);
  }
  const rasterize = (sym, spec = BUNDLED) => Core.rasterizeSymbol(rasterCtx(spec.size * Core.RASTER_SS), sym, spec);
  const rasterSpec = s => ({ size: s.size, fit: s.fit, fitIncludesPen: !!s.fitIncludesPen, penWidth: s.penWidth, inkIsWhite: s.inkIsWhite });
  let workerDraws = false;

  /* ── the worker ────────────────────────────────────────────────────────── */
  let worker = null, reqId = 0;
  const pending = new Map();

  function makeWorker() {
    // Standalone build: the worker source is inlined in the page → Blob URL.
    const inline = document.getElementById('worker-src');
    if (inline) return new Worker(URL.createObjectURL(new Blob([inline.textContent], { type: 'text/javascript' })));
    return new Worker('src/worker/inference-worker.js');
  }

  function infer(jobs, symbols) {
    return new Promise((resolve, reject) => {
      const id = ++reqId;
      pending.set(id, { resolve, reject });
      const transfer = jobs.filter(j => j.batch).map(j => j.batch.buffer);   // transfer, no copy
      worker.postMessage({ type: 'run', id, jobs, symbols }, transfer);
    });
  }

  /* ── loading ───────────────────────────────────────────────────────────── */
  function inlineBytes(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const bin = atob(el.textContent.replace(/\s+/g, ''));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  }
  async function loadPretrainedSpec() {
    const inline = document.getElementById('pretrained-json');
    try {
      const meta = inline ? JSON.parse(inline.textContent) : await (await fetch('models/pretrained.json')).json();
      return { id: 'pretrained', url: 'models/pretrained.onnx', ...meta };
    } catch (e) { return null; }       // not installed → bundled model only
  }

  async function load() {
    setState('loading');
    worker = makeWorker();
    worker.onerror = e => setState('error', { error: e.message || 'worker failed' });

    const toLoad = [];
    PRETRAINED = location.protocol === 'file:' && !document.getElementById('pretrained-json') ? null : await loadPretrainedSpec();
    if (PRETRAINED) {
      try {
        toLoad.push({ spec: PRETRAINED, bytes: inlineBytes('model-pretrained') || await fetchBytes(PRETRAINED.url) });
      } catch (e) { console.warn('[CalcInk] pre-trained model not available:', e.message); }
    }
    try {
      toLoad.push({ spec: BUNDLED, bytes: inlineBytes('model-onnx') || await fetchBytes(BUNDLED.url) });
    } catch (e) { console.warn('[CalcInk] bundled model not available:', e.message); }

    worker.onmessage = ({ data: m }) => {
      if (m.type === 'ready') {
        specs.length = 0;
        for (const id of m.models) specs.push(toLoad.find(t => t.spec.id === id).spec);
        if (!specs.length) return setState('error', { error: 'no model could be loaded' });
        workerDraws = !!m.offscreen;
        setState('ready', { engine: m.engine, models: specs.map(s => s.name || s.id) });
      } else if (m.type === 'fatal') {
        setState('error', { error: m.message });
      } else if (m.type === 'result' || m.type === 'error') {
        const p = pending.get(m.id); pending.delete(m.id);
        if (p) m.type === 'result' ? p.resolve(m.outputs) : p.reject(new Error(m.message));
      }
    };
    worker.postMessage({
      type: 'init',
      ortUrls: [ORT_LOCAL, ORT_CDN],
      models: toLoad.map(t => ({ id: t.spec.id, bytes: t.bytes, size: t.spec.size, layout: t.spec.layout,
                                 numClasses: t.spec.classes.length })),
    }, toLoad.map(t => t.bytes.buffer));
  }

  /* ── recognise the whole page ──────────────────────────────────────────── */
  async function recognize(strokes) {
    const lines = Core.segment(strokes.filter(s => s.kind === 'pen'));
    const queue = [];
    for (const line of lines)
      for (const sym of line.symbols) {
        if (Core.isDecimalPoint(sym.box, line.box)) { sym.char = '.'; sym.conf = 1; sym.source = 'geometry'; }
        else queue.push(sym);
      }
    if (queue.length && specs.length) {
      const n = queue.length;
      let jobs, symbols;
      if (workerDraws) {
        // send only the vectors; the worker draws the images itself
        symbols = queue.map(s => ({ box: s.box, strokes: s.strokes.map(st => ({ pts: st.pts })) }));
        jobs = specs.map(spec => ({ model: spec.id, spec: rasterSpec(spec), n }));
      } else {
        jobs = specs.map(spec => {
          const S = spec.size * spec.size, batch = new Float32Array(n * S);
          queue.forEach((s, i) => batch.set(rasterize(s, spec), i * S));
          return { model: spec.id, batch, n };
        });
      }
      const outs = await infer(jobs, symbols);
      const primary = { classes: specs[0].classes, probs: outs[0] };
      const secondary = specs[1] ? { classes: specs[1].classes, probs: outs[1] } : null;
      Core.combinePredictions(primary, secondary, n).forEach((p, i) => Object.assign(queue[i], p));
    }
    for (const line of lines) line.text = Core.lineChars(line).join('');   // includes ^ for raised digits
    return lines;
  }

  CI.recognizer = {
    load, recognize, rasterize,
    get state() { return state.value; },
    get engine() { return state.engine; },
    get models() { return state.models; },
    get error() { return state.error; },
    get workerDraws() { return workerDraws; },
    onState: fn => stateListeners.push(fn),
    specs: { BUNDLED, get PRETRAINED() { return PRETRAINED; } },
  };
})();
