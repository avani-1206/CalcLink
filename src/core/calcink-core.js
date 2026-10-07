/*
 * calcink-core.js — the "brain" of CalcInk, with NO browser dependencies.
 *
 * Everything here is a pure function (same input → same output, no DOM,
 * no canvas), which is why it can be unit-tested with plain Node:
 *     npm test   (or: node --test)
 *
 * Contents
 *   1. Geometry   – group strokes into lines and symbols, place the answer
 *   2. Tokenizer  – turn recognised characters into numbers / operators
 *   3. Parser     – Shunting Yard algorithm  (infix  →  Reverse Polish)
 *   4. Evaluator  – run the RPN with a stack (never uses eval())
 *   5. Document   – evaluate every line top→bottom, remembering variables
 *   6. Gestures   – detect a "scratch-out" scribble and what it covers
 *   7. Plotting   – sample y = f(x) for the mini graph
 *   8. Recognition helpers – combine two models' predictions
 *   9. Rasteriser – draw a symbol into a model's input image (needs a 2D
 *      context passed in, so it works with <canvas> AND OffscreenCanvas)
 *
 * Works both as a classic <script> (defines window.CalcCore) and as a
 * Node module (module.exports), so the browser and the tests share ONE copy.
 */
(function (root) {
  'use strict';

  /* ═══════════════════════ 1. GEOMETRY ═══════════════════════════════════ */

  function bboxOf(pts, pad = 0) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pts) {
      if (p.x < x0) x0 = p.x; if (p.y < y0) y0 = p.y;
      if (p.x > x1) x1 = p.x; if (p.y > y1) y1 = p.y;
    }
    return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
  }

  // Union-find grouping: items i and j end up together if same(i, j) is true
  // for them or for any chain of items between them.
  function groupBy(items, same) {
    const parent = items.map((_, i) => i);
    const find = i => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let i = 0; i < items.length; i++)
      for (let j = i + 1; j < items.length; j++)
        if (same(items[i], items[j])) parent[find(i)] = find(j);
    const groups = new Map();
    items.forEach((it, i) => {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(it);
    });
    return [...groups.values()];
  }

  // Box around the stroke centre-lines (pen width ignored).
  const coreBox = strokes => bboxOf(strokes.flatMap(s => s.pts));

  /**
   * strokes → lines → symbols.
   *  - two strokes are on the same LINE if their vertical ranges overlap
   *  - two strokes are the same SYMBOL if the centre of the narrower one
   *    lies inside the horizontal range of the wider one  ("=", "+", "÷", "4")
   *  - a big horizontal gap (> 1.5 × line height) starts a new expression,
   *    so two sums can sit side by side on one row
   * Returns lines sorted top→bottom (then left→right), symbols left→right.
   */
  function segment(strokes) {
    const items = strokes.map(s => ({ s, b: bboxOf(s.pts) }));
    const lines = groupBy(items, (a, b) => Math.min(a.b.y1, b.b.y1) - Math.max(a.b.y0, b.b.y0) >= 0);
    return lines.map(line => {
      const lb = coreBox(line.map(i => i.s)), lineH = lb.y1 - lb.y0;
      const isDot = it => Math.max(it.b.x1 - it.b.x0, it.b.y1 - it.b.y0) < Math.max(5, lineH * 0.22);
      const symbols = groupBy(line, (a, b) => {
        const [n, w] = (a.b.x1 - a.b.x0) < (b.b.x1 - b.b.x0) ? [a, b] : [b, a];
        const c = (n.b.x0 + n.b.x1) / 2, margin = 2;
        if (!(c >= w.b.x0 - margin && c <= w.b.x1 + margin)) return false;
        // A dot only belongs to its neighbour when it floats above/below it
        // (the dots of "÷"). A dot level with a digit is a decimal point.
        if (isDot(n) && !isDot(w)) {
          const pad = Math.max(1, lineH * 0.04);
          const verticalOverlap = n.b.y0 <= w.b.y1 + pad && n.b.y1 >= w.b.y0 - pad;
          if (verticalOverlap) return false;
        }
        return true;
      }).map(g => {
        const st = g.map(i => i.s);
        return { strokes: st, box: coreBox(st) };
      });
      symbols.sort((a, b) => (a.box.x0 + a.box.x1) - (b.box.x0 + b.box.x1));
      // Two sums written side by side on the same row: split at big gaps.
      const parts = [[]], bigGap = Math.max(48, lineH * 1.5);
      symbols.forEach((sym, i) => {
        if (i && sym.box.x0 - symbols[i - 1].box.x1 > bigGap) parts.push([]);
        parts[parts.length - 1].push(sym);
      });
      return parts.map(syms => ({ box: coreBox(syms.flatMap(x => x.strokes)), symbols: syms }));
    }).flat().sort((a, b) => (a.box.y0 - b.box.y0) || (a.box.x0 - b.box.x0));
  }

  // A tiny blob inside a much taller line is a decimal point (no model needed).
  function isDecimalPoint(symBox, lineBox) {
    const span = Math.max(symBox.x1 - symBox.x0, symBox.y1 - symBox.y0);
    const lineH = lineBox.y1 - lineBox.y0;
    return lineH > 20 && span < Math.max(5, lineH * 0.15);
  }

  /**
   * Powers (the "O" in BODMAS) are written as small raised digits: 2³.
   * Given a recognised line (symbols with .char and .box), return its
   * characters with "^" inserted before every raised group:
   *   2 ³ + 1  →  2 ^ 3 + 1        x ² ⁺ ¹ (raised "2+1") → x ^ ( 2 + 1 )
   * A symbol is raised relative to the symbol before the group (its "base")
   * when it is clearly smaller, sits in the upper part of the base, and its
   * bottom is above the base's middle. The base must be a number, ")" or a
   * variable, so a normal "−" at mid-height is never mistaken for a power.
   */
  function lineChars(line) {
    const syms = line.symbols, out = [];
    const canBase = c => (c >= '0' && c <= '9') || c === ')' || c === 'x' || c === 'y' || c === '.';
    const canStart = c => (c >= '0' && c <= '9') || c === '(' || c === 'x' || c === 'y';
    const raisedOver = (s, base) => {
      const H = base.box.y1 - base.box.y0, h = s.box.y1 - s.box.y0;
      if (H < 8) return false;
      return h <= 0.75 * H && (s.box.y0 + s.box.y1) / 2 <= base.box.y0 + 0.3 * H && s.box.y1 <= base.box.y0 + 0.55 * H;
    };
    let i = 0;
    while (i < syms.length) {
      const base = syms[i];
      out.push(base.char ?? '?');
      i++;
      if (!canBase(base.char) || i >= syms.length || !canStart(syms[i].char) || !raisedOver(syms[i], base)) continue;
      const group = [];
      while (i < syms.length && syms[i].char !== '=' && raisedOver(syms[i], base)) group.push(syms[i++].char ?? '?');
      out.push('^');
      if (group.length > 1) out.push('(', ...group, ')'); else out.push(group[0]);
    }
    return out;
  }

  /**
   * Where should the answer go?  Just right of the "=" symbol, vertically
   * centred on it, with a font about as tall as the user's handwriting.
   * Coordinates are CSS pixels relative to the canvas (same as the strokes).
   */
  function resultAnchor(eqBox, lineBox) {
    const lineH = Math.max(lineBox.y1 - lineBox.y0, eqBox.y1 - eqBox.y0);
    const fontSize = Math.round(Math.min(72, Math.max(18, lineH * 0.95)));
    return {
      x: eqBox.x1 + Math.round(fontSize * 0.35),
      y: (eqBox.y0 + eqBox.y1) / 2,           // vertical centre of the "="
      fontSize,
    };
  }

  /* ═══════════════════════ 2. TOKENIZER ══════════════════════════════════ */

  const isDigit = c => c >= '0' && c <= '9';
  const VAR_NAMES = new Set(['x', 'y']);
  const OPS = {
    '+': { prec: 1, assoc: 'L', fn: (a, b) => a + b },
    '-': { prec: 1, assoc: 'L', fn: (a, b) => a - b },
    '×': { prec: 2, assoc: 'L', fn: (a, b) => a * b },
    '÷': { prec: 2, assoc: 'L', fn: (a, b) => (b === 0 ? NaN : a / b) },
    'neg': { prec: 3, assoc: 'R', unary: true, fn: a => -a },
    '^': { prec: 4, assoc: 'R', fn: (a, b) => Math.pow(a, b) },   // Orders: powers bind tightest
  };
  // BODMAS letter for each operator (B is added for steps inside brackets)
  const BODMAS = { '^': 'O', '÷': 'D', '×': 'M', '+': 'A', '-': 'S' };

  class CalcError extends Error {}

  /**
   * Handwritten "x" (variable) and "×" (times) look identical, so the model
   * returns one class for both. We decide from the neighbours:
   *   number/")"  x  number/"("   →  ×   (e.g. 3x5, (2)x4)
   *   anything else               →  variable x   (e.g. x=5, x+10, 2x)
   */
  function resolveX(chars) {
    const leftOperand = c => c !== undefined && (isDigit(c) || c === '.' || c === ')');
    const rightOperand = c => c !== undefined && (isDigit(c) || c === '.' || c === '(');
    return chars.map((c, i) =>
      c === 'x' && leftOperand(chars[i - 1]) && rightOperand(chars[i + 1]) ? '×' : c);
  }

  /** chars (after resolveX) → tokens  {type:'num'|'var'|'op'|'(' |')', value} */
  function tokenize(chars) {
    const tokens = [];
    let i = 0;
    while (i < chars.length) {
      const c = chars[i];
      if (isDigit(c) || c === '.') {
        let s = '';
        while (i < chars.length && (isDigit(chars[i]) || chars[i] === '.')) s += chars[i++];
        if ((s.match(/\./g) || []).length > 1 || s === '.') throw new CalcError(`bad number "${s}"`);
        tokens.push({ type: 'num', value: parseFloat(s) });
        continue;
      }
      if (VAR_NAMES.has(c)) tokens.push({ type: 'var', value: c });
      else if (c === '(' || c === ')') tokens.push({ type: c });
      else if (c === '−') tokens.push({ type: 'op', value: '-' });
      else if (c === '*') tokens.push({ type: 'op', value: '×' });
      else if (c === '/') tokens.push({ type: 'op', value: '÷' });
      else if (OPS[c]) tokens.push({ type: 'op', value: c });
      else throw new CalcError(`unexpected "${c}"`);
      i++;
    }
    return insertImplicitMultiply(tokens);
  }

  // 2(3+4) → 2×(3+4),  3x → 3×x,  (1)(2) → (1)×(2)
  function insertImplicitMultiply(tokens) {
    const out = [];
    const endsOperand = t => t && (t.type === 'num' || t.type === 'var' || t.type === ')');
    const startsOperand = t => t.type === 'num' || t.type === 'var' || t.type === '(';
    for (const t of tokens) {
      if (endsOperand(out[out.length - 1]) && startsOperand(t)) out.push({ type: 'op', value: '×' });
      out.push(t);
    }
    return out;
  }

  /* ═══════════════════════ 3. SHUNTING YARD ══════════════════════════════ */
  /*
   * Dijkstra's algorithm. Read tokens left→right:
   *   number/variable → straight to the output
   *   operator        → first pop operators that must run before it, then push
   *   "("             → push;   ")" → pop until the matching "("
   * Output is Reverse Polish Notation:  3 + 4 × 2  →  3 4 2 × +
   */
  function toRPN(tokens) {
    const out = [], stack = [];
    let prev = null, depth = 0;                       // previous token (to spot unary minus), bracket depth
    for (const tok of tokens) {
      const t = tok.type === 'op' ? { ...tok, depth } : tok;
      if (t.type === '(') depth++;
      else if (t.type === ')') depth = Math.max(0, depth - 1);
      if (t.type === 'num' || t.type === 'var') out.push(t);
      else if (t.type === 'op') {
        const unaryPosition = prev === null || prev.type === 'op' || prev.type === '(';
        if (unaryPosition) {
          if (t.value === '+') { prev = t; continue; }     // unary plus: ignore
          if (t.value !== '-') throw new CalcError(`"${t.value}" needs a number before it`);
          stack.push({ type: 'op', value: 'neg', depth });
        } else {
          const o1 = OPS[t.value];
          while (stack.length) {
            const top = stack[stack.length - 1];
            if (top.type !== 'op') break;
            const o2 = OPS[top.value];
            if (o2.prec > o1.prec || (o2.prec === o1.prec && o1.assoc === 'L')) out.push(stack.pop());
            else break;
          }
          stack.push(t);
        }
      } else if (t.type === '(') stack.push(t);
      else if (t.type === ')') {
        if (prev && (prev.type === '(' || prev.type === 'op')) throw new CalcError('empty or unfinished brackets');
        while (stack.length && stack[stack.length - 1].type !== '(') out.push(stack.pop());
        if (!stack.length) throw new CalcError('too many ")"');
        stack.pop();
      }
      prev = t;
    }
    while (stack.length) {
      const t = stack.pop();
      if (t.type === '(') throw new CalcError('missing ")"');
      out.push(t);
    }
    return out;
  }

  /* ═══════════════════════ 4. EVALUATOR ══════════════════════════════════ */

  function evalRPN(rpn, vars = {}, trace = null) {
    const st = [];
    for (const t of rpn) {
      if (t.type === 'num') st.push(t.value);
      else if (t.type === 'var') {
        if (!(t.value in vars)) throw new CalcError(`${t.value} is not defined yet`);
        st.push(vars[t.value]);
      } else {
        const op = OPS[t.value];
        if (op.unary) {
          if (st.length < 1) throw new CalcError('missing number');
          st.push(op.fn(st.pop()));
        } else {
          if (st.length < 2) throw new CalcError('missing number');
          const b = st.pop(), a = st.pop(), r = op.fn(a, b);
          if (trace) trace.push({ rule: t.depth > 0 ? 'B' : BODMAS[t.value], a, op: t.value, b, r });
          st.push(r);
        }
      }
    }
    if (st.length !== 1) throw new CalcError('incomplete expression');
    return st[0];
  }

  /**
   * Evaluate one expression (array of characters or a string).
   * NEVER throws. Returns one of:
   *   { status:'ok', value }         a normal number
   *   { status:'undefined' }         division by zero / overflow
   *   { status:'error', message }    could not understand the expression
   */
  function evaluate(chars, vars = {}) {
    try {
      const arr = typeof chars === 'string' ? [...chars.replace(/\s+/g, '')] : chars;
      if (!arr.length) throw new CalcError('empty');
      const trace = [];
      const value = evalRPN(toRPN(tokenize(resolveX(arr))), vars, trace);
      if (!Number.isFinite(value)) return { status: 'undefined', steps: formatSteps(trace) };
      return { status: 'ok', value: value === 0 ? 0 : value, steps: formatSteps(trace) };   // -0 → 0
    } catch (err) {
      if (err instanceof CalcError) return { status: 'error', message: err.message };
      return { status: 'error', message: 'internal error' };    // never crash the UI
    }
  }

  /**
   * The order the evaluator actually worked in — this IS BODMAS made visible:
   *   18+4×3  →  [M] 4 × 3 = 12,  [A] 18 + 12 = 30
   */
  function formatSteps(trace) {
    const f = v => (Number.isFinite(v) ? formatNumber(v).replace(/^-/, '−') : 'undefined');
    const wrap = v => (v < 0 ? `(${f(v)})` : f(v));
    return trace.map(s => ({ rule: s.rule, text: `${wrap(s.a)} ${s.op === '-' ? '−' : s.op} ${wrap(s.b)} = ${f(s.r)}` }));
  }

  /** Pretty number: hides floating-point noise (0.1+0.2 → "0.3"). */
  function formatNumber(v) {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1e15 || a < 1e-9) return v.toExponential(6).replace(/\.?0+e/, 'e');
    return String(Number(v.toPrecision(12)));
  }

  /* ═══════════════════════ 5. WHOLE DOCUMENT ═════════════════════════════ */
  /**
   * Interpret one line:
   *   "18+4×3="  → result
   *   "x=5"      → assignment (remembered for the lines BELOW)
   *   "x=2+3="   → assignment AND result
   *   "12+"      → incomplete (still being written, show nothing)
   *   "2+2=4"    → ignored (the user wrote their own answer)
   */
  function evaluateLine(chars, vars = {}) {
    const arr = typeof chars === 'string' ? [...chars.replace(/\s+/g, '')] : chars.slice();
    const eqs = arr.reduce((a, c, i) => (c === '=' ? a.concat(i) : a), []);
    if (!eqs.length) return { kind: 'incomplete' };

    const endsWithEq = eqs[eqs.length - 1] === arr.length - 1;
    const isAssign = arr.length >= 3 && VAR_NAMES.has(arr[0]) && arr[1] === '=';

    if (isAssign) {
      if (eqs.length > 2 || (eqs.length === 2 && !endsWithEq)) return { kind: 'ignored' };
      const expr = arr.slice(2, endsWithEq && eqs.length === 2 ? -1 : undefined);
      // "y = 2x + 1" while x has no value → it's a function of x: draw a graph
      if (arr[0] === 'y' && eqs.length === 1 && !('x' in vars) && resolveX(expr).includes('x')) {
        return { kind: 'plot', name: 'y', expr };
      }
      const r = evaluate(expr, vars);
      return { kind: 'assign', name: arr[0], showResult: eqs.length === 2, ...r };
    }
    if (eqs.length === 1 && endsWithEq) {
      return { kind: 'result', ...evaluate(arr.slice(0, -1), vars) };
    }
    return { kind: 'ignored' };
  }

  /** lines: array of char-arrays (top→bottom). Variables flow downwards. */
  function evaluateDocument(lines) {
    const vars = {};
    return lines.map(chars => {
      const r = evaluateLine(chars, vars);
      if (r.kind === 'assign' && r.status === 'ok') vars[r.name] = r.value;
      return r;
    });
  }

  /** What text to draw next to the "=" (or null to draw nothing). */
  function displayText(r) {
    const wantsResult = r.kind === 'result' || (r.kind === 'assign' && r.showResult);
    if (!wantsResult) return null;
    if (r.status === 'ok') return formatNumber(r.value).replace(/^-/, '−');  // real minus sign
    if (r.status === 'undefined') return 'Undefined';
    return '?';
  }

  /* ═══════════════════════ 6. SCRATCH-OUT GESTURE ═══════════════════════ */
  /*
   * A scribble = a stroke that goes back and forth many times inside a small
   * area. We count direction reversals along x and y (ignoring wiggles
   * smaller than `h` px) and compare the path length with the box size.
   *   normal digits:  ≤ 3 reversals, path ≈ 1–3 × box diagonal
   *   scribble:       ≥ 5 reversals, path ≥ 3.5 × box diagonal
   */
  function countReversals(vals, h) {
    let dir = 0, ext = vals[0], n = 0;
    for (const v of vals) {
      if (dir === 0) {
        if (v - ext > h) { dir = 1; ext = v; } else if (ext - v > h) { dir = -1; ext = v; }
      } else if (dir === 1) {
        if (v > ext) ext = v; else if (ext - v > h) { dir = -1; ext = v; n++; }
      } else {
        if (v < ext) ext = v; else if (v - ext > h) { dir = 1; ext = v; n++; }
      }
    }
    return n;
  }

  function pathLength(pts) {
    let L = 0;
    for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
    return L;
  }

  function isScratch(pts, opts = {}) {
    const minReversals = opts.minReversals ?? 5, minRatio = opts.minRatio ?? 3.5;
    if (pts.length < 10) return false;
    const b = bboxOf(pts), w = b.x1 - b.x0, hgt = b.y1 - b.y0, diag = Math.hypot(w, hgt);
    if (diag < 12) return false;
    const rx = countReversals(pts.map(p => p.x), Math.max(3, w * 0.25));
    const ry = countReversals(pts.map(p => p.y), Math.max(3, hgt * 0.25));
    return Math.max(rx, ry) >= minReversals && pathLength(pts) / diag >= minRatio;
  }

  /** ids of strokes that the scribble covers (≥ half of their points under it). */
  function scratchTargets(scratchPts, strokes, pad = 4) {
    const b = bboxOf(scratchPts, pad);
    const inside = p => p.x >= b.x0 && p.x <= b.x1 && p.y >= b.y0 && p.y <= b.y1;
    return strokes
      .filter(s => s.pts.filter(inside).length >= s.pts.length * 0.5)
      .map(s => s.id);
  }

  /* ═══════════════════════ 7. FUNCTION PLOTTING ══════════════════════════ */
  /** Sample y = expr(x) on [xmin, xmax]. Points where it's undefined get y = null. */
  function plotSamples(expr, vars = {}, { xmin = -10, xmax = 10, n = 161 } = {}) {
    let rpn;
    try { rpn = toRPN(tokenize(resolveX(expr))); } catch (e) { return null; }
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = xmin + (xmax - xmin) * i / (n - 1);
      let y = null;
      try { const v = evalRPN(rpn, { ...vars, x }); if (Number.isFinite(v)) y = v; } catch (e) { /* gap */ }
      out.push({ x, y });
    }
    return out.some(p => p.y !== null) ? out : null;
  }

  /* ═══════════════════════ 8. RECOGNITION HELPERS ════════════════════════ */
  function argmax(probs, offset, k) {
    let best = 0;
    for (let c = 1; c < k; c++) if (probs[offset + c] > probs[offset + best]) best = c;
    return best;
  }

  /**
   * Merge two models, symbol by symbol.
   *   primary   – the pre-trained model (digits, + − × ÷ =)
   *   secondary – the bundled CalcInk model, which also knows ( ) y
   * We trust the primary, except when the secondary is very sure it sees a
   * symbol the primary simply cannot output.
   */
  function combinePredictions(primary, secondary, n, { extra = ['(', ')', 'y'], threshold = 0.85 } = {}) {
    const out = [];
    const P = primary.classes.length, S = secondary ? secondary.classes.length : 0;
    for (let i = 0; i < n; i++) {
      const pb = argmax(primary.probs, i * P, P);
      let pick = { char: primary.classes[pb], conf: primary.probs[i * P + pb], source: 'primary' };
      if (secondary) {
        const sb = argmax(secondary.probs, i * S, S);
        const sc = secondary.classes[sb], sconf = secondary.probs[i * S + sb];
        if (extra.includes(sc) && sconf >= threshold) pick = { char: sc, conf: sconf, source: 'secondary' };
      }
      out.push(pick);
    }
    return out;
  }

  /* ═══════════════════════ 9. RASTERISER ════════════════════════════════ */
  /*
   * Redraw a symbol FROM ITS VECTORS into the exact format a model expects:
   * longest side → spec.fit px, centred, fixed pen width, 4× supersampling
   * then 4×4 box-filter downsampling (= OpenCV INTER_AREA used in training).
   * ctx must be a 2D context of size (spec.size × 4)².
   * Returns Float32Array(size²), values 0..1. For 1-channel images NCHW and
   * NHWC have the same memory order, so one array fits both layouts.
   */
  const RASTER_SS = 4;
  function rasterizeSymbol(ctx, sym, spec) {
    const N = spec.size, W = N * RASTER_SS, b = sym.box;
    const span = Math.max(b.x1 - b.x0, b.y1 - b.y0, 1e-6);
    const pen = spec.penWidth;
    const target = spec.fitIncludesPen ? Math.max(1, spec.fit - pen) : spec.fit;
    const k = target / span * RASTER_SS, cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2, o = W / 2;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, W);
    ctx.strokeStyle = ctx.fillStyle = '#fff';
    ctx.lineWidth = pen * RASTER_SS; ctx.lineCap = ctx.lineJoin = 'round';
    for (const s of sym.strokes) {
      const P = s.pts.map(q => [(q.x - cx) * k + o, (q.y - cy) * k + o]);
      ctx.beginPath();
      if (P.length === 1) { ctx.arc(P[0][0], P[0][1], pen * RASTER_SS / 2 + 1, 0, Math.PI * 2); ctx.fill(); }
      else { ctx.moveTo(P[0][0], P[0][1]); for (const p of P) ctx.lineTo(p[0], p[1]); ctx.stroke(); }
    }
    const px = ctx.getImageData(0, 0, W, W).data;
    const out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let sum = 0;
      for (let dy = 0; dy < RASTER_SS; dy++) for (let dx = 0; dx < RASTER_SS; dx++)
        sum += px[((y * RASTER_SS + dy) * W + x * RASTER_SS + dx) * 4];
      const v = sum / (RASTER_SS * RASTER_SS * 255);
      out[y * N + x] = spec.inkIsWhite ? v : 1 - v;
    }
    return out;
  }

  const api = {
    bboxOf, groupBy, segment, isDecimalPoint, resultAnchor, lineChars, formatSteps,
    resolveX, tokenize, toRPN, evalRPN, evaluate, formatNumber,
    evaluateLine, evaluateDocument, displayText,
    countReversals, pathLength, isScratch, scratchTargets,
    plotSamples, argmax, combinePredictions,
    RASTER_SS, rasterizeSymbol,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CalcCore = api;
})(typeof self !== 'undefined' ? self : this);
