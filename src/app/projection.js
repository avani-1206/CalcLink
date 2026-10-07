/*
 * projection.js — PHASE 4: put the answers back onto the paper
 * ─────────────────────────────────────────────────────────────────────────
 * CalcCore.evaluateDocument() (Shunting Yard, in calcink-core.js) turns every
 * recognised line into a result, remembering variables such as  x = 5  for
 * the lines below. This file draws those results:
 *   • a number right after the "=" (different font + colour from user ink)
 *   • optional BODMAS steps under it, e.g.  [M] 4 × 3 = 12   [A] 18 + 12 = 30
 *   • "Undefined" for division by zero, a grey "?" for unreadable input
 *   • a mini graph for  y = f(x)  lines  (Phase 5 creativity)
 *   • optional debug labels showing what the model read ("Show reads")
 *
 * Each answer is keyed by the stroke ids of its "=" sign. When you edit a
 * line only that answer changes (and re-animates); the others stay still.
 *
 * Public API (window.CalcInk.projection):
 *   render(lines)        evaluate + draw everything
 *   prune(strokes)       drop answers whose "=" was erased (instant feedback)
 *   setLabels(bool), labelsOn       debug view: what the model read
 *   setSteps(bool), stepsOn         BODMAS steps under each answer
 *   onAnswer(fn)                    fn(result, text) when an answer appears/changes
 *   paint(ctx)           draw answers + graphs into a 2D context (PNG export)
 */
(function () {
  'use strict';
  const CI = (window.CalcInk = window.CalcInk || {});
  const Core = window.CalcCore;
  const LOW_CONF = 0.6;

  const resultsLayer = document.getElementById('results');
  const labelsLayer  = document.getElementById('labels');
  const els = new Map();               // key → element (answers and graphs)
  let showLabels = false, showSteps = false, lastLines = [];
  const answerListeners = [];                // micro-interactions (haptics/sound) subscribe here

  const keyOf = (prefix, sym) => prefix + sym.strokes.map(s => s.id).sort((a, b) => a - b).join('-');

  function replayAnimation(el) {
    el.style.animation = 'none';
    void el.offsetWidth;               // force a style flush so the animation restarts
    el.style.animation = '';
  }

  function getEl(key, tag, cls) {
    let el = els.get(key);
    if (!el) {
      el = document.createElement(tag);
      el.className = cls;
      el.dataset.key = key;
      resultsLayer.appendChild(el);
      els.set(key, el);
    }
    return el;
  }

  /* ── answers ───────────────────────────────────────────────────────────── */
  function renderAnswer(line, r, seen) {
    const text = Core.displayText(r);
    const eq = line.symbols[line.symbols.length - 1];
    if (text === null || !eq || eq.char !== '=') return;
    const key = keyOf('r', eq);
    seen.add(key);
    const a = Core.resultAnchor(eq.box, line.box);
    const el = getEl(key, 'div', 'res');
    const unsure = line.symbols.filter(s => (s.conf ?? 1) < LOW_CONF);
    el.style.left = `${a.x}px`;
    el.style.top = `${a.y}px`;
    el.style.fontSize = `${text === 'Undefined' ? Math.round(a.fontSize * 0.6) : a.fontSize}px`;
    el.title = r.status === 'error' ? `Couldn't evaluate “${line.text}”: ${r.message}`
             : unsure.length ? `Read as “${line.text}” — not sure about: ${unsure.map(s => s.char).join(' ')}`
             : `Read as “${line.text}”`;
    el.classList.toggle('unsure', unsure.length > 0 && text !== '?');
    if (el.dataset.text !== text) {
      el.dataset.text = text;
      el.textContent = text;
      el.classList.toggle('undef', text === 'Undefined');
      el.classList.toggle('err', text === '?');
      replayAnimation(el);
      answerListeners.forEach(fn => fn(r, text));
    }
    if (showSteps && r.steps && r.steps.length) renderSteps(key, r.steps, a, seen);
  }

  /* ── BODMAS steps under an answer ("Steps" button / T) ─────────────────── */
  function renderSteps(key, steps, a, seen) {
    const k = 's' + key.slice(1);
    seen.add(k);
    const el = getEl(k, 'div', 'steps');
    el.style.left = `${a.x}px`;
    el.style.top = `${a.y + a.fontSize * 0.62 + 4}px`;
    const sig = steps.map(st => st.rule + st.text).join('|');
    if (el.dataset.sig === sig) return;
    el.dataset.sig = sig;
    el.textContent = '';
    for (const st of steps) {
      const row = document.createElement('div');
      const rule = document.createElement('span');
      rule.className = 'rule';
      rule.textContent = st.rule;
      rule.title = { B: 'Brackets', O: 'Orders (powers)', D: 'Division', M: 'Multiplication', A: 'Addition', S: 'Subtraction' }[st.rule];
      row.append(rule, document.createTextNode(st.text));
      el.appendChild(row);
    }
    replayAnimation(el);
  }

  /* ── y = f(x) graphs ───────────────────────────────────────────────────── */
  const PLOT_W = 190, PLOT_H = 120;
  function renderPlot(line, r, vars, seen) {
    const eq = line.symbols[1];
    if (!eq) return;
    const key = keyOf('p', eq);
    const sig = r.expr.join('');
    const pts = Core.plotSamples(r.expr, vars);
    if (!pts) return;
    seen.add(key);
    const el = getEl(key, 'canvas', 'plot');
    el.style.left = `${line.box.x1 + 28}px`;
    // centred on the line, but never pushed above the top edge of the paper
    el.style.top = `${Math.max(PLOT_H / 2 + 6, (line.box.y0 + line.box.y1) / 2)}px`;
    el.style.width = `${PLOT_W}px`; el.style.height = `${PLOT_H}px`;
    el.title = `y = ${sig}   (x from −10 to 10)`;
    if (el.dataset.sig === sig && el.dataset.dark === String(CI.canvas.dark)) return;
    el.dataset.sig = sig; el.dataset.dark = String(CI.canvas.dark);
    el._pts = pts;
    drawPlot(el, pts);
    replayAnimation(el);
  }

  function drawPlot(el, pts) {
    const dpr = window.devicePixelRatio || 1;
    el.width = PLOT_W * dpr; el.height = PLOT_H * dpr;
    const g = el.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintPlot(g, pts, 0, 0);
  }

  function paintPlot(g, pts, ox, oy) {
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue('--C-ink').trim() || '#1d2a44';
    const accent = css.getPropertyValue('--C-result').trim() || '#c2410c';
    const ys = pts.filter(p => p.y !== null).map(p => p.y);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = x => ox + 8 + (x + 10) / 20 * (PLOT_W - 16);
    const Y = y => oy + 8 + (hi - y) / (hi - lo) * (PLOT_H - 16);
    g.save();
    g.lineWidth = 1; g.strokeStyle = ink; g.globalAlpha = 0.25;
    g.beginPath(); g.moveTo(X(-10), Y(0)); g.lineTo(X(10), Y(0)); g.moveTo(X(0), Y(lo)); g.lineTo(X(0), Y(hi)); g.stroke();
    g.globalAlpha = 1; g.strokeStyle = accent; g.lineWidth = 2.2; g.lineJoin = 'round';
    g.beginPath();
    let pen = false, prev = null;
    for (const p of pts) {
      // break the curve at gaps and at huge jumps (e.g. around 1÷x at x = 0)
      const jump = prev && p.y !== null && Math.abs(Y(p.y) - Y(prev)) > PLOT_H;
      if (p.y === null || p.y < lo || p.y > hi || jump) { pen = false; prev = p.y; continue; }
      pen ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y));
      pen = true; prev = p.y;
    }
    g.stroke();
    g.fillStyle = ink; g.globalAlpha = 0.45; g.font = '10px system-ui, sans-serif';
    g.fillText('−10', ox + 6, oy + PLOT_H - 6); g.fillText('10', ox + PLOT_W - 18, oy + PLOT_H - 6);
    g.restore();
  }

  /* ── debug labels ──────────────────────────────────────────────────────── */
  function renderLabels(lines) {
    labelsLayer.textContent = '';
    if (!showLabels) return;
    for (const line of lines) {
      if (!line.symbols.length) continue;
      const el = document.createElement('div');
      el.className = 'lbl';
      for (const s of line.symbols) {
        const span = document.createElement('span');
        span.textContent = s.char ?? '?';
        if ((s.conf ?? 0) < LOW_CONF) span.className = 'low';
        span.title = `${s.char} · ${Math.round((s.conf ?? 0) * 100)}% sure` + (s.source ? ` · ${s.source}` : '');
        el.appendChild(span);
      }
      el.style.left = `${line.box.x0}px`;
      el.style.top = `${line.box.y1 + 8}px`;
      labelsLayer.appendChild(el);
    }
  }

  /* ── public ────────────────────────────────────────────────────────────── */
  function render(lines) {
    lastLines = lines;
    // lineChars() also turns small raised digits into powers (2³ → 2^3)
    const evals = Core.evaluateDocument(lines.map(l => Core.lineChars(l)));
    const seen = new Set(), vars = {};
    lines.forEach((line, i) => {
      const r = evals[i];
      if (r.kind === 'plot') renderPlot(line, r, vars, seen);
      else renderAnswer(line, r, seen);
      if (r.kind === 'assign' && r.status === 'ok') vars[r.name] = r.value;
    });
    for (const [key, el] of els) if (!seen.has(key)) { el.remove(); els.delete(key); }
    renderLabels(lines);
    return evals;
  }

  function prune(strokes) {
    const ids = new Set(strokes.map(s => s.id));
    for (const [key, el] of els)
      if (!key.slice(1).split('-').every(id => ids.has(+id))) { el.remove(); els.delete(key); }
    if (showLabels) labelsLayer.textContent = '';
  }

  function paint(g) {
    g.textBaseline = 'middle';
    for (const el of els.values()) {
      if (el.tagName === 'CANVAS') {
        const r = el.getBoundingClientRect(), s = resultsLayer.getBoundingClientRect();
        paintPlot(g, el._pts, r.left - s.left, r.top - s.top);
      } else {
        const cs = getComputedStyle(el);
        g.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
        g.fillStyle = cs.color;
        g.fillText(el.textContent, parseFloat(el.style.left), parseFloat(el.style.top));
      }
    }
  }

  function setLabels(on) { showLabels = on; renderLabels(lastLines); }
  function setSteps(on) { showSteps = on; if (lastLines.length) render(lastLines); }

  CI.projection = {
    render, prune, paint, setLabels, setSteps,
    get labelsOn() { return showLabels; },
    get stepsOn() { return showSteps; },
    onAnswer: fn => answerListeners.push(fn),
    redrawPlots() { for (const el of els.values()) if (el.tagName === 'CANVAS' && el._pts) { el.dataset.dark = String(CI.canvas.dark); drawPlot(el, el._pts); } },
  };
})();
