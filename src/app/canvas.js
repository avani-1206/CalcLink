/*
 * canvas.js — PHASE 1: the digital-ink canvas
 * ─────────────────────────────────────────────────────────────────────────
 * Big idea: the canvas pixels are NOT the source of truth.
 * The source of truth is `strokes`: a list of vector strokes (lists of points).
 * The picture can always be re-drawn from that list. That one decision gives
 * cheap undo/redo, sharp re-drawing on resize / high-DPI screens, dark-mode
 * re-colouring and — most importantly — clean input for the recogniser.
 *
 * Public API (window.CalcInk.canvas):
 *   strokes        current stroke list (read-only, immutable array)
 *   isDrawing()    true while a pointer is down
 *   onChange(fn)   fn() is called after every committed change
 *   toast(msg)     small message at the bottom of the page
 *   exportPNG(fn)  download a PNG; fn(ctx) may paint extra layers on it
 *   dark           whether dark mode is on
 */
(function () {
  'use strict';
  const CI = (window.CalcInk = window.CalcInk || {});
  const Core = window.CalcCore;

  /* ── ELEMENTS ──────────────────────────────────────────────────────────── */
  const $       = id => document.getElementById(id);
  const canvas  = $('ink');
  const ctx     = canvas.getContext('2d', { alpha: true, desynchronized: true });
  const stage   = $('stage');
  const ring    = $('ring');
  const toastEl = $('toast');

  /*
   * DUAL-CANVAS ARCHITECTURE
   *   committed = offscreen canvas holding every FINISHED stroke
   *   canvas    = what you see: committed + the stroke being drawn right now
   * While the pen moves we only copy `committed` (one fast GPU blit) and draw
   * the single live stroke on top, so drawing cost does not grow with the
   * amount of ink already on the page.
   */
  const committed = document.createElement('canvas');
  const cc        = committed.getContext('2d', { alpha: true });

  /* ── STATE ─────────────────────────────────────────────────────────────── */
  /*
   * A stroke:  { id, kind:'pen'|'hl', pts:[{x,y},…], w, color, alpha, box }
   * `strokes` is IMMUTABLE: every change builds a new array. Undo just keeps
   * the old arrays — they share stroke objects, so history is almost free
   * (storing full-screen bitmaps instead would cost ~8–30 MB per step).
   */
  let strokes   = [];
  const history = { undo: [], redo: [] };
  const MAX_HISTORY = 300;

  let tool = 'pen', color = 'ink', lineWidth = 4, opacity = 1, darkMode = false;

  let cur           = null;   // live stroke {kind, pts, w, color, alpha, erase, scratch}
  let gesture       = null;   // 'draw' | 'stroke-erase'
  let activeId      = null;   // pointerId that owns the gesture (other fingers ignored)
  let strokesAtDown = null;   // strokes when the gesture began → pushed to undo if changed
  let lastErasePt   = null;
  let penSeen       = false;  // once a stylus is used, finger touches count as palm
  let DPR = 1, nextId = 1;
  const listeners = [];
  let fading = [];            // erased strokes that are fading out {strokes, t0}

  /* ── HELPERS ───────────────────────────────────────────────────────────── */
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const d2  = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  const boxesTouch = (a, b) => a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1;

  // Add points so neighbours are at most `step` px apart (fast strokes are sparse).
  function densify(pts, step) {
    if (pts.length < 2) return pts.slice();
    const out = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i];
      const n = Math.ceil(Math.sqrt(d2(a, b)) / step);
      for (let k = 1; k < n; k++) {
        const q = { x: a.x + (b.x - a.x) * k / n, y: a.y + (b.y - a.y) * k / n };
        if (a.p !== undefined && b.p !== undefined) q.p = a.p + (b.p - a.p) * k / n;   // stylus pressure
        out.push(q);
      }
      out.push(b);
    }
    return out;
  }

  function makeStroke(base, pts) {
    return { id: nextId++, kind: base.kind, pts, w: base.w, color: base.color,
             alpha: base.alpha, box: Core.bboxOf(pts, base.w * 0.65 + 1), dense: true };
  }

  let toastTimer;
  function toast(msg, ms = 1600) {
    toastEl.textContent = msg;
    toastEl.classList.add('on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('on'), ms);
  }

  /* ── CANVAS SIZE (high-DPI aware) ──────────────────────────────────────── */
  // The backing store is CSS size × devicePixelRatio, and drawing is scaled by
  // DPR, so lines stay crisp on Retina screens. Strokes are vectors, so a
  // resize simply re-draws them — nothing gets blurry or cut off.
  function resizeCanvases() {
    DPR = window.devicePixelRatio || 1;
    const r = stage.getBoundingClientRect();
    canvas.width  = committed.width  = Math.max(1, Math.round(r.width  * DPR));
    canvas.height = committed.height = Math.max(1, Math.round(r.height * DPR));
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    cc .setTransform(DPR, 0, 0, DPR, 0, 0);
    rebake();
    flush();
  }

  /* ── DRAWING PRIMITIVES ────────────────────────────────────────────────── */
  const resolveColor = c => (c === 'ink' ? (darkMode ? '#e8eaf0' : '#1d2a44') : c);

  function applyStyle(cx, s) {
    if (s.erase) {
      cx.globalCompositeOperation = 'destination-out';   // "cut" pixels away
      cx.strokeStyle = cx.fillStyle = '#000';
      cx.globalAlpha = 1;
    } else {
      cx.globalCompositeOperation = 'source-over';
      cx.strokeStyle = cx.fillStyle = s.scratch ? '#e63946' : resolveColor(s.color);
      cx.globalAlpha = s.scratch ? 0.45 : s.alpha;
    }
    cx.lineWidth = s.w;
    cx.lineCap   = 'round';
    cx.lineJoin  = 'round';
  }
  function resetStyle(cx) {
    cx.globalCompositeOperation = 'source-over';
    cx.globalAlpha = 1;
  }

  /*
   * PEN INK — drawn as a filled outline instead of a fixed-width line, so it
   * behaves like a real pen: stroke ends taper, and with a stylus the line
   * gets thicker when you press harder (PointerEvent.pressure). Mouse and
   * finger input have no real pressure and get a steady width.
   * Outline = left edge (forward) + round cap + right edge (backward) + cap,
   * smoothed with quadratic curves; one fill keeps opacity uniform.
   */
  function drawInk(cx, s, alphaMul) {
    const p = s.pts.length > 1 ? (s.dense ? s.pts : densify(s.pts, 2)) : s.pts;
    const n = p.length;
    const L = new Float32Array(n);
    for (let i = 1; i < n; i++) L[i] = L[i - 1] + Math.hypot(p[i].x - p[i - 1].x, p[i].y - p[i - 1].y);
    const total = L[n - 1], taper = Math.max(1, Math.min(s.w * 3, total * 0.35));
    const half = i => {
      const t = Math.min(1, (L[i] + s.w * 0.4) / taper, (total - L[i] + s.w * 0.4) / taper);
      const ends = 0.6 + 0.4 * Math.sin(t * Math.PI / 2);
      const press = p[i].p === undefined ? 1 : 0.7 + 0.6 * p[i].p;    // 0.7× … 1.3×
      return Math.max(0.35, s.w * ends * press / 2);
    };
    const left = [], right = [];
    for (let i = 0; i < n; i++) {
      const a = p[Math.max(0, i - 2)], b = p[Math.min(n - 1, i + 2)];
      let dx = b.x - a.x, dy = b.y - a.y; const d = Math.hypot(dx, dy) || 1;
      dx /= d; dy /= d;
      const h = half(i);
      left.push({ x: p[i].x - dy * h, y: p[i].y + dx * h });
      right.push({ x: p[i].x + dy * h, y: p[i].y - dx * h });
    }
    const curve = pts => {
      for (let i = 1; i < pts.length - 1; i++) {
        const m = mid(pts[i], pts[i + 1]);
        cx.quadraticCurveTo(pts[i].x, pts[i].y, m.x, m.y);
      }
      cx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    };
    applyStyle(cx, s);
    cx.globalAlpha *= alphaMul;
    cx.beginPath();
    cx.moveTo(left[0].x, left[0].y);
    curve(left);
    const e = p[n - 1], ae = Math.atan2(left[n - 1].y - e.y, left[n - 1].x - e.x);
    cx.arc(e.x, e.y, half(n - 1), ae, ae - Math.PI, true);            // end cap
    right.reverse();
    curve(right);
    const b0 = p[0], ab = Math.atan2(right[n - 1].y - b0.y, right[n - 1].x - b0.x);
    cx.arc(b0.x, b0.y, half(0), ab, ab - Math.PI, true);             // start cap
    cx.closePath();
    cx.fill('nonzero');
    resetStyle(cx);
  }

  // Smooth stroke: quadratic curves through the mid-points of neighbours.
  function drawFull(cx, s, alphaMul = 1) {
    const p = s.pts;
    if (!p.length) return;
    if (s.kind === 'pen' && !s.erase && !s.scratch && p.length > 1) return drawInk(cx, s, alphaMul);
    applyStyle(cx, s);
    cx.globalAlpha *= alphaMul;
    cx.beginPath();
    if (p.length === 1) {
      cx.arc(p[0].x, p[0].y, s.w / 2, 0, Math.PI * 2);
      cx.fill();
    } else {
      cx.moveTo(p[0].x, p[0].y);
      const m = mid(p[0], p[1]);
      cx.lineTo(m.x, m.y);
      for (let i = 1; i < p.length - 1; i++) {
        const nm = mid(p[i], p[i + 1]);
        cx.quadraticCurveTo(p[i].x, p[i].y, nm.x, nm.y);
      }
      cx.lineTo(p[p.length - 1].x, p[p.length - 1].y);
      cx.stroke();
    }
    resetStyle(cx);
  }

  function rebake() {
    cc.save(); cc.setTransform(1, 0, 0, 1, 0, 0);
    cc.clearRect(0, 0, committed.width, committed.height);
    cc.restore();
    for (const s of strokes) drawFull(cc, s);
  }

  function flush() {
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(committed, 0, 0);
    ctx.restore();
    const now = performance.now();
    for (const f of fading) {
      const a = 1 - (now - f.t0) / 300;
      if (a > 0) for (const s of f.strokes) drawFull(ctx, s, a);
    }
    if (cur) drawFull(ctx, cur);
  }

  // Erased strokes fade out over 300 ms instead of vanishing (Phase 5 polish).
  function fadeOut(removed) {
    if (!removed.length) return;
    fading.push({ strokes: removed, t0: performance.now() });
    const tick = () => {
      fading = fading.filter(f => performance.now() - f.t0 < 300);
      flush();
      if (fading.length) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  /* ── UNDO / REDO ───────────────────────────────────────────────────────── */
  function pushHistory(before) {
    history.undo.push(before);
    if (history.undo.length > MAX_HISTORY) history.undo.shift();
    history.redo = [];
  }
  function afterChange() {
    rebake(); flush(); syncButtons();
    for (const fn of listeners) fn();
  }
  function undo() {
    if (cur || !history.undo.length) return;          // never undo mid-stroke
    history.redo.push(strokes);
    strokes = history.undo.pop();
    afterChange();
  }
  function redo() {
    if (cur || !history.redo.length) return;
    history.undo.push(strokes);
    strokes = history.redo.pop();
    afterChange();
  }
  function syncButtons() {
    $('bUndo').disabled = !history.undo.length;
    $('bRedo').disabled = !history.redo.length;
  }

  /* ── ERASERS (both are VECTOR operations) ──────────────────────────────── */
  /*
   * Pixel eraser: points under the eraser are removed and each stroke is
   * split into its surviving pieces. So what you SEE is exactly what the
   * recogniser READS (an invisible "eraser stroke" would not be).
   */
  function erasePixels(list, path, r) {
    const P  = densify(path, Math.max(1, r / 2));
    const eb = Core.bboxOf(P, r);
    let changed = false;
    const out = [];
    for (const s of list) {
      if (!boxesTouch(s.box, eb)) { out.push(s); continue; }
      const lim2 = (r + s.w / 4) ** 2;
      const keep = s.pts.map(q => !P.some(e => d2(e, q) < lim2));
      if (keep.every(Boolean)) { out.push(s); continue; }
      changed = true;
      let run = [];
      const emit = () => {
        if (run.length >= 2 || (run.length === 1 && s.pts.length === 1)) out.push(makeStroke(s, run));
        run = [];
      };
      s.pts.forEach((q, i) => (keep[i] ? run.push(q) : emit()));
      emit();
    }
    return changed ? out : list;
  }

  // Stroke eraser: delete every whole stroke the eraser path touches.
  function strokeEraseAlong(from, to) {
    const r = Math.max(10, lineWidth);
    const P = densify([from, to], r / 2);
    const eb = Core.bboxOf(P, r);
    const hit = s => boxesTouch(s.box, eb) && s.pts.some(q => P.some(e => d2(e, q) <= (r + s.w / 2) ** 2));
    const removed = strokes.filter(hit);
    if (removed.length) {
      strokes = strokes.filter(s => !hit(s));
      rebake(); fadeOut(removed); flush();
    }
  }

  /* ── POINTER EVENTS (mouse, pen and touch all arrive as pointer events) ── */
  const pt = e => {
    const r = canvas.getBoundingClientRect();
    const q = { x: e.clientX - r.left, y: e.clientY - r.top };
    if (e.pointerType === 'pen' && e.pressure > 0) q.p = e.pressure;   // real stylus pressure only
    return q;
  };
  const toolWidth = t => (t === 'pixel' ? Math.max(lineWidth * 3, 14)
                       : t === 'highlighter' ? Math.max(lineWidth * 2, 16) : lineWidth);

  canvas.addEventListener('pointerdown', e => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();                                  // no text selection / image drag / long-press menu
    if (activeId !== null) return;                       // a second finger: ignore
    if (e.pointerType === 'pen') penSeen = true;
    if (e.pointerType === 'touch' && penSeen) return;    // palm rejection
    activeId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    strokesAtDown = strokes;
    const p = pt(e);

    if (tool === 'stroke') {
      gesture = 'stroke-erase';
      lastErasePt = p;
      strokeEraseAlong(p, p);
      return;
    }
    gesture = 'draw';
    cur = {
      kind:  tool === 'highlighter' ? 'hl' : 'pen',
      pts:   [p],
      w:     toolWidth(tool),
      color,
      alpha: tool === 'highlighter' ? 0.3 : opacity,
      erase: tool === 'pixel',
      scratch: false,
    };
    flush();
  });

  canvas.addEventListener('pointermove', e => {
    if (e.pointerType !== 'touch') {                     // cursor ring for mouse / stylus hover
      const p = pt(e), sz = tool === 'stroke' ? Math.max(10, lineWidth) * 2 : toolWidth(tool);
      ring.style.cssText = `left:${p.x}px;top:${p.y}px;width:${sz}px;height:${sz}px;display:block;`;
    }
    if (e.pointerId !== activeId) return;
    const evts = e.getCoalescedEvents?.() ?? [e];         // every sub-frame sample
    if (gesture === 'stroke-erase') {
      for (const ev of evts) { const p = pt(ev); strokeEraseAlong(lastErasePt, p); lastErasePt = p; }
      return;
    }
    if (!cur) return;
    for (const ev of evts) cur.pts.push(pt(ev));
    // live feedback: a pen stroke that turns into a scribble over ink goes red
    if (cur.kind === 'pen' && !cur.erase && cur.pts.length % 6 === 0) {
      cur.scratch = Core.isScratch(cur.pts) && Core.scratchTargets(cur.pts, strokes).length > 0;
    }
    flush();
  });

  function endGesture(e) {
    if (e.pointerId !== activeId) return;
    activeId = null;
    if (cur) {
      const s = cur; cur = null;
      if (s.erase) {
        strokes = erasePixels(strokes, s.pts, s.w / 2);
      } else if (s.kind === 'pen' && Core.isScratch(s.pts)) {
        // SCRATCH-TO-ERASE: scribbling over ink deletes the ink under it
        const ids = new Set(Core.scratchTargets(s.pts, strokes));
        if (ids.size) {
          const removed = strokes.filter(x => ids.has(x.id));
          strokes = strokes.filter(x => !ids.has(x.id));
          fadeOut(removed);
          toast(`Scratched out ${removed.length} stroke${removed.length > 1 ? 's' : ''}`);
        } else {
          strokes = [...strokes, makeStroke(s, densify(s.pts, 2))];   // scribble on empty paper = ink
        }
      } else {
        strokes = [...strokes, makeStroke(s, densify(s.pts, 2))];
      }
    }
    gesture = null;
    if (strokes !== strokesAtDown) { pushHistory(strokesAtDown); afterChange(); }
    else flush();
  }
  canvas.addEventListener('pointerup',     endGesture);
  canvas.addEventListener('pointercancel', endGesture);
  canvas.addEventListener('lostpointercapture', endGesture);   // e.g. the window lost focus mid-stroke
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerleave',  () => { ring.style.display = 'none'; });

  /* ── TOOLBAR ───────────────────────────────────────────────────────────── */
  function setTool(t) {
    tool = t;
    const map = { pen: 'tPen', highlighter: 'tHL', pixel: 'tPE', stroke: 'tSE' };
    Object.values(map).forEach(id => $(id).setAttribute('aria-pressed', 'false'));
    $(map[t]).setAttribute('aria-pressed', 'true');
    canvas.style.cursor = (t === 'stroke' || t === 'pixel') ? 'cell' : 'crosshair';
  }
  $('tPen').onclick = () => setTool('pen');
  $('tHL') .onclick = () => setTool('highlighter');
  $('tPE') .onclick = () => setTool('pixel');
  $('tSE') .onclick = () => setTool('stroke');

  function setWidth(w) {
    lineWidth = Math.min(32, Math.max(1, w));
    $('wSlider').value = lineWidth; $('wVal').textContent = lineWidth;
  }
  $('wSlider').oninput = e => setWidth(+e.target.value);
  $('oSlider').oninput = e => { opacity = +e.target.value / 100; $('oVal').textContent = e.target.value + '%'; };

  function setColor(c) {
    color = c;
    document.querySelectorAll('.sw').forEach(s => s.classList.toggle('on', s.dataset.c === c));
    if (/^#[0-9a-f]{6}$/i.test(c)) $('cp').value = c;
  }
  document.querySelectorAll('.sw').forEach(s => (s.onclick = () => setColor(s.dataset.c)));
  $('cp').oninput = e => setColor(e.target.value);

  $('bUndo').onclick = undo;
  $('bRedo').onclick = redo;
  function clearAll() {
    if (!strokes.length || cur) return;
    pushHistory(strokes);
    strokes = [];
    afterChange();
    toast('Cleared — Ctrl+Z to bring it back');
  }
  $('bClear').onclick = clearAll;

  /* ── EXPORT PNG ────────────────────────────────────────────────────────── */
  function exportPNG(paintExtras) {
    const off = document.createElement('canvas');
    off.width = canvas.width; off.height = canvas.height;
    const oc = off.getContext('2d');
    oc.fillStyle = darkMode ? '#1a1d27' : '#f7f3e8';
    oc.fillRect(0, 0, off.width, off.height);
    oc.save(); oc.setTransform(DPR, 0, 0, DPR, 0, 0);
    oc.strokeStyle = darkMode ? 'rgba(255,255,255,.06)' : 'rgba(0,0,0,.09)';
    oc.lineWidth = 1;
    for (let y = 36; y < off.height / DPR; y += 36) {
      oc.beginPath(); oc.moveTo(0, y); oc.lineTo(off.width / DPR, y); oc.stroke();
    }
    oc.strokeStyle = darkMode ? 'rgba(255,120,120,.16)' : 'rgba(220,80,80,.28)';      // margin line
    oc.beginPath(); oc.moveTo(47.5, 0); oc.lineTo(47.5, off.height / DPR); oc.stroke();
    oc.restore();
    oc.drawImage(committed, 0, 0);
    if (paintExtras) { oc.save(); oc.setTransform(DPR, 0, 0, DPR, 0, 0); paintExtras(oc); oc.restore(); }
    const a = document.createElement('a');
    a.download = `calcink-${Date.now()}.png`;
    a.href = off.toDataURL('image/png');
    a.click();
    toast('Exported ✓');
  }

  /* ── DARK MODE ─────────────────────────────────────────────────────────── */
  $('bDark').onclick = () => {
    darkMode = !darkMode;
    document.documentElement.classList.toggle('dark', darkMode);
    document.querySelector('meta[name=theme-color]')?.setAttribute('content', darkMode ? '#1a1d27' : '#f7f3e8');
    $('iMoon').style.display = darkMode ? 'none' : '';
    $('iSun') .style.display = darkMode ? ''     : 'none';
    rebake(); flush();                                  // 'ink' strokes switch colour
  };

  /* ── RESIZE / DPR CHANGES ──────────────────────────────────────────────── */
  new ResizeObserver(resizeCanvases).observe(stage);
  (function watchDPR() {
    window.matchMedia(`(resolution:${window.devicePixelRatio}dppx)`)
      .addEventListener('change', () => { resizeCanvases(); watchDPR(); }, { once: true });
  })();

  setTool('pen');
  setColor('ink');

  CI.canvas = {
    get strokes() { return strokes; },
    get dark() { return darkMode; },
    isDrawing: () => activeId !== null,
    onChange: fn => listeners.push(fn),
    setTool, setWidth: w => setWidth(w), getWidth: () => lineWidth,
    undo, redo, clearAll, exportPNG, toast,
  };
})();
