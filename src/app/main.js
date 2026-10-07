/*
 * main.js — wires the pieces together
 * ─────────────────────────────────────────────────────────────────────────
 *   canvas  ──(strokes changed)──►  wait for a pause  ──►  recognizer
 *   recognizer ──(lines)──►  projection.render()  ──►  answers on the page
 *
 * Rules that keep drawing smooth (60 FPS):
 *   • recognition starts only 350 ms after the last change, and never while
 *     the pen is down
 *   • only one recognition runs at a time; if ink changes meanwhile, exactly
 *     one more run is queued (the newest state wins)
 */
(function () {
  'use strict';
  const CI = window.CalcInk;
  const { canvas, recognizer, projection } = CI;
  const $ = id => document.getElementById(id);

  /* ── scheduling ────────────────────────────────────────────────────────── */
  const PAUSE_MS = 350;
  let timer = null, busy = false, again = false;

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(run, PAUSE_MS);
  }

  async function run() {
    if (recognizer.state !== 'ready') return;
    if (canvas.isDrawing()) return schedule();            // pen is down — wait
    if (busy) { again = true; return; }
    busy = true;
    try {
      const lines = await recognizer.recognize(canvas.strokes);
      CI.lines = lines;
      CI.evals = projection.render(lines);
    } catch (err) {
      console.error('[CalcInk] recognition failed', err);
    }
    busy = false;
    if (again) { again = false; run(); }
  }

  canvas.onChange(() => {
    projection.prune(canvas.strokes);                     // instant: erased "=" → answer gone
    $('hint').classList.toggle('gone', canvas.strokes.length > 0);
    schedule();
  });

  /* ── model status ──────────────────────────────────────────────────────── */
  recognizer.onState(s => {
    $('mDot').dataset.s = s.value;
    const info = s.value === 'ready' ? `${s.models.join(' + ')}\nEngine: ${s.engine}`
               : s.value === 'error' ? s.error : 'loading…';
    $('bRec').title = `Show what the model reads (R)\nModel: ${info}`;
    if (s.value === 'ready') run();
    if (s.value === 'error') {
      canvas.toast(location.protocol === 'file:'
        ? 'Open with “npm run dev” or use calcink-standalone.html'
        : 'Recognition model failed to load', 4000);
    }
  });

  /* ── toolbar + keyboard ────────────────────────────────────────────────── */
  function toggleLabels() {
    projection.setLabels(!projection.labelsOn);
    $('bRec').setAttribute('aria-pressed', String(projection.labelsOn));
  }
  $('bRec').onclick = toggleLabels;

  /* ── preferences (remembered per browser; storage may be unavailable) ──── */
  const prefs = (() => { try { return JSON.parse(localStorage.getItem('calcink-prefs')) || {}; } catch (e) { return {}; } })();
  const savePrefs = () => { try { localStorage.setItem('calcink-prefs', JSON.stringify(prefs)); } catch (e) { /* private mode */ } };

  function setSteps(on) {
    projection.setSteps(on);
    $('bSteps').setAttribute('aria-pressed', String(on));
    prefs.steps = on; savePrefs();
  }
  $('bSteps').onclick = () => setSteps(!projection.stepsOn);

  /* ── micro-interactions: haptic tick + optional soft "pop" ──────────────── */
  let audio = null;
  function pop(ok) {
    if (!prefs.sound) return;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(ok ? 880 : 330, t);
      o.frequency.exponentialRampToValueAtTime(ok ? 1320 : 220, t + 0.08);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.08, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
      o.connect(g).connect(audio.destination);
      o.start(t); o.stop(t + 0.2);
      o.onended = () => { o.disconnect(); g.disconnect(); };      // free the nodes (no leaks)
    } catch (e) { /* audio not available */ }
  }
  projection.onAnswer((r, text) => {
    const ok = r.status === 'ok';
    if (navigator.vibrate) { try { navigator.vibrate(ok ? 12 : [8, 40, 8]); } catch (e) { /* not allowed */ } }
    pop(ok);
  });
  function setSound(on, silent = false) {
    prefs.sound = on; savePrefs();
    $('bSound').setAttribute('aria-pressed', String(on));
    $('iWave').style.display = on ? '' : 'none';
    if (on && !silent) pop(true);
  }
  $('bSound').onclick = () => setSound(!prefs.sound);
  $('bDark').addEventListener('click', () => { prefs.dark = canvas.dark; savePrefs(); });
  $('bExport').onclick = () => canvas.exportPNG(projection.paint);
  $('bDark').addEventListener('click', () => projection.redrawPlots());

  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.ctrlKey || e.metaKey) {
      const k = e.key.toLowerCase();
      if (k === 'z') { e.preventDefault(); e.shiftKey ? canvas.redo() : canvas.undo(); }
      else if (k === 'y') { e.preventDefault(); canvas.redo(); }
      return;
    }
    switch (e.key.toLowerCase()) {
      case 'p': canvas.setTool('pen'); break;
      case 'h': canvas.setTool('highlighter'); break;
      case 'e': canvas.setTool('pixel'); break;
      case 's': canvas.setTool('stroke'); break;
      case 'r': toggleLabels(); break;
      case 't': setSteps(!projection.stepsOn); break;
      case '[': canvas.setWidth(canvas.getWidth() - 1); break;
      case ']': canvas.setWidth(canvas.getWidth() + 1); break;
    }
  });

  /* ── offline support ───────────────────────────────────────────────────── */
  // The service worker caches every file (and ONNX Runtime) on first visit, so
  // the app keeps working in airplane mode. Not available on file:// pages.
  if ('serviceWorker' in navigator && location.protocol.startsWith('http') &&
      !document.querySelector('meta[name="calcink-standalone"]')) {
    navigator.serviceWorker.register('sw.js').catch(err => console.warn('[CalcInk] offline cache unavailable', err));
  }

  // restore preferences
  if (prefs.dark && !canvas.dark) $('bDark').click();
  if (prefs.steps) setSteps(true);
  setSound(!!prefs.sound, true);

  recognizer.load();
})();
