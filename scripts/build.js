#!/usr/bin/env node
/*
 * Production build (zero dependencies)
 *   npm run build
 * Produces
 *   dist/site/                     → upload this folder to GitHub Pages / Netlify / Vercel
 *   dist/calcink-standalone.html   → ONE file with everything inlined (CSS, JS, worker,
 *                                    models as base64). Works even when double-clicked.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const SITE = path.join(DIST, 'site');
const read = p => fs.readFileSync(path.join(ROOT, p));
const exists = p => fs.existsSync(path.join(ROOT, p));

function copy(rel) {
  const from = path.join(ROOT, rel), to = path.join(SITE, rel);
  if (!fs.existsSync(from)) return;
  if (fs.statSync(from).isDirectory()) {
    for (const f of fs.readdirSync(from)) copy(path.join(rel, f));
  } else {
    fs.mkdirSync(path.dirname(to), { recursive: true });
    fs.copyFileSync(from, to);
  }
}

function buildSite() {
  fs.rmSync(SITE, { recursive: true, force: true });
  for (const f of ['index.html', 'sw.js', 'manifest.webmanifest', 'icon.svg', 'src', 'models', 'vendor']) copy(f);
  fs.writeFileSync(path.join(SITE, '.nojekyll'), '');     // GitHub Pages: serve files as-is
}

function buildStandalone() {
  let html = read('index.html').toString();
  const b64 = p => read(p).toString('base64').replace(/.{1,120}/g, '$&\n');
  html = html.replace('<link rel="stylesheet" href="src/styles.css">', () => `<style>\n${read('src/styles.css')}</style>`);
  html = html.replace(/<link rel="(icon|manifest)"[^>]*>\n/g, '');
  html = html.replace('<meta charset="UTF-8">', '<meta charset="UTF-8">\n<meta name="calcink-standalone" content="1">');
  html = html.replace(/<script src="(src\/[^"]+)"><\/script>/g, (_, src) => {
    const code = read(src).toString();
    if (code.includes('</script')) throw new Error(`${src} contains "</script" and cannot be inlined`);
    return `<script>\n${code}</script>`;
  });
  let data = `<script id="worker-src" type="text/plain">\n${read('src/core/calcink-core.js')}\n${read('src/worker/inference-worker.js')}</script>\n`;
  data += `<script id="model-onnx" type="application/octet-stream">\n${b64('models/calcink_symbols.onnx')}</script>\n`;
  if (exists('models/pretrained.onnx') && exists('models/pretrained.json')) {
    data += `<script id="pretrained-json" type="application/json">${read('models/pretrained.json')}</script>\n`;
    data += `<script id="model-pretrained" type="application/octet-stream">\n${b64('models/pretrained.onnx')}</script>\n`;
  }
  // data blocks must come BEFORE the app scripts, which read them at start-up
  html = html.replace('<!-- App code:', () => data + '<!-- App code:');
  fs.mkdirSync(DIST, { recursive: true });
  fs.writeFileSync(path.join(DIST, 'calcink-standalone.html'), html);
  return html.length;
}

buildSite();
const size = buildStandalone();
console.log('built dist/site/');
console.log(`built dist/calcink-standalone.html (${Math.round(size / 1024)} KB)`);
