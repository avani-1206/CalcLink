#!/usr/bin/env node
/*
 * Copy ONNX Runtime Web's browser files from node_modules into vendor/ort/,
 * so the app never needs a CDN (required for offline use).
 * Runs automatically after `npm install` (the "postinstall" script).
 */
const fs = require('fs');
const path = require('path');

const src = path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist');
const dst = path.join(__dirname, '..', 'vendor', 'ort');

if (!fs.existsSync(src)) {
  console.warn('[vendor-ort] onnxruntime-web is not installed — skipping (the app will use the CDN or its built-in engine).');
  process.exit(0);
}
fs.mkdirSync(dst, { recursive: true });
const wanted = f => f === 'ort.min.js' || /^ort-wasm.*\.(wasm|mjs)$/.test(f);
let n = 0;
for (const f of fs.readdirSync(src)) {
  if (!wanted(f)) continue;
  fs.copyFileSync(path.join(src, f), path.join(dst, f));
  n++;
}
console.log(`[vendor-ort] copied ${n} files to vendor/ort/`);
