// Checks the production build: `npm test` runs this too.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');

test('build produces a deployable site and a self-contained standalone file', () => {
  execFileSync(process.execPath, [path.join(ROOT, 'scripts', 'build.js')], { cwd: ROOT });
  for (const f of ['index.html', 'sw.js', 'src/app/main.js', 'src/worker/inference-worker.js', 'models/calcink_symbols.onnx'])
    assert.ok(fs.existsSync(path.join(ROOT, 'dist/site', f)), f);
  const html = fs.readFileSync(path.join(ROOT, 'dist/calcink-standalone.html'), 'utf8');
  assert.ok(!/<script src=/.test(html), 'no external <script src> left');
  assert.ok(!/<link rel="stylesheet"/.test(html), 'CSS is inlined');
  assert.ok(html.includes('id="worker-src"') && html.includes('id="model-onnx"'));
  assert.ok(html.indexOf('id="model-onnx"') < html.indexOf('recognizer.load()'), 'model data comes before the app code');
});

test('service worker pre-caches every app file', () => {
  const sw = fs.readFileSync(path.join(ROOT, 'sw.js'), 'utf8');
  const index = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  for (const [, src] of index.matchAll(/<script src="([^"]+)"/g)) assert.ok(sw.includes(`'${src}'`), `${src} missing from sw.js`);
  assert.ok(sw.includes("'src/worker/inference-worker.js'") && sw.includes("'models/calcink_symbols.onnx'"));
});

test('no eval() or new Function() anywhere in the app code', () => {
  for (const dir of ['src/core', 'src/app', 'src/worker']) {
    for (const f of fs.readdirSync(path.join(ROOT, dir))) {
      const code = fs.readFileSync(path.join(ROOT, dir, f), 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
      assert.ok(!/\beval\s*\(|new\s+Function\s*\(/.test(code), `${dir}/${f}`);
    }
  }
});
