// The built-in ONNX interpreter (offline fallback in the worker) must give the
// same answers as onnxruntime. Expected outputs were produced with onnxruntime
// (Python) from the same deterministic input pattern.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

global.CalcCore = {}; global.self = {}; global.importScripts = () => {};
const { parseOnnx, runGraph } = require('../src/worker/inference-worker.js');
const expected = require('./fixtures/onnx-expected.json');

const pattern = n => Float32Array.from({ length: n }, (_, i) => ((i * 37) % 101) / 1000 + ((i * 7) % 13 === 0 ? 0.9 : 0));

for (const [name, layout, size] of [['calcink_symbols', 'NCHW', 28], ['pretrained', 'NHWC', 64]]) {
  test(`built-in engine matches onnxruntime: ${name}.onnx`, () => {
    const bytes = new Uint8Array(fs.readFileSync(path.join(__dirname, '..', 'models', `${name}.onnx`)));
    const g = parseOnnx(bytes);
    const n = expected[name].shape[0];
    const out = runGraph(g, pattern(n * size * size), n, { layout, size });
    assert.equal(out.length, expected[name].probs.length);
    out.forEach((v, i) => assert.ok(Math.abs(v - expected[name].probs[i]) < 1e-5, `${name}[${i}] ${v} vs ${expected[name].probs[i]}`));
  });
}

test('pre-trained model metadata describes a 15-class 64×64 NHWC model', () => {
  const meta = require('../models/pretrained.json');
  assert.equal(meta.classes.length, 15);
  assert.deepEqual(meta.classes.slice(10), ['+', '÷', '=', 'x', '-']);
  assert.equal(meta.size, 64);
  assert.equal(meta.layout, 'NHWC');
  assert.equal(meta.license, 'MIT');
});
