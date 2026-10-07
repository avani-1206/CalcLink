// Run with:  npm test   (or: node --test)
const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../src/core/calcink-core.js');

const val = (expr, vars) => {
  const r = C.evaluate(expr, vars);
  assert.equal(r.status, 'ok', `${expr} → ${JSON.stringify(r)}`);
  return r.value;
};
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} ≈ ${b}`);

test('operator precedence (BODMAS)', () => {
  assert.equal(val('18+4×3'), 30);
  assert.equal(val('2+3×4-5'), 9);
  assert.equal(val('(2+3)×4'), 20);
  assert.equal(val('8÷2×4'), 16);          // left-to-right for same precedence
  assert.equal(val('100-10-10'), 80);
  assert.equal(val('2×(3+(4-1))×2'), 24);
  assert.equal(val('10-2×3+8÷4'), 6);
});

test('multi-digit and decimal numbers', () => {
  assert.equal(val('123+877'), 1000);
  assert.equal(val('3.5×2'), 7);
  assert.equal(val('.5+.5'), 1);
  near(val('0.1+0.2'), 0.3);
  assert.equal(C.formatNumber(val('0.1+0.2')), '0.3');
  assert.equal(val('10÷4'), 2.5);
});

test('negative numbers (unary minus)', () => {
  assert.equal(val('-5+3'), -2);
  assert.equal(val('2×-3'), -6);
  assert.equal(val('-(2+3)'), -5);
  assert.equal(val('--2'), 2);
  assert.equal(val('5--2'), 7);
  assert.equal(val('-2×3'), -6);
  assert.equal(val('+4'), 4);
  assert.equal(val('3÷-2'), -1.5);
});

test('division by zero gives Undefined, not a crash', () => {
  for (const e of ['5÷0', '0÷0', '1÷(2-2)', '7÷(3×0)', '-4÷0']) {
    assert.equal(C.evaluate(e).status, 'undefined', e);
  }
  assert.equal(C.displayText(C.evaluateLine('5÷0=')), 'Undefined');
});

test('malformed input returns an error object and never throws', () => {
  for (const e of ['2++', '×3', '(2+3', '2+3)', '1.2.3', '', '()', '(+)', '÷', '.', '3(', ')(', '2+', '=', '9÷']) {
    const r = C.evaluate(e);
    assert.equal(r.status, 'error', `${e} → ${JSON.stringify(r)}`);
  }
});

test('random garbage never throws (fuzz)', () => {
  const alphabet = [...'0123456789+-×÷=().xy'];
  let seed = 7;
  const rand = n => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n; };
  for (let k = 0; k < 5000; k++) {
    const len = 1 + rand(12);
    const s = Array.from({ length: len }, () => alphabet[rand(alphabet.length)]);
    assert.doesNotThrow(() => C.evaluateDocument([s, s.slice().reverse()]));
    const r = C.evaluate(s);
    assert.ok(['ok', 'error', 'undefined'].includes(r.status));
  }
});

test('implicit multiplication', () => {
  assert.equal(val('2(3+4)'), 14);
  assert.equal(val('(2)(3)'), 6);
  assert.equal(val('3x', { x: 4 }), 12);
  assert.equal(val('xy', { x: 2, y: 5 }), 10);
});

test('handwritten x: multiply between numbers, variable otherwise', () => {
  assert.deepEqual(C.resolveX([...'3x5']), [...'3×5']);
  assert.deepEqual(C.resolveX([...'(2)x(4)']), [...'(2)×(4)']);
  assert.deepEqual(C.resolveX([...'x=5']), [...'x=5']);
  assert.deepEqual(C.resolveX([...'x+10']), [...'x+10']);
  assert.equal(val('3x5'), 15);
  assert.equal(C.evaluate('x+1').status, 'error');           // x not defined yet
});

test('lines: result, assignment, incomplete, ignored', () => {
  assert.equal(C.evaluateLine('12+').kind, 'incomplete');
  assert.equal(C.evaluateLine('2+2=4').kind, 'ignored');
  const a = C.evaluateLine('x=5');
  assert.equal(a.kind, 'assign'); assert.equal(a.value, 5);
  assert.equal(C.displayText(a), null);                      // nothing drawn for x=5
  const b = C.evaluateLine('x=2+3=');
  assert.equal(b.kind, 'assign'); assert.equal(C.displayText(b), '5');
  assert.equal(C.displayText(C.evaluateLine('3-10=')), '−7');
  assert.equal(C.displayText(C.evaluateLine('2++=')), '?');
});

test('variables flow down the page', () => {
  const r = C.evaluateDocument([[...'x=5'], [...'x+10='], [...'y=x×2'], [...'y+1='], [...'xy=']]);
  assert.deepEqual(r.map(C.displayText), [null, '15', null, '11', '50']);
  // a line ABOVE the definition can't use it
  const r2 = C.evaluateDocument([[...'x+1='], [...'x=3']]);
  assert.equal(C.displayText(r2[0]), '?');
  // re-assignment affects only lines below it
  const r3 = C.evaluateDocument([[...'x=1'], [...'x='], [...'x=2'], [...'x=']]);
  assert.deepEqual(r3.map(C.displayText), [null, '1', null, '2']);
});

test('number formatting', () => {
  assert.equal(C.formatNumber(1 / 3), '0.333333333333');
  assert.equal(C.formatNumber(-0), '0');
  assert.equal(C.formatNumber(2.5), '2.5');
  assert.equal(C.formatNumber(1e21), '1e+21');
  assert.equal(C.formatNumber(123456789), '123456789');
});

/* ── geometry / coordinate tests ─────────────────────────────────────── */
const line = (x0, y0, x1, y1, n = 10) =>
  ({ pts: Array.from({ length: n + 1 }, (_, i) => ({ x: x0 + (x1 - x0) * i / n, y: y0 + (y1 - y0) * i / n })) });
const dot = (x, y) => ({ pts: [{ x, y }] });

test('segment: "1+2" on one line, "=" bars and "÷" dots stay together', () => {
  const strokes = [
    line(10, 0, 10, 30),                         // 1
    line(25, 15, 45, 15), line(35, 5, 35, 25),   // +
    line(55, 0, 70, 30),                         // something
    line(80, 10, 100, 10), line(80, 20, 100, 20),// =
    line(110, 15, 130, 15), dot(120, 5), dot(120, 25), // ÷
  ];
  const [l] = C.segment(strokes);
  assert.equal(C.segment(strokes).length, 1);
  assert.deepEqual(l.symbols.map(s => s.strokes.length), [1, 2, 1, 2, 3]);
});

test('segment: two lines are ordered top to bottom', () => {
  const lines = C.segment([line(0, 100, 0, 130), line(0, 0, 0, 30)]);
  assert.equal(lines.length, 2);
  assert.ok(lines[0].box.y0 < lines[1].box.y0);
});

test('decimal point detection by size', () => {
  const lineBox = { x0: 0, y0: 0, x1: 100, y1: 40 };
  assert.ok(C.isDecimalPoint({ x0: 50, y0: 37, x1: 52, y1: 39 }, lineBox));
  assert.ok(!C.isDecimalPoint({ x0: 50, y0: 0, x1: 70, y1: 40 }, lineBox));
});

test('result anchor sits right of "=" and is centred on it', () => {
  const eq = { x0: 200, y0: 50, x1: 230, y1: 70 };
  const a = C.resultAnchor(eq, { x0: 0, y0: 40, x1: 230, y1: 80 });
  assert.ok(a.x > eq.x1 && a.x < eq.x1 + 30);
  assert.equal(a.y, 60);
  assert.equal(a.fontSize, 38);
  // font size is clamped for tiny / huge handwriting
  const tinyEq = { x0: 0, y0: 0, x1: 4, y1: 3 };
  assert.equal(C.resultAnchor(tinyEq, { x0: 0, y0: 0, x1: 4, y1: 5 }).fontSize, 18);
  assert.equal(C.resultAnchor(eq, { x0: 0, y0: 0, x1: 1, y1: 500 }).fontSize, 72);
});

/* ── Phase 5: gestures, plotting, model combining ────────────────────── */
const zigzag = (x0, y0, w, h, turns, n = 12) => {
  const pts = [];
  for (let t = 0; t <= turns; t++)
    for (let i = 0; i < n; i++) {
      const f = i / n;
      pts.push({ x: x0 + (t % 2 ? w * (1 - f) : w * f), y: y0 + h * (t + f) / (turns + 1) });
    }
  return pts;
};

test('scratch gesture: a scribble is detected, normal symbols are not', () => {
  assert.ok(C.isScratch(zigzag(0, 0, 60, 30, 8)));
  const eight = { pts: [] };
  for (let t = 0; t <= 2 * Math.PI; t += 0.1) eight.pts.push({ x: 15 + 14 * Math.sin(2 * t), y: 20 - 20 * Math.cos(t) });
  assert.ok(!C.isScratch(eight.pts), '8 is not a scribble');
  assert.ok(!C.isScratch(line(0, 0, 100, 0, 40).pts), 'a line is not a scribble');
  const M = [{ x: 0, y: 30 }, { x: 8, y: 0 }, { x: 16, y: 20 }, { x: 24, y: 0 }, { x: 32, y: 30 }];
  assert.ok(!C.isScratch(line(0, 0, 0, 0).pts.concat(M.flatMap((p, i) => i ? line(M[i - 1].x, M[i - 1].y, p.x, p.y, 6).pts : []))), 'M is not a scribble');
  assert.ok(!C.isScratch(zigzag(0, 0, 4, 4, 8)), 'too tiny');
});

test('scratch targets: only strokes mostly under the scribble', () => {
  const s1 = { id: 1, ...line(10, 10, 20, 30) };     // under the scribble
  const s2 = { id: 2, ...line(200, 10, 220, 30) };   // far away
  const s3 = { id: 3, ...line(50, 10, 150, 10) };    // only partly covered
  const ids = C.scratchTargets(zigzag(0, 0, 60, 40, 8), [s1, s2, s3]);
  assert.deepEqual(ids, [1]);
});

test('countReversals ignores small wiggles', () => {
  assert.equal(C.countReversals([0, 10, 0, 10, 0], 3), 3);
  assert.equal(C.countReversals([0, 1, 0, 1, 0, 1], 3), 0);
});

test('y = f(x) becomes a plot when x is unknown', () => {
  const r = C.evaluateLine('y=2x+1');
  assert.equal(r.kind, 'plot');
  const pts = C.plotSamples(r.expr, {}, { xmin: 0, xmax: 2, n: 3 });
  assert.deepEqual(pts.map(p => p.y), [1, 3, 5]);
  // division by zero leaves a gap instead of breaking the curve
  const g = C.plotSamples([...'1÷x'], {}, { xmin: -1, xmax: 1, n: 3 });
  assert.deepEqual(g.map(p => p.y), [-1, null, 1]);
  // once x is defined, y = … is a normal assignment again
  const doc = C.evaluateDocument([[...'x=3'], [...'y=2x+1'], [...'y=']]);
  assert.equal(doc[1].kind, 'assign');
  assert.equal(C.displayText(doc[2]), '7');
  assert.equal(C.displayText(C.evaluateLine('y=2x+1')), null);   // plots draw no text
  assert.equal(C.plotSamples([...'2++'], {}), null);
});

test('combinePredictions: primary wins unless secondary is sure about ( ) y', () => {
  const primary = { classes: ['1', '7', '='], probs: [0.9, 0.05, 0.05, 0.6, 0.3, 0.1, 0.2, 0.2, 0.6] };
  const secondary = { classes: ['1', '(', 'y'], probs: [0.9, 0.1, 0, 0.02, 0.97, 0.01, 0.3, 0, 0.7] };
  const r = C.combinePredictions(primary, secondary, 3);
  assert.deepEqual(r.map(p => p.char), ['1', '(', '=']);   // 3rd: y only 0.7 sure → keep primary
  assert.deepEqual(C.combinePredictions(primary, null, 3).map(p => p.char), ['1', '1', '=']);
});

test('segment: a decimal point squeezed against a digit stays separate; ÷ dots still join', () => {
  const two = line(0, 0, 18, 30), dotLow = dot(19, 29), five = line(21, 0, 36, 30);   // "2.5" with no gaps
  const [l] = C.segment([two, dotLow, five]);
  assert.equal(l.symbols.length, 3);
  assert.ok(C.isDecimalPoint(l.symbols[1].box, l.box));
  const bar = line(20, 15, 40, 15), d1 = dot(30, 6), d2 = dot(30, 24);             // "÷"
  const [m] = C.segment([line(0, 0, 0, 30), bar, d1, d2]);
  assert.deepEqual(m.symbols.map(s => s.strokes.length), [1, 3]);
});

/* ── BODMAS: Orders (powers) and the step trace ─────────────────────── */
test('powers: Orders come before ×÷ and +−, right-associative, above unary minus', () => {
  assert.equal(val('2^3'), 8);
  assert.equal(val('2+3^2'), 11);            // O before A
  assert.equal(val('2×3^2'), 18);            // O before M
  assert.equal(val('(2+3)^2'), 25);          // B before O
  assert.equal(val('2^3^2'), 512);           // right-associative: 2^(3^2)
  assert.equal(val('-2^2'), -4);             // −(2²), as in standard maths
  assert.equal(val('2^-1'), 0.5);
  assert.equal(val('4^0.5'), 2);
  assert.equal(val('x^2+1', { x: 3 }), 10);
  assert.equal(C.evaluate('0^-1').status, 'undefined');   // 1/0
  assert.equal(C.evaluate('(-8)^0.5').status, 'undefined'); // not a real number
  assert.equal(C.evaluate('2^').status, 'error');
  assert.equal(C.evaluate('^2').status, 'error');
});

test('full BODMAS precedence table', () => {
  const cases = { '6+4÷2': 8, '6÷2×3': 9, '6-2+1': 5, '2×(3+4)^2': 98, '100÷10÷2': 5,
                  '3+4×2÷(1-5)^2': 3.5, '1+2×3-4÷2+2^2': 9, '-3^2+10': 1, '(1+2)×(3+4)': 21 };
  for (const [e, v] of Object.entries(cases)) assert.equal(val(e), v, e);
});

test('steps show the order of operations with their BODMAS letter', () => {
  const r = C.evaluate('18+4×3');
  assert.deepEqual(r.steps.map(s => `${s.rule} ${s.text}`), ['M 4 × 3 = 12', 'A 18 + 12 = 30']);
  const b = C.evaluate('(2+3)^2');
  assert.deepEqual(b.steps.map(s => s.rule), ['B', 'O']);
  assert.equal(C.evaluate('5÷0').steps[0].text, '5 ÷ 0 = undefined');
  assert.equal(C.evaluate('2-5').steps[0].text, '2 − 5 = −3');
});

const sym = (char, x0, y0, x1, y1) => ({ char, box: { x0, y0, x1, y1 } });
test('superscripts become powers: 2³+1 → 2^3+1, x²⁺¹ → x^(2+1)', () => {
  const line = { symbols: [sym('2', 0, 10, 18, 40), sym('3', 20, 2, 28, 16), sym('+', 34, 18, 50, 34), sym('1', 56, 10, 60, 40), sym('=', 66, 20, 84, 30)] };
  assert.deepEqual(C.lineChars(line), [...'2^3+1=']);
  assert.equal(C.displayText(C.evaluateLine(C.lineChars(line))), '9');
  const grp = { symbols: [sym('x', 0, 15, 20, 40), sym('2', 22, 4, 28, 14), sym('+', 30, 5, 36, 11), sym('1', 38, 4, 41, 14)] };
  assert.deepEqual(C.lineChars(grp), [...'x^(2+1)']);
});

test('superscripts: a normal minus or a same-size digit is NOT a power', () => {
  const minus = { symbols: [sym('5', 0, 10, 18, 40), sym('-', 22, 24, 36, 26), sym('2', 40, 10, 58, 40)] };
  assert.deepEqual(C.lineChars(minus), [...'5-2']);
  const same = { symbols: [sym('1', 0, 10, 6, 40), sym('2', 10, 8, 28, 38)] };
  assert.deepEqual(C.lineChars(same), [...'12']);
  const dotLine = { symbols: [sym('1', 0, 10, 6, 40), sym('.', 8, 37, 10, 39), sym('5', 12, 10, 28, 40)] };
  assert.deepEqual(C.lineChars(dotLine), [...'1.5']);
});

test('segment: two sums side by side on one row become two lines', () => {
  const left = [line(0, 0, 0, 30), line(10, 15, 25, 15)];         // "1-"
  const right = [line(200, 0, 200, 30), line(210, 15, 225, 15)];  // far to the right
  const lines = C.segment([...right, ...left]);
  assert.equal(lines.length, 2);
  assert.ok(lines[0].box.x0 < lines[1].box.x0);
  // normal spacing stays one line
  assert.equal(C.segment([line(0, 0, 0, 30), line(20, 0, 20, 30)]).length, 1);
});
