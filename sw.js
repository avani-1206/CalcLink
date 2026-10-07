/*
 * sw.js — service worker for 100% offline use
 * ─────────────────────────────────────────────────────────────────────────
 * A service worker sits between the page and the network. On first visit we
 * download every file the app needs into a cache; afterwards requests are
 * answered from that cache first, so the app works with zero connectivity
 * (try DevTools → Network → Offline, or airplane mode).
 *
 * Bump VERSION whenever you deploy, so users get the new files.
 */
const VERSION = 'calcink-v1.1.0';
const CORE = [
  './',
  'index.html',
  'manifest.webmanifest',
  'icon.svg',
  'src/styles.css',
  'src/core/calcink-core.js',
  'src/app/canvas.js',
  'src/app/recognizer.js',
  'src/app/projection.js',
  'src/app/main.js',
  'src/worker/inference-worker.js',
  'models/calcink_symbols.onnx',
  'models/pretrained.onnx',
  'models/pretrained.json',
];
// Optional files: cached if they exist (vendored ONNX Runtime after `npm install`).
const OPTIONAL = [
  'vendor/ort/ort.min.js',
  'vendor/ort/ort-wasm-simd-threaded.wasm',
  'vendor/ort/ort-wasm-simd-threaded.mjs',
  'vendor/ort/ort-wasm-simd-threaded.jsep.wasm',
  'vendor/ort/ort-wasm-simd-threaded.jsep.mjs',
];

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    await cache.addAll(CORE);
    await Promise.all(OPTIONAL.map(url => cache.add(url).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== VERSION) await caches.delete(key);
    await self.clients.claim();
  })());
});

// Cache-first. Anything fetched later (e.g. ONNX Runtime from the CDN) is
// stored too, so it is also available offline next time.
self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const cacheable = url.origin === location.origin || url.hostname === 'cdn.jsdelivr.net';
  if (!cacheable) return;
  event.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const hit = await cache.match(req, { ignoreSearch: true });
    if (hit) return hit;
    try {
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    } catch (err) {
      if (req.mode === 'navigate') return (await cache.match('index.html')) || Response.error();
      throw err;
    }
  })());
});
