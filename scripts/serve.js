#!/usr/bin/env node
/*
 * Tiny static file server (zero dependencies).
 *   npm run dev              → serves the project folder on http://localhost:5173
 *   npm run preview          → serves the production build in dist/site
 *   PORT=8080 npm run dev    → another port
 * Why not just double-click index.html? Browsers block fetch() and workers on
 * file:// pages, so the model couldn't load. (dist/calcink-standalone.html is
 * the exception — everything is inlined into it.)
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = path.resolve(process.argv[2] || '.');
const port = Number(process.env.PORT) || 5173;
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
};

const server = http.createServer((req, res) => {
  let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (rel.endsWith('/')) rel += 'index.html';
  const file = path.join(root, rel);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  });
});

if (require.main === module) {
  server.listen(port, () => console.log(`CalcInk running at http://localhost:${port}  (serving ${root})`));
}
module.exports = server;
