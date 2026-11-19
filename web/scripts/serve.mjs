/**
 * A minimal static file server for previewing dist/ the way GitHub Pages serves it (and for the
 * end-to-end tests). Only for local use: it serves files under the given folder and nothing else.
 *
 *   node scripts/serve.mjs dist [port]
 */
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".map": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".zip": "application/zip",
  ".txt": "text/plain; charset=utf-8",
};

export function serve(dir, port = 4173) {
  const root = resolve(dir);
  const server = createServer((req, res) => {
    const path = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
    let file = normalize(join(root, path));
    if (file !== root && !file.startsWith(root + sep)) {
      res.writeHead(403).end();
      return;
    }
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
    if (!existsSync(file)) {
      res.writeHead(404, { "content-type": "text/plain" }).end("Not found");
      return;
    }
    res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(port, "127.0.0.1", () => ok(server)));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [dir = "dist", port = "4173"] = process.argv.slice(2);
  await serve(dir, Number(port));
  console.log(`Serving ${dir}/ at http://127.0.0.1:${port}/`);
}
