"""Static file server for leaderboard/site with correct JS MIME types.

Windows' default `python -m http.server` serves .js as text/plain, which
breaks ES module loading.  Use this script instead when previewing/sharing.

Usage (from repo root):
    python leaderboard/site/serve.py
    python leaderboard/site/serve.py --port 8000
"""

from __future__ import annotations

import argparse
import os
import posixpath
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _mime_for(path: str) -> str | None:
    _base, ext = posixpath.splitext(path.split("?", 1)[0])
    ext = ext.lower()
    if ext in {".js", ".mjs"}:
        return "application/javascript; charset=utf-8"
    if ext == ".json":
        return "application/json; charset=utf-8"
    if ext == ".wasm":
        return "application/wasm"
    if ext == ".svg":
        return "image/svg+xml"
    if ext == ".css":
        return "text/css; charset=utf-8"
    if ext == ".html":
        return "text/html; charset=utf-8"
    return None


class SiteHandler(SimpleHTTPRequestHandler):
    _force_mime: str | None = None

    def guess_type(self, path: str) -> str:
        forced = _mime_for(path)
        if forced:
            return forced
        return super().guess_type(path)

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() == "content-type" and self._force_mime:
            value = self._force_mime
        super().send_header(keyword, value)

    def end_headers(self) -> None:
        if self._force_mime:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        self._force_mime = _mime_for(self.path)
        try:
            super().do_GET()
        finally:
            self._force_mime = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve leaderboard/site")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    site_dir = Path(__file__).resolve().parent
    os.chdir(site_dir)

    handler = lambda *h_args, **h_kwargs: SiteHandler(  # noqa: E731
        *h_args, directory=str(site_dir), **h_kwargs
    )

    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {site_dir}")
        print(f"Local:  http://{args.host}:{args.port}/")
        print("MIME:   .js -> application/javascript")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
