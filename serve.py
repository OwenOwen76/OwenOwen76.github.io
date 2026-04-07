#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/javascript"
        return super().guess_type(path)


if __name__ == "__main__":
    port = 8080
    print(f"Serving on http://localhost:{port}")
    HTTPServer(("", port), Handler).serve_forever()
