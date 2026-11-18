"""Serve the evaluator UI (runs on Laptop 2).

    python ui/serve.py --port 8080
Then open http://localhost:8080/?api=http://<laptop1-ip>:8000
"""
import argparse
import functools
import http.server
import os

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(os.path.abspath(__file__)))
print(f"PathSnap UI on http://0.0.0.0:{args.port}/  (add ?api=http://<api-host>:8000)")
http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
