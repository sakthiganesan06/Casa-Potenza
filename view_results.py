"""
view_results.py — Launches a local interactive visual dashboard for RAG evaluation results.

Usage:
    python view_results.py
"""
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
PORT = 8585


def get_latest_results():
    if not RESULTS_DIR.exists():
        return None
    json_files = sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not json_files:
        return None
    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            return json.load(f), json_files[0].name
    except Exception:
        return None


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/results":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            all_runs = []
            if RESULTS_DIR.exists():
                for p in sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_runs.append({"filename": p.name, "data": data})
                    except Exception:
                        pass
            self.wfile.write(json.dumps(all_runs).encode("utf-8"))
            return
            
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(ROOT_DIR / "results_dashboard.html", "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode("utf-8"))
            return

        super().do_GET()


def main():
    os.chdir(str(ROOT_DIR))
    print(f"Starting Evaluation Dashboard at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    
    # Try opening browser automatically
    webbrowser.open(f"http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
