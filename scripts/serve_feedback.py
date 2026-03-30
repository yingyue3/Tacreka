"""Standalone local website for video display and human feedback collection.

Serves a web UI that lets users watch MP4 videos and submit structured feedback
(preference selection, text comments, numeric rating). Results are saved as JSON.

Usage examples:
    # Auto-discover videos in a directory
    python scripts/serve_feedback.py --video-dir recordings/

    # Specify videos explicitly with labels
    python scripts/serve_feedback.py \\
        recordings/quad_tac_71.mp4 \\
        recordings/quad_tac_85.mp4 \\
        --labels "Reward v1" "Reward v2" \\
        --task "Quadcopter hover task"

    # On a cluster: SSH tunnel from local machine
    #   ssh -L 8889:localhost:8889 user@<cluster-hostname>
    #   then open http://localhost:8889 in your browser
"""

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Feedback — Video Review</title>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg:        #090917;
            --surface:   rgba(255,255,255,0.04);
            --border:    rgba(255,255,255,0.08);
            --accent:    #4cc9f0;
            --green:     #4ade80;
            --text:      #d4d4d8;
            --muted:     #71717a;
            --radius:    14px;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: radial-gradient(ellipse at 20% 10%, #0e1229 0%, var(--bg) 60%);
            color: var(--text);
            min-height: 100vh;
            padding: 32px 20px 60px;
        }}

        /* ── header ── */
        header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        header h1 {{
            font-size: clamp(1.8rem, 4vw, 2.8rem);
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent) 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        header p.subtitle {{
            color: var(--muted);
            font-size: 1rem;
        }}

        /* ── task info ── */
        .task-box {{
            max-width: 820px;
            margin: 0 auto 36px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 0 var(--radius) var(--radius) 0;
            padding: 18px 24px;
        }}
        .task-box h2 {{
            color: var(--accent);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 6px;
        }}
        .task-box p {{
            font-size: 1rem;
            line-height: 1.6;
        }}

        /* ── video grid ── */
        .video-grid {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 28px;
            margin-bottom: 40px;
        }}

        .video-card {{
            background: var(--surface);
            border: 2px solid var(--border);
            border-radius: var(--radius);
            padding: 22px;
            width: min(480px, 100%);
            text-align: center;
            transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
            cursor: default;
        }}
        .video-card:hover {{
            border-color: var(--accent);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(76,201,240,0.12);
        }}
        .video-card.selected {{
            border-color: var(--green);
            background: rgba(74,222,128,0.06);
            box-shadow: 0 12px 40px rgba(74,222,128,0.15);
        }}
        .video-card h3 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 14px;
        }}
        .video-card video {{
            width: 100%;
            max-height: 280px;
            border-radius: 10px;
            background: #000;
            display: block;
            margin-bottom: 16px;
        }}

        /* ── select button ── */
        .btn-select {{
            display: inline-block;
            background: linear-gradient(135deg, var(--accent) 0%, #3b82f6 100%);
            color: #fff;
            border: none;
            padding: 11px 36px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .btn-select:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(76,201,240,0.4);
        }}
        .video-card.selected .btn-select {{
            background: linear-gradient(135deg, var(--green) 0%, #16a34a 100%);
        }}

        /* ── form sections ── */
        .form-section {{
            max-width: 640px;
            margin: 0 auto 28px;
        }}
        .form-section label {{
            display: block;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 10px;
        }}
        textarea {{
            width: 100%;
            height: 110px;
            padding: 14px;
            background: var(--surface);
            border: 1.5px solid var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 0.95rem;
            resize: vertical;
            transition: border-color 0.2s;
        }}
        textarea:focus {{ outline: none; border-color: var(--accent); }}

        /* ── rating ── */
        .rating-row {{
            display: flex;
            align-items: center;
            gap: 14px;
            justify-content: center;
        }}
        .rating-row span.val {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent);
            min-width: 2.5ch;
            text-align: right;
        }}
        .rating-row span.max {{
            color: var(--muted);
        }}
        input[type=range] {{
            -webkit-appearance: none;
            appearance: none;
            width: 220px;
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            outline: none;
        }}
        input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px; height: 20px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
            box-shadow: 0 0 8px rgba(76,201,240,0.6);
        }}
        input[type=range]::-moz-range-thumb {{
            width: 20px; height: 20px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
            border: none;
        }}

        /* ── submit ── */
        .submit-wrap {{
            text-align: center;
            margin: 36px 0;
        }}
        .btn-submit {{
            background: linear-gradient(135deg, var(--green) 0%, #16a34a 100%);
            border: none;
            padding: 18px 64px;
            font-size: 1.15rem;
            font-weight: 700;
            border-radius: 10px;
            cursor: pointer;
            color: #fff;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .btn-submit:hover:not(:disabled) {{
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(74,222,128,0.4);
        }}
        .btn-submit:disabled {{
            background: #3f3f46;
            cursor: not-allowed;
        }}
        .hint {{
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.9rem;
        }}
        .hint.ok {{ color: var(--green); }}

        /* ── success overlay ── */
        #success {{
            display: none;
            text-align: center;
            padding: 60px 40px;
            background: rgba(74,222,128,0.06);
            border: 1px solid rgba(74,222,128,0.2);
            border-radius: 20px;
            max-width: 520px;
            margin: 0 auto;
        }}
        #success .check {{ font-size: 4rem; margin-bottom: 16px; }}
        #success h2 {{ color: var(--green); font-size: 2rem; margin-bottom: 12px; }}
        #success p {{ color: var(--muted); line-height: 1.6; }}

        /* ── footer ── */
        footer {{
            text-align: center;
            margin-top: 60px;
            color: var(--muted);
            font-size: 0.8rem;
        }}
        footer a {{ color: var(--accent); text-decoration: none; }}
    </style>
</head>
<body>

<header>
    <h1>Human Feedback Required</h1>
    <p class="subtitle">Watch the videos below and select your preferred policy</p>
</header>

{task_section}

<div class="video-grid" id="video-grid">
{video_cards}
</div>

{feedback_section}
{rating_section}

<div class="submit-wrap" id="submit-wrap">
    <button class="btn-submit" id="btn-submit" onclick="submitFeedback()" disabled>
        Submit Feedback
    </button>
    <p class="hint" id="hint">Please select a video first</p>
</div>

<div id="success">
    <div class="check">✓</div>
    <h2>Thank You!</h2>
    <p>Your feedback has been recorded and the training loop will continue.<br>You can now close this window.</p>
</div>

<footer>
    Tacreka · Human-in-the-Loop RL Feedback · <a href="/api/results" target="_blank">View saved results</a>
</footer>

<script>
    let selectedIndex = -1;

    function selectVideo(index) {{
        selectedIndex = index;
        document.querySelectorAll('.video-card').forEach((card, i) => {{
            card.classList.toggle('selected', i === index);
        }});
        document.getElementById('btn-submit').disabled = false;
        const hint = document.getElementById('hint');
        hint.textContent = 'Option ' + (index + 1) + ' selected — ready to submit';
        hint.className = 'hint ok';
    }}

    {rating_js}

    function submitFeedback() {{
        if (selectedIndex < 0) return;
        const btn = document.getElementById('btn-submit');
        btn.disabled = true;
        btn.textContent = 'Submitting…';

        const data = {{
            selection: selectedIndex,
            {feedback_js}
            {rating_collect_js}
            timestamp: new Date().toISOString(),
        }};

        fetch('/submit', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(data)
        }}).then(r => {{
            if (r.ok) {{
                document.getElementById('video-grid').style.display = 'none';
                document.getElementById('submit-wrap').style.display = 'none';
                document.querySelectorAll('.form-section, .task-box').forEach(
                    el => el.style.display = 'none'
                );
                document.getElementById('success').style.display = 'block';
            }} else {{
                btn.disabled = false;
                btn.textContent = 'Submit Feedback';
                alert('Submission failed — please try again.');
            }}
        }}).catch(() => {{
            btn.disabled = false;
            btn.textContent = 'Submit Feedback';
            alert('Network error — please try again.');
        }});
    }}
</script>
</body>
</html>
"""


def _build_html(
    video_names: list[str],
    labels: list[str],
    task_description: str | None,
    allow_text_feedback: bool,
    allow_rating: bool,
) -> str:
    # video cards
    cards = ""
    for i, (name, label) in enumerate(zip(video_names, labels)):
        cards += f"""    <div class="video-card" id="card-{i}">
        <h3>{label}</h3>
        <video controls preload="metadata">
            <source src="/videos/{name}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <button class="btn-select" onclick="selectVideo({i})">Select This One</button>
    </div>\n"""

    task_section = ""
    if task_description:
        task_section = f"""<div class="task-box">
    <h2>Task</h2>
    <p>{task_description}</p>
</div>\n"""

    feedback_section = ""
    if allow_text_feedback:
        feedback_section = """<div class="form-section">
    <label for="feedback">Additional Feedback <span style="text-transform:none;letter-spacing:0;color:#52525b">(optional)</span></label>
    <textarea id="feedback" placeholder="What did you like or dislike? Any specific observations?"></textarea>
</div>\n"""

    rating_section = ""
    rating_js = ""
    rating_collect_js = ""
    if allow_rating:
        rating_section = """<div class="form-section" style="text-align:center">
    <label>Overall Quality Rating</label>
    <div class="rating-row">
        <span style="color:var(--muted);font-size:.85rem">1</span>
        <input type="range" id="rating" min="1" max="10" value="5">
        <span class="val" id="rating-val">5</span><span class="max">/10</span>
    </div>
</div>\n"""
        rating_js = "document.getElementById('rating').addEventListener('input', e => { document.getElementById('rating-val').textContent = e.target.value; });"
        rating_collect_js = "rating: parseInt(document.getElementById('rating').value),"

    feedback_js = "feedback: document.getElementById('feedback') ? document.getElementById('feedback').value : ''," if allow_text_feedback else ""

    return HTML_TEMPLATE.format(
        task_section=task_section,
        video_cards=cards,
        feedback_section=feedback_section,
        rating_section=rating_section,
        rating_js=rating_js,
        feedback_js=feedback_js,
        rating_collect_js=rating_collect_js,
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class FeedbackHandler(http.server.BaseHTTPRequestHandler):

    video_dir: Path = Path(".")
    html_content: str = ""
    output_file: Path = Path("feedback.json")
    _feedback_received = threading.Event()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(self.html_content)
        elif self.path.startswith("/videos/"):
            video_name = self.path[len("/videos/"):]
            video_path = self.video_dir / video_name
            self._send_file(video_path, "video/mp4")
        elif self.path == "/api/results":
            self._send_results()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/submit":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return

        # Append to results file (list of feedback objects)
        results = []
        if self.output_file.exists():
            with open(self.output_file) as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = []
        if not isinstance(results, list):
            results = [results]
        results.append(data)
        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[✓] Feedback received:")
        print(f"    Selection : Option {data.get('selection', '?') + 1} — {data.get('selected_video', '')}")
        if data.get("feedback"):
            print(f"    Comment   : {data['feedback']}")
        if data.get("rating") is not None:
            print(f"    Rating    : {data['rating']}/10")
        print(f"    Saved to  : {self.output_file}\n")

        self._feedback_received.set()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def _send_html(self, content: str):
        encoded = content.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def _send_results(self):
        if self.output_file.exists():
            content = self.output_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            content = b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Serve a local web UI for video feedback collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "videos",
        nargs="*",
        help="Explicit list of video file paths. If omitted, all .mp4 files in --video-dir are used.",
    )
    p.add_argument(
        "--video-dir", "-d",
        default=None,
        help="Directory to scan for .mp4 files (used when no explicit videos given).",
    )
    p.add_argument(
        "--labels", "-l",
        nargs="*",
        help="Display labels for each video (must match number of videos).",
    )
    p.add_argument(
        "--task", "-t",
        default=None,
        help="Task description shown at the top of the page.",
    )
    p.add_argument(
        "--port", "-p",
        type=int,
        default=8889,
        help="Port to bind (default: 8889).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save feedback JSON. Defaults to <video-dir>/feedback_results.json.",
    )
    p.add_argument(
        "--no-text-feedback",
        action="store_true",
        help="Hide the text comment box.",
    )
    p.add_argument(
        "--rating",
        action="store_true",
        help="Show a 1–10 rating slider.",
    )
    p.add_argument(
        "--wait",
        action="store_true",
        help="Block until one feedback submission is received, then exit.",
    )
    return p.parse_args()


def resolve_videos(args) -> tuple[list[Path], Path]:
    """Return (list_of_video_paths, video_dir)."""
    if args.videos:
        paths = [Path(v).resolve() for v in args.videos]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"[ERROR] Videos not found: {missing}", file=sys.stderr)
            sys.exit(1)
        video_dir = paths[0].parent
        return paths, video_dir

    scan_dir = Path(args.video_dir).resolve() if args.video_dir else Path.cwd()
    if not scan_dir.is_dir():
        print(f"[ERROR] Not a directory: {scan_dir}", file=sys.stderr)
        sys.exit(1)
    paths = sorted(scan_dir.glob("*.mp4"))
    if not paths:
        print(f"[ERROR] No .mp4 files found in {scan_dir}", file=sys.stderr)
        sys.exit(1)
    return paths, scan_dir


def print_instructions(port: int, video_count: int, output_file: Path):
    hostname = socket.gethostname()
    print()
    print("=" * 62)
    print("  VIDEO FEEDBACK SERVER READY")
    print("=" * 62)
    print(f"  Videos loaded : {video_count}")
    print(f"  Results file  : {output_file}")
    print(f"  Hostname      : {hostname}")
    print()
    print("  ▶  Local access:   http://localhost:{port}".format(port=port))
    print(f"  ▶  Remote access:")
    print(f"       1. On your local machine open a new terminal and run:")
    print(f"            ssh -L {port}:localhost:{port} user@{hostname}")
    print(f"       2. Then open: http://localhost:{port}")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 62)
    print()


def main():
    args = parse_args()
    video_paths, video_dir = resolve_videos(args)

    labels = args.labels
    if labels is None:
        labels = [f"Reward v{i+1}" for i in range(len(video_paths))]
    elif len(labels) != len(video_paths):
        print(f"[ERROR] --labels count ({len(labels)}) != video count ({len(video_paths)})", file=sys.stderr)
        sys.exit(1)

    output_file = Path(args.output).resolve() if args.output else video_dir / "feedback_results.json"

    video_names = [p.name for p in video_paths]
    html = _build_html(
        video_names=video_names,
        labels=labels,
        task_description=args.task,
        allow_text_feedback=not args.no_text_feedback,
        allow_rating=args.rating,
    )

    # Patch handler class with shared state (avoids global variables)
    FeedbackHandler.video_dir = video_dir
    FeedbackHandler.html_content = html
    FeedbackHandler.output_file = output_file
    FeedbackHandler._feedback_received = threading.Event()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), FeedbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print_instructions(args.port, len(video_paths), output_file)
    for i, (p, lbl) in enumerate(zip(video_paths, labels)):
        print(f"    [{i+1}] {lbl:20s}  {p}")
    print()

    try:
        if args.wait:
            print("Waiting for one feedback submission…")
            FeedbackHandler._feedback_received.wait()
            server.shutdown()
            print("Server stopped after receiving feedback.")
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server…")
        server.shutdown()


if __name__ == "__main__":
    main()
