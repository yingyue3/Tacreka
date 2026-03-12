"""Human feedback manager for video selection and task feedback.

This module provides a web-based interface for collecting human feedback
on recorded videos. It works on headless clusters (H100, A100) by hosting
a web server that can be accessed from a local machine via SSH port forwarding.
"""

import http.server
import socket
import threading
import time
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FeedbackResult:
    """Result from human feedback."""
    selected_index: int
    selected_video: str
    text_feedback: Optional[str] = None
    rating: Optional[int] = None


class HumanFeedbackManager:
    """Web-based interface for human video selection and feedback.
    
    This manager hosts a web server that can be accessed remotely,
    making it suitable for headless cluster environments.
    
    Usage on cluster (H100/A100):
        1. Run your script that calls this manager
        2. On your local machine, set up SSH port forwarding:
           ssh -L 8889:localhost:8889 user@cluster-node
        3. Open http://localhost:8889 in your local browser
        4. Watch videos, provide feedback, and submit
    """
    
    def __init__(self, port: int = 8889, timeout: int = 3600):
        """Initialize the feedback manager.
        
        Args:
            port: Port to host the web server on.
            timeout: Maximum seconds to wait for human feedback (default: 1 hour).
        """
        self.port = port
        self.timeout = timeout
        self.server = None
        self._original_dir = None
        
    def get_hostname(self) -> str:
        """Get the hostname of this machine."""
        return socket.gethostname()
    
    def _print_access_instructions(self):
        """Print instructions for accessing the web interface."""
        hostname = self.get_hostname()
        print("\n" + "=" * 60)
        print("HUMAN FEEDBACK REQUIRED")
        print("=" * 60)
        print(f"\nWeb server started on port {self.port}")
        print(f"Hostname: {hostname}")
        print("\nTo access from your local machine:")
        print(f"  1. Open a new terminal and run:")
        print(f"     ssh -L {self.port}:localhost:{self.port} user@{hostname}")
        print(f"  2. Open in browser: http://localhost:{self.port}")
        print("\nWaiting for human feedback...")
        print("=" * 60 + "\n")

    def select_video(
        self,
        video_paths: list[str],
        descriptions: list[str] = None,
        task_description: str = None,
        allow_text_feedback: bool = True,
        allow_rating: bool = False,
    ) -> FeedbackResult:
        """Display videos and collect human feedback.
        
        Args:
            video_paths: List of paths to video files.
            descriptions: Optional descriptions for each video.
            task_description: Optional task context to display.
            allow_text_feedback: Whether to show text feedback box.
            allow_rating: Whether to show rating slider.
            
        Returns:
            FeedbackResult with selection and optional feedback.
        """
        if descriptions is None:
            descriptions = [f"Option {i+1}" for i in range(len(video_paths))]
        
        # Ensure all videos exist
        for path in video_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video not found: {path}")
        
        # Create HTML in the video directory
        video_dir = os.path.dirname(os.path.abspath(video_paths[0]))
        html = self._generate_html(
            video_paths, 
            descriptions, 
            task_description,
            allow_text_feedback,
            allow_rating,
        )
        
        html_path = os.path.join(video_dir, "index.html")
        with open(html_path, 'w') as f:
            f.write(html)
        
        # Selection file for receiving feedback
        selection_file = os.path.join(video_dir, "feedback.json")
        if os.path.exists(selection_file):
            os.remove(selection_file)
        
        # Start server - bind to 0.0.0.0 to allow remote access
        self._original_dir = os.getcwd()
        os.chdir(video_dir)
        handler = self._create_handler(selection_file)
        self.server = http.server.HTTPServer(('0.0.0.0', self.port), handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        
        # Print access instructions
        self._print_access_instructions()
        
        # Wait for feedback with timeout
        start_time = time.time()
        while not os.path.exists(selection_file):
            if time.time() - start_time > self.timeout:
                self._cleanup()
                raise TimeoutError(f"No feedback received within {self.timeout} seconds")
            time.sleep(0.5)
        
        # Read result
        with open(selection_file) as f:
            data = json.load(f)
        
        self._cleanup()
        
        result = FeedbackResult(
            selected_index=data['selection'],
            selected_video=video_paths[data['selection']],
            text_feedback=data.get('feedback'),
            rating=data.get('rating'),
        )
        
        print(f"\n[Feedback received]")
        print(f"  Selected: Option {result.selected_index + 1} - {result.selected_video}")
        if result.text_feedback:
            print(f"  Feedback: {result.text_feedback}")
        if result.rating is not None:
            print(f"  Rating: {result.rating}/10")
        
        return result
    
    def _cleanup(self):
        """Clean up server and restore directory."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self._original_dir:
            os.chdir(self._original_dir)
            self._original_dir = None

    def _generate_html(
        self,
        video_paths: list[str],
        descriptions: list[str],
        task_description: str = None,
        allow_text_feedback: bool = True,
        allow_rating: bool = False,
    ) -> str:
        """Generate the HTML page for video selection."""
        video_cards = ""
        for i, (path, desc) in enumerate(zip(video_paths, descriptions)):
            video_name = os.path.basename(path)
            video_cards += f"""
            <div class="video-card" id="card-{i}">
                <h3>{desc}</h3>
                <video controls preload="metadata">
                    <source src="{video_name}" type="video/mp4">
                    Your browser does not support video.
                </video>
                <button class="select-btn" onclick="selectVideo({i})">
                    Select This One
                </button>
            </div>
            """
        
        task_section = ""
        if task_description:
            task_section = f"""
            <div class="task-info">
                <h2>Task Description</h2>
                <p>{task_description}</p>
            </div>
            """
        
        feedback_section = ""
        if allow_text_feedback:
            feedback_section = """
            <div class="feedback-section">
                <h3>Additional Feedback (Optional)</h3>
                <textarea id="feedback" placeholder="What did you like or dislike? Any suggestions?"></textarea>
            </div>
            """
        
        rating_section = ""
        if allow_rating:
            rating_section = """
            <div class="rating-section">
                <h3>Overall Rating</h3>
                <input type="range" id="rating" min="1" max="10" value="5">
                <span id="rating-value">5</span>/10
            </div>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Human Feedback - Video Selection</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%); 
            color: #e0e0e0; 
            padding: 20px;
            min-height: 100vh;
            margin: 0;
        }}
        h1 {{ 
            text-align: center; 
            color: #4cc9f0; 
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .task-info {{
            background: rgba(76, 201, 240, 0.1);
            border-left: 4px solid #4cc9f0;
            padding: 15px 20px;
            margin: 20px auto;
            max-width: 800px;
            border-radius: 0 10px 10px 0;
        }}
        .task-info h2 {{
            color: #4cc9f0;
            margin-top: 0;
        }}
        .container {{ 
            display: flex; 
            flex-wrap: wrap; 
            justify-content: center; 
            gap: 30px; 
            padding: 20px;
        }}
        .video-card {{ 
            background: rgba(255,255,255,0.05); 
            padding: 20px; 
            border-radius: 15px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            max-width: 500px;
        }}
        .video-card:hover {{
            border-color: #4cc9f0;
            transform: translateY(-5px);
        }}
        .video-card.selected {{
            border-color: #4ade80;
            background: rgba(74, 222, 128, 0.1);
        }}
        .video-card h3 {{
            color: #fff;
            margin-top: 0;
        }}
        video {{ 
            width: 100%;
            max-width: 450px;
            border-radius: 10px; 
            margin: 10px 0;
            background: #000;
        }}
        .select-btn {{ 
            background: linear-gradient(135deg, #4cc9f0 0%, #3b82f6 100%);
            border: none; 
            padding: 15px 40px; 
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px; 
            cursor: pointer;
            transition: all 0.3s;
            color: white;
        }}
        .select-btn:hover {{ 
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(76, 201, 240, 0.4);
        }}
        .feedback-section, .rating-section {{
            max-width: 600px;
            margin: 30px auto;
            text-align: center;
        }}
        textarea {{
            width: 100%;
            height: 100px;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #333;
            background: rgba(255,255,255,0.05);
            color: white;
            font-size: 14px;
            resize: vertical;
        }}
        textarea:focus {{
            outline: none;
            border-color: #4cc9f0;
        }}
        input[type="range"] {{
            width: 200px;
            margin: 10px;
        }}
        #rating-value {{
            font-size: 1.5em;
            color: #4cc9f0;
        }}
        .submit-section {{
            text-align: center;
            margin: 30px 0;
        }}
        .submit-btn {{
            background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
            border: none;
            padding: 20px 60px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 10px;
            cursor: pointer;
            color: white;
            transition: all 0.3s;
        }}
        .submit-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 20px rgba(74, 222, 128, 0.4);
        }}
        .submit-btn:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        .success-message {{
            display: none;
            text-align: center;
            padding: 40px;
            background: rgba(74, 222, 128, 0.1);
            border-radius: 15px;
            margin: 30px auto;
            max-width: 500px;
        }}
        .success-message h2 {{
            color: #4ade80;
            font-size: 2em;
        }}
        .checkmark {{
            font-size: 4em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Human Feedback Required</h1>
    <p class="subtitle">Watch the videos and select your preferred option</p>
    
    {task_section}
    
    <div class="container" id="video-container">
        {video_cards}
    </div>
    
    {feedback_section}
    {rating_section}
    
    <div class="submit-section" id="submit-section">
        <button class="submit-btn" id="submit-btn" onclick="submitFeedback()" disabled>
            Submit Selection
        </button>
        <p id="selection-hint" style="color: #888;">Please select a video first</p>
    </div>
    
    <div class="success-message" id="success-message">
        <div class="checkmark">✓</div>
        <h2>Thank You!</h2>
        <p>Your feedback has been recorded.</p>
        <p>You can close this window now.</p>
    </div>
    
    <script>
        let selectedIndex = -1;
        
        function selectVideo(index) {{
            selectedIndex = index;
            
            // Update card styles
            document.querySelectorAll('.video-card').forEach((card, i) => {{
                card.classList.toggle('selected', i === index);
            }});
            
            // Enable submit button
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('selection-hint').textContent = 
                'Option ' + (index + 1) + ' selected';
            document.getElementById('selection-hint').style.color = '#4ade80';
        }}
        
        {"document.getElementById('rating').addEventListener('input', function() { document.getElementById('rating-value').textContent = this.value; });" if allow_rating else ""}
        
        function submitFeedback() {{
            if (selectedIndex < 0) return;
            
            const data = {{
                selection: selectedIndex,
                {"feedback: document.getElementById('feedback').value," if allow_text_feedback else ""}
                {"rating: parseInt(document.getElementById('rating').value)," if allow_rating else ""}
            }};
            
            fetch('/submit', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }}).then(response => {{
                if (response.ok) {{
                    document.getElementById('video-container').style.display = 'none';
                    document.getElementById('submit-section').style.display = 'none';
                    document.querySelectorAll('.feedback-section, .rating-section').forEach(
                        el => el.style.display = 'none'
                    );
                    document.getElementById('success-message').style.display = 'block';
                }}
            }});
        }}
    </script>
</body>
</html>
        """
    
    def _create_handler(self, selection_file: str):
        """Create the HTTP request handler."""
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_POST(self):
                if self.path == '/submit':
                    length = int(self.headers['Content-Length'])
                    data = json.loads(self.rfile.read(length))
                    with open(selection_file, 'w') as f:
                        json.dump(data, f)
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                pass
                
        return Handler


# Legacy alias for backwards compatibility
HumanVideoSelector = HumanFeedbackManager