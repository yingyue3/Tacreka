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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class FeatureSpec:
    """Detailed specification for a reward feature."""
    feature_name: str
    intent: str = ""
    measurable_signals: List[str] = field(default_factory=list)
    proxy_metric: str = ""
    weight: float = 1.0
    desired_direction: str = ""  # "minimize" or "maximize"
    typical_failure_mode: str = ""


@dataclass
class RewardInfo:
    """Information about a reward function for display."""
    name: str = ""
    code: str = ""
    features: Dict[str, Any] = field(default_factory=dict)  # Simple key-value features
    feature_specs: List[FeatureSpec] = field(default_factory=list)  # Detailed feature specs
    # metrics: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class FeedbackResult:
    """Result from human feedback."""
    selected_index: int
    selected_video: str
    text_feedback: Optional[str] = None
    rating: Optional[int] = None
    selected_reward_info: Optional[RewardInfo] = None


@dataclass
class FeatureSelectionResult:
    """Result from feature selection feedback."""
    selected_features: List[FeatureSpec]
    removed_features: List[FeatureSpec]
    selected_indices: List[int]
    removed_indices: List[int]
    text_feedback: Optional[str] = None


@dataclass
class FeatureRankingResult:
    """Result from feature ranking + filtering feedback.

    Attributes:
        ranked_features: Kept features in the human's preferred order
            (index 0 = most important).
        dropped_features: Features the human chose to remove entirely.
        ranked_indices: Original 0-based indices of ``ranked_features``
            in the order supplied by the human.
        dropped_indices: Original 0-based indices of ``dropped_features``.
        text_feedback: Optional free-text comment from the human.
    """
    ranked_features: List[Any]
    dropped_features: List[Any]
    ranked_indices: List[int]
    dropped_indices: List[int]
    text_feedback: Optional[str] = None


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
        print(f"     ssh -L {self.port}:{hostname}:{self.port} user@LOGIN_NODE")
        print(f"  2. Open in browser: http://localhost:{self.port}")
        print("\nWaiting for human feedback...")
        print("=" * 60 + "\n")

    def select_video(
        self,
        video_paths: List[str] = None,
        descriptions: List[str] = None,
        task_description: str = None,
        reward_infos: List[RewardInfo] = None,
        allow_text_feedback: bool = True,
        allow_rating: bool = False,
        output_dir: str = None,
    ) -> FeedbackResult:
        """Display videos and/or reward info and collect human feedback.
        
        Args:
            video_paths: Optional list of paths to video files. Can be None or empty
                if only comparing reward functions.
            descriptions: Optional descriptions for each option.
            task_description: Optional task context to display.
            reward_infos: Optional list of RewardInfo for each option with
                reward function details (code, features, metrics).
            allow_text_feedback: Whether to show text feedback box.
            allow_rating: Whether to show rating slider.
            output_dir: Directory to save HTML file. Required if video_paths is empty.
            
        Returns:
            FeedbackResult with selection and optional feedback.
        """
        # Handle empty or None video_paths
        if video_paths is None:
            video_paths = []
        
        # Determine number of options from videos or reward_infos
        if video_paths:
            num_options = len(video_paths)
        elif reward_infos:
            num_options = len(reward_infos)
        else:
            raise ValueError("Must provide either video_paths or reward_infos")
        
        if descriptions is None:
            descriptions = [f"Option {i+1}" for i in range(num_options)]
        
        if reward_infos is None:
            reward_infos = [None] * num_options
        
        # Ensure all videos exist (skip "NONE" or None entries)
        for path in video_paths:
            if path and path.upper() != "NONE" and not os.path.exists(path):
                raise FileNotFoundError(f"Video not found: {path}")
        
        # Determine output directory - find first valid video path
        valid_video_paths = [p for p in video_paths if p and p.upper() != "NONE"]
        if valid_video_paths:
            video_dir = os.path.dirname(os.path.abspath(valid_video_paths[0]))
        elif output_dir:
            video_dir = os.path.abspath(output_dir)
            os.makedirs(video_dir, exist_ok=True)
        else:
            video_dir = os.getcwd()
        html = self._generate_html(
            video_paths, 
            descriptions, 
            task_description,
            reward_infos,
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
        
        selected_idx = data['selection']
        selected_video = None
        if video_paths and selected_idx < len(video_paths):
            path = video_paths[selected_idx]
            if path and path.upper() != "NONE":
                selected_video = path
        result = FeedbackResult(
            selected_index=selected_idx,
            selected_video=selected_video,
            text_feedback=data.get('feedback'),
            rating=data.get('rating'),
            selected_reward_info=reward_infos[selected_idx] if reward_infos else None,
        )
        
        print(f"\n[Feedback received]")
        print(f"  Selected: Option {result.selected_index + 1}" + (f" - {result.selected_video}" if result.selected_video else ""))
        if result.text_feedback:
            print(f"  Feedback: {result.text_feedback}")
        if result.rating is not None:
            print(f"  Rating: {result.rating}/10")
        
        return result

    def select_features(
        self,
        feature_specs: List,
        task_description: str = None,
        allow_text_feedback: bool = True,
        output_dir: str = None,
        preselected: List[int] = None,
    ) -> FeatureSelectionResult:
        """Display features and let human select which ones to keep.
        
        Args:
            feature_specs: List of FeatureSpec objects or dicts to choose from.
            task_description: Optional task context to display.
            allow_text_feedback: Whether to show text feedback box.
            output_dir: Directory to save HTML file.
            preselected: List of indices to pre-select (default: all selected).
            
        Returns:
            FeatureSelectionResult with selected and removed features.
        """
        if not feature_specs:
            raise ValueError("Must provide at least one feature")
        
        if output_dir is None:
            output_dir = os.getcwd()
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if preselected is None:
            preselected = list(range(len(feature_specs)))
        
        # Generate HTML
        html = self._generate_feature_selection_html(
            feature_specs,
            task_description,
            allow_text_feedback,
            preselected,
        )
        
        html_path = os.path.join(output_dir, "feature_selection.html")
        with open(html_path, 'w') as f:
            f.write(html)
        
        # Selection file
        selection_file = os.path.join(output_dir, "feature_selection.json")
        if os.path.exists(selection_file):
            os.remove(selection_file)
        
        # Start server
        self._original_dir = os.getcwd()
        os.chdir(output_dir)
        handler = self._create_handler(
            selection_file, endpoint="/submit_features",
            default_page="feature_selection.html",
        )
        self.server = http.server.HTTPServer(('0.0.0.0', self.port), handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        
        # Print instructions
        self._print_access_instructions()
        
        # Wait for feedback
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
        
        selected_indices = data.get('selected', [])
        all_indices = set(range(len(feature_specs)))
        removed_indices = list(all_indices - set(selected_indices))
        
        selected_features = [feature_specs[i] for i in selected_indices]
        removed_features = [feature_specs[i] for i in removed_indices]
        
        result = FeatureSelectionResult(
            selected_features=selected_features,
            removed_features=removed_features,
            selected_indices=selected_indices,
            removed_indices=removed_indices,
            text_feedback=data.get('feedback'),
        )
        
        print(f"\n[Feature Selection Received]")
        print(f"  Selected: {len(selected_features)} features")
        print(f"  Removed: {len(removed_features)} features")
        if result.text_feedback:
            print(f"  Feedback: {result.text_feedback}")
        
        return result

    def rank_and_filter_features(
        self,
        feature_specs: List,
        task_description: str = None,
        allow_text_feedback: bool = True,
        output_dir: str = None,
    ) -> FeatureRankingResult:
        """Display features for drag-and-drop ranking and optional removal.

        The human reorders the features from most to least important and can
        drop any features they consider unhelpful.

        Args:
            feature_specs: List of FeatureSpec objects or dicts.
            task_description: Optional task context to display.
            allow_text_feedback: Whether to show text feedback box.
            output_dir: Directory to save HTML / result files.

        Returns:
            FeatureRankingResult with ranked and dropped features.
        """
        if not feature_specs:
            raise ValueError("Must provide at least one feature")

        if output_dir is None:
            output_dir = os.getcwd()
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        html = self._generate_ranking_html(
            feature_specs,
            task_description,
            allow_text_feedback,
        )

        html_path = os.path.join(output_dir, "feature_ranking.html")
        with open(html_path, 'w') as f:
            f.write(html)

        selection_file = os.path.join(output_dir, "feature_ranking.json")
        if os.path.exists(selection_file):
            os.remove(selection_file)

        self._original_dir = os.getcwd()
        os.chdir(output_dir)
        handler = self._create_handler(
            selection_file, endpoint="/submit_ranking",
            default_page="feature_ranking.html",
        )
        self.server = http.server.HTTPServer(('0.0.0.0', self.port), handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        self._print_access_instructions()

        start_time = time.time()
        while not os.path.exists(selection_file):
            if time.time() - start_time > self.timeout:
                self._cleanup()
                raise TimeoutError(f"No feedback received within {self.timeout} seconds")
            time.sleep(0.5)

        with open(selection_file) as f:
            data = json.load(f)

        self._cleanup()

        ranked_indices = data.get('ranked', [])
        dropped_indices = data.get('dropped', [])

        ranked_features = [feature_specs[i] for i in ranked_indices]
        dropped_features = [feature_specs[i] for i in dropped_indices]

        result = FeatureRankingResult(
            ranked_features=ranked_features,
            dropped_features=dropped_features,
            ranked_indices=ranked_indices,
            dropped_indices=dropped_indices,
            text_feedback=data.get('feedback'),
        )

        print(f"\n[Feature Ranking Received]")
        print(f"  Ranked: {len(ranked_features)} features")
        print(f"  Dropped: {len(dropped_features)} features")
        for i, feat in enumerate(ranked_features):
            name = self._get_spec_attr(feat, "feature_name", f"Feature {ranked_indices[i]}")
            print(f"    Rank {i + 1}: {name}")
        if result.text_feedback:
            print(f"  Feedback: {result.text_feedback}")

        return result

    def _generate_feature_selection_html(
        self,
        feature_specs: List,
        task_description: str = None,
        allow_text_feedback: bool = True,
        preselected: List[int] = None,
    ) -> str:
        """Generate HTML for feature selection interface."""
        
        feature_cards = ""
        for i, spec in enumerate(feature_specs):
            feature_name = self._get_spec_attr(spec, "feature_name", f"Feature {i+1}")
            intent = self._get_spec_attr(spec, "intent", "")
            weight = self._get_spec_attr(spec, "weight", 1.0)
            desired_direction = self._get_spec_attr(spec, "desired_direction", "")
            proxy_metric = self._get_spec_attr(spec, "proxy_metric", "")
            measurable_signals = self._get_spec_attr(spec, "measurable_signals", [])
            failure_mode = self._get_spec_attr(spec, "typical_failure_mode", "")
            
            signals = ", ".join(measurable_signals) if measurable_signals else "N/A"
            direction_icon = "↓" if desired_direction == "minimize" else "↑"
            direction_color = "#f87171" if desired_direction == "minimize" else "#4ade80"
            
            checked = "checked" if preselected is None or i in preselected else ""
            
            feature_cards += f"""
            <div class="feature-item" id="feature-{i}" data-index="{i}">
                <div class="feature-checkbox">
                    <input type="checkbox" id="check-{i}" {checked} onchange="updateFeature({i})">
                    <label for="check-{i}"></label>
                </div>
                <div class="feature-content">
                    <div class="feature-header">
                        <span class="feature-name">{feature_name}</span>
                        <span class="feature-weight">weight: {weight}</span>
                        <span class="feature-direction" style="color: {direction_color};">
                            {direction_icon} {desired_direction}
                        </span>
                    </div>
                    <div class="feature-intent">
                        <strong>Intent:</strong> {intent}
                    </div>
                    <div class="feature-details">
                        <div class="detail-row">
                            <span class="detail-label">Signals:</span>
                            <code>{signals}</code>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Metric:</span>
                            <code>{proxy_metric}</code>
                        </div>
                    </div>
                    <div class="feature-failure">
                        <strong>Failure mode:</strong> {failure_mode}
                    </div>
                </div>
                <div class="feature-actions">
                    <button class="keep-btn" onclick="keepFeature({i})" title="Keep this feature">✓</button>
                    <button class="remove-btn" onclick="removeFeature({i})" title="Remove this feature">✕</button>
                </div>
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
                <textarea id="feedback" placeholder="What other suggestions do you have? What other features do you think are important?"></textarea>
            </div>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Selection</title>
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
            max-width: 900px;
            border-radius: 0 10px 10px 0;
        }}
        .task-info h2 {{
            color: #4cc9f0;
            margin-top: 0;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
        }}
        .control-btn {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #555;
            color: #ccc;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .control-btn:hover {{
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
        }}
        .stats {{
            text-align: center;
            margin: 20px 0;
            font-size: 16px;
            color: #888;
        }}
        .stats span {{
            margin: 0 15px;
        }}
        .stats .selected-count {{
            color: #4ade80;
        }}
        .stats .removed-count {{
            color: #f87171;
        }}
        .feature-list {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .feature-item {{
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
            gap: 15px;
            transition: all 0.3s ease;
        }}
        .feature-item:hover {{
            border-color: #4cc9f0;
        }}
        .feature-item.selected {{
            border-color: #4ade80;
            background: rgba(74, 222, 128, 0.05);
        }}
        .feature-item.removed {{
            border-color: #f87171;
            background: rgba(248, 113, 113, 0.05);
            opacity: 0.6;
        }}
        .feature-checkbox {{
            flex-shrink: 0;
            padding-top: 5px;
        }}
        .feature-checkbox input[type="checkbox"] {{
            width: 24px;
            height: 24px;
            cursor: pointer;
            accent-color: #4ade80;
        }}
        .feature-content {{
            flex: 1;
        }}
        .feature-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .feature-name {{
            font-weight: bold;
            color: #fff;
            font-size: 16px;
        }}
        .feature-weight {{
            background: rgba(76, 201, 240, 0.2);
            color: #4cc9f0;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .feature-direction {{
            font-weight: bold;
            font-size: 13px;
        }}
        .feature-intent {{
            color: #ccc;
            font-size: 14px;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        .feature-details {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .detail-row {{
            display: flex;
            margin-bottom: 5px;
            font-size: 12px;
        }}
        .detail-row:last-child {{
            margin-bottom: 0;
        }}
        .detail-label {{
            color: #888;
            min-width: 70px;
        }}
        .detail-row code {{
            background: rgba(76, 201, 240, 0.1);
            color: #7dd3fc;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        }}
        .feature-failure {{
            color: #f87171;
            font-size: 12px;
            font-style: italic;
            padding: 8px;
            background: rgba(248, 113, 113, 0.1);
            border-radius: 6px;
            border-left: 3px solid #f87171;
        }}
        .feature-actions {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex-shrink: 0;
        }}
        .keep-btn, .remove-btn {{
            width: 36px;
            height: 36px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.2s;
        }}
        .keep-btn {{
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }}
        .keep-btn:hover {{
            background: rgba(74, 222, 128, 0.4);
        }}
        .remove-btn {{
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }}
        .remove-btn:hover {{
            background: rgba(248, 113, 113, 0.4);
        }}
        .feedback-section {{
            max-width: 900px;
            margin: 30px auto;
        }}
        .feedback-section h3 {{
            color: #4cc9f0;
            margin-bottom: 10px;
        }}
        textarea {{
            width: 100%;
            height: 100px;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #333;
            background: rgba(255, 255, 255, 0.05);
            color: white;
            font-size: 14px;
            resize: vertical;
        }}
        textarea:focus {{
            outline: none;
            border-color: #4cc9f0;
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
    <h1>Feature Selection</h1>
    <p class="subtitle">Select the features you want to keep and remove the ones you don't</p>
    
    {task_section}
    
    <div class="controls">
        <button class="control-btn" onclick="selectAll()">Select All</button>
        <button class="control-btn" onclick="deselectAll()">Deselect All</button>
        <button class="control-btn" onclick="invertSelection()">Invert Selection</button>
    </div>
    
    <div class="stats">
        <span>Total: {len(feature_specs)}</span>
        <span class="selected-count">Selected: <span id="selected-count">{len(preselected) if preselected else len(feature_specs)}</span></span>
        <span class="removed-count">Removed: <span id="removed-count">{len(feature_specs) - len(preselected) if preselected else 0}</span></span>
    </div>
    
    <div class="feature-list" id="feature-list">
        {feature_cards}
    </div>
    
    {feedback_section}
    
    <div class="submit-section" id="submit-section">
        <button class="submit-btn" onclick="submitSelection()">
            Submit Selection
        </button>
    </div>
    
    <div class="success-message" id="success-message">
        <div class="checkmark">✓</div>
        <h2>Thank You!</h2>
        <p>Your feature selection has been recorded.</p>
        <p>You can close this window now.</p>
    </div>
    
    <script>
        const totalFeatures = {len(feature_specs)};
        
        function updateStats() {{
            let selected = 0;
            for (let i = 0; i < totalFeatures; i++) {{
                if (document.getElementById('check-' + i).checked) {{
                    selected++;
                }}
            }}
            document.getElementById('selected-count').textContent = selected;
            document.getElementById('removed-count').textContent = totalFeatures - selected;
        }}
        
        function updateFeature(index) {{
            const checkbox = document.getElementById('check-' + index);
            const item = document.getElementById('feature-' + index);
            
            if (checkbox.checked) {{
                item.classList.add('selected');
                item.classList.remove('removed');
            }} else {{
                item.classList.remove('selected');
                item.classList.add('removed');
            }}
            updateStats();
        }}
        
        function keepFeature(index) {{
            document.getElementById('check-' + index).checked = true;
            updateFeature(index);
        }}
        
        function removeFeature(index) {{
            document.getElementById('check-' + index).checked = false;
            updateFeature(index);
        }}
        
        function selectAll() {{
            for (let i = 0; i < totalFeatures; i++) {{
                document.getElementById('check-' + i).checked = true;
                updateFeature(i);
            }}
        }}
        
        function deselectAll() {{
            for (let i = 0; i < totalFeatures; i++) {{
                document.getElementById('check-' + i).checked = false;
                updateFeature(i);
            }}
        }}
        
        function invertSelection() {{
            for (let i = 0; i < totalFeatures; i++) {{
                const checkbox = document.getElementById('check-' + i);
                checkbox.checked = !checkbox.checked;
                updateFeature(i);
            }}
        }}
        
        function submitSelection() {{
            const selected = [];
            for (let i = 0; i < totalFeatures; i++) {{
                if (document.getElementById('check-' + i).checked) {{
                    selected.push(i);
                }}
            }}
            
            const data = {{
                selected: selected,
                {"feedback: document.getElementById('feedback').value," if allow_text_feedback else ""}
            }};
            
            fetch('/submit_features', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }}).then(response => {{
                if (response.ok) {{
                    document.getElementById('feature-list').style.display = 'none';
                    document.getElementById('submit-section').style.display = 'none';
                    document.querySelector('.controls').style.display = 'none';
                    document.querySelector('.stats').style.display = 'none';
                    const feedbackSection = document.querySelector('.feedback-section');
                    if (feedbackSection) feedbackSection.style.display = 'none';
                    document.getElementById('success-message').style.display = 'block';
                }}
            }});
        }}
        
        // Initialize state
        for (let i = 0; i < totalFeatures; i++) {{
            updateFeature(i);
        }}
    </script>
</body>
</html>
        """
    
    def _generate_ranking_html(
        self,
        feature_specs: List,
        task_description: str = None,
        allow_text_feedback: bool = True,
    ) -> str:
        """Generate HTML for the drag-and-drop feature ranking interface."""

        feature_data_js = []
        for i, spec in enumerate(feature_specs):
            feature_data_js.append({
                "index": i,
                "feature_name": self._get_spec_attr(spec, "feature_name", f"Feature {i+1}"),
                "intent": self._get_spec_attr(spec, "intent", ""),
                "weight": self._get_spec_attr(spec, "weight", 1.0),
                "desired_direction": self._get_spec_attr(spec, "desired_direction", ""),
                "proxy_metric": self._get_spec_attr(spec, "proxy_metric", ""),
                "measurable_signals": self._get_spec_attr(spec, "measurable_signals", []),
                "typical_failure_mode": self._get_spec_attr(spec, "typical_failure_mode", ""),
            })
        feature_json = json.dumps(feature_data_js)

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
                <textarea id="feedback" placeholder="Why did you rank / drop features this way?"></textarea>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Ranking</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            color: #4cc9f0;
            font-size: 2.4em;
            margin-bottom: 8px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 25px;
            font-size: 15px;
        }}
        .task-info {{
            background: rgba(76, 201, 240, 0.1);
            border-left: 4px solid #4cc9f0;
            padding: 15px 20px;
            margin: 20px auto;
            max-width: 960px;
            border-radius: 0 10px 10px 0;
        }}
        .task-info h2 {{
            color: #4cc9f0;
            margin-bottom: 6px;
            font-size: 18px;
        }}
        .task-info p {{
            line-height: 1.5;
        }}

        /* Two-column layout */
        .columns {{
            display: flex;
            gap: 30px;
            max-width: 960px;
            margin: 0 auto;
        }}
        .column {{
            flex: 1;
            min-width: 0;
        }}
        .column-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
        }}
        .column-header h2 {{
            font-size: 20px;
        }}
        .column-header .count {{
            background: rgba(255,255,255,0.1);
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 13px;
            color: #aaa;
        }}
        .keep-col .column-header h2 {{ color: #4ade80; }}
        .drop-col .column-header h2 {{ color: #f87171; }}

        /* Drop zones */
        .drop-zone {{
            min-height: 120px;
            border: 2px dashed #444;
            border-radius: 12px;
            padding: 10px;
            transition: background 0.25s, border-color 0.25s;
        }}
        .drop-zone.drag-over {{
            background: rgba(76, 201, 240, 0.08);
            border-color: #4cc9f0;
        }}
        .drop-zone.empty::after {{
            content: attr(data-empty);
            display: block;
            text-align: center;
            color: #555;
            padding: 30px 0;
            font-style: italic;
        }}

        /* Feature cards */
        .feature-card {{
            background: rgba(255, 255, 255, 0.06);
            border: 2px solid #333;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 10px;
            cursor: grab;
            transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
            position: relative;
        }}
        .feature-card:active {{ cursor: grabbing; }}
        .feature-card.dragging {{
            opacity: 0.45;
            transform: scale(0.97);
        }}
        .feature-card .rank-badge {{
            position: absolute;
            top: -10px;
            left: -10px;
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #4cc9f0, #3b82f6);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            color: #fff;
            box-shadow: 0 2px 8px rgba(76, 201, 240, 0.4);
        }}
        .drop-col .feature-card .rank-badge {{
            display: none;
        }}
        .feature-card .handle {{
            position: absolute;
            top: 14px;
            right: 14px;
            color: #555;
            font-size: 18px;
            cursor: grab;
        }}
        .feature-card:hover {{
            border-color: #4cc9f0;
            box-shadow: 0 0 12px rgba(76, 201, 240, 0.15);
        }}

        .feature-card .feat-name {{
            font-weight: bold;
            color: #fff;
            font-size: 16px;
            margin-bottom: 6px;
            padding-right: 30px;
        }}
        .feature-card .feat-intent {{
            color: #bbb;
            font-size: 13px;
            margin-bottom: 8px;
            line-height: 1.45;
        }}
        .feat-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .meta-tag {{
            font-size: 11px;
            padding: 3px 10px;
            border-radius: 10px;
        }}
        .meta-weight {{
            background: rgba(76, 201, 240, 0.2);
            color: #4cc9f0;
        }}
        .meta-dir-min {{
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }}
        .meta-dir-max {{
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }}
        .feat-details {{
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            padding: 8px 10px;
            margin-bottom: 8px;
        }}
        .detail-row {{
            display: flex;
            font-size: 12px;
            margin-bottom: 4px;
        }}
        .detail-row:last-child {{ margin-bottom: 0; }}
        .detail-label {{
            color: #777;
            min-width: 65px;
            flex-shrink: 0;
        }}
        .detail-row code {{
            background: rgba(76, 201, 240, 0.1);
            color: #7dd3fc;
            padding: 1px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        }}
        .feat-failure {{
            color: #f87171;
            font-size: 12px;
            font-style: italic;
            padding: 7px 10px;
            background: rgba(248, 113, 113, 0.08);
            border-radius: 6px;
            border-left: 3px solid #f87171;
        }}

        /* Feedback */
        .feedback-section {{
            max-width: 960px;
            margin: 25px auto;
        }}
        .feedback-section h3 {{
            color: #4cc9f0;
            margin-bottom: 8px;
        }}
        textarea {{
            width: 100%;
            height: 90px;
            padding: 14px;
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

        /* Submit */
        .submit-section {{
            text-align: center;
            margin: 30px 0;
        }}
        .submit-btn {{
            background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
            border: none;
            padding: 18px 55px;
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
            margin-bottom: 10px;
        }}
        .checkmark {{
            font-size: 4em;
            margin-bottom: 15px;
        }}

        /* Responsive */
        @media (max-width: 700px) {{
            .columns {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <h1>Feature Ranking</h1>
    <p class="subtitle">Drag features to reorder by importance. Move unwanted features to the drop zone on the right.</p>

    {task_section}

    <div class="columns" id="main-area">
        <div class="column keep-col">
            <div class="column-header">
                <h2>Ranked Features</h2>
                <span class="count" id="keep-count">0</span>
            </div>
            <div class="drop-zone" id="keep-zone" data-empty="Drag features here to keep them"></div>
        </div>
        <div class="column drop-col">
            <div class="column-header">
                <h2>Dropped Features</h2>
                <span class="count" id="drop-count">0</span>
            </div>
            <div class="drop-zone" id="drop-zone" data-empty="Drag features here to remove them"></div>
        </div>
    </div>

    {feedback_section}

    <div class="submit-section" id="submit-section">
        <button class="submit-btn" id="submit-btn" onclick="submitRanking()">
            Submit Ranking
        </button>
    </div>

    <div class="success-message" id="success-message">
        <div class="checkmark">&#10003;</div>
        <h2>Thank You!</h2>
        <p>Your feature ranking has been recorded.</p>
        <p>You can close this window now.</p>
    </div>

    <script>
    const features = {feature_json};

    function buildCard(feat) {{
        const dirClass = feat.desired_direction === 'minimize' ? 'meta-dir-min' : 'meta-dir-max';
        const dirIcon  = feat.desired_direction === 'minimize' ? '&#8595;' : '&#8593;';
        const signals  = (feat.measurable_signals || []).join(', ') || 'N/A';

        const card = document.createElement('div');
        card.className = 'feature-card';
        card.draggable = true;
        card.dataset.index = feat.index;
        card.innerHTML = `
            <div class="rank-badge"></div>
            <span class="handle">&#9776;</span>
            <div class="feat-name">${{feat.feature_name}}</div>
            <div class="feat-intent">${{feat.intent}}</div>
            <div class="feat-meta">
                <span class="meta-tag meta-weight">weight: ${{feat.weight}}</span>
                <span class="meta-tag ${{dirClass}}">${{dirIcon}} ${{feat.desired_direction}}</span>
            </div>
            <div class="feat-details">
                <div class="detail-row"><span class="detail-label">Signals:</span><code>${{signals}}</code></div>
                <div class="detail-row"><span class="detail-label">Metric:</span><code>${{feat.proxy_metric}}</code></div>
            </div>
            ${{feat.typical_failure_mode ? `<div class="feat-failure"><strong>Failure mode:</strong> ${{feat.typical_failure_mode}}</div>` : ''}}
        `;
        return card;
    }}

    const keepZone = document.getElementById('keep-zone');
    const dropZone = document.getElementById('drop-zone');

    features.forEach(f => keepZone.appendChild(buildCard(f)));
    refreshBadges();

    /* ---------- Drag & Drop ---------- */
    let dragEl = null;

    document.addEventListener('dragstart', e => {{
        const card = e.target.closest('.feature-card');
        if (!card) return;
        dragEl = card;
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    }});

    document.addEventListener('dragend', e => {{
        if (dragEl) dragEl.classList.remove('dragging');
        dragEl = null;
        document.querySelectorAll('.drop-zone').forEach(z => z.classList.remove('drag-over'));
        refreshBadges();
    }});

    function getDropTarget(zone, y) {{
        const cards = [...zone.querySelectorAll('.feature-card:not(.dragging)')];
        for (const card of cards) {{
            const box = card.getBoundingClientRect();
            if (y < box.top + box.height / 2) return card;
        }}
        return null;
    }}

    [keepZone, dropZone].forEach(zone => {{
        zone.addEventListener('dragover', e => {{
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            zone.classList.add('drag-over');

            if (!dragEl) return;
            const target = getDropTarget(zone, e.clientY);
            if (target) {{
                zone.insertBefore(dragEl, target);
            }} else {{
                zone.appendChild(dragEl);
            }}
        }});

        zone.addEventListener('dragleave', e => {{
            if (!zone.contains(e.relatedTarget)) {{
                zone.classList.remove('drag-over');
            }}
        }});

        zone.addEventListener('drop', e => {{
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (!dragEl) return;
            const target = getDropTarget(zone, e.clientY);
            if (target) {{
                zone.insertBefore(dragEl, target);
            }} else {{
                zone.appendChild(dragEl);
            }}
            refreshBadges();
        }});
    }});

    /* ---------- Touch support (mobile / trackpad) ---------- */
    let touchCard = null;
    let touchClone = null;
    let touchOffsetX = 0;
    let touchOffsetY = 0;

    document.addEventListener('touchstart', e => {{
        const card = e.target.closest('.feature-card');
        if (!card) return;
        touchCard = card;
        const rect = card.getBoundingClientRect();
        touchOffsetX = e.touches[0].clientX - rect.left;
        touchOffsetY = e.touches[0].clientY - rect.top;

        touchClone = card.cloneNode(true);
        touchClone.style.position = 'fixed';
        touchClone.style.width = rect.width + 'px';
        touchClone.style.zIndex = 10000;
        touchClone.style.opacity = '0.85';
        touchClone.style.pointerEvents = 'none';
        document.body.appendChild(touchClone);

        card.classList.add('dragging');
    }}, {{ passive: true }});

    document.addEventListener('touchmove', e => {{
        if (!touchCard) return;
        e.preventDefault();
        const x = e.touches[0].clientX - touchOffsetX;
        const y = e.touches[0].clientY - touchOffsetY;
        touchClone.style.left = x + 'px';
        touchClone.style.top  = y + 'px';

        const cx = e.touches[0].clientX;
        const cy = e.touches[0].clientY;

        [keepZone, dropZone].forEach(zone => {{
            const r = zone.getBoundingClientRect();
            if (cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom) {{
                zone.classList.add('drag-over');
                const target = getDropTarget(zone, cy);
                if (target) zone.insertBefore(touchCard, target);
                else zone.appendChild(touchCard);
            }} else {{
                zone.classList.remove('drag-over');
            }}
        }});
    }}, {{ passive: false }});

    document.addEventListener('touchend', () => {{
        if (touchClone) {{ touchClone.remove(); touchClone = null; }}
        if (touchCard) {{ touchCard.classList.remove('dragging'); touchCard = null; }}
        document.querySelectorAll('.drop-zone').forEach(z => z.classList.remove('drag-over'));
        refreshBadges();
    }});

    /* ---------- Helpers ---------- */
    function refreshBadges() {{
        const keepCards = keepZone.querySelectorAll('.feature-card');
        const dropCards = dropZone.querySelectorAll('.feature-card');

        keepCards.forEach((c, i) => {{
            c.querySelector('.rank-badge').textContent = i + 1;
        }});

        document.getElementById('keep-count').textContent = keepCards.length;
        document.getElementById('drop-count').textContent = dropCards.length;

        keepZone.classList.toggle('empty', keepCards.length === 0);
        dropZone.classList.toggle('empty', dropCards.length === 0);
    }}

    /* ---------- Submit ---------- */
    function submitRanking() {{
        const ranked  = [...keepZone.querySelectorAll('.feature-card')].map(c => parseInt(c.dataset.index));
        const dropped = [...dropZone.querySelectorAll('.feature-card')].map(c => parseInt(c.dataset.index));

        const data = {{
            ranked: ranked,
            dropped: dropped,
            {"feedback: document.getElementById('feedback').value," if allow_text_feedback else ""}
        }};

        fetch('/submit_ranking', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(data)
        }}).then(response => {{
            if (response.ok) {{
                document.getElementById('main-area').style.display = 'none';
                document.getElementById('submit-section').style.display = 'none';
                const fb = document.querySelector('.feedback-section');
                if (fb) fb.style.display = 'none';
                document.getElementById('success-message').style.display = 'block';
            }}
        }});
    }}
    </script>
</body>
</html>
        """

    def _create_handler(
        self,
        selection_file: str,
        endpoint: str = "/submit",
        default_page: str = None,
    ):
        """Create an HTTP request handler that serves files and handles POST submissions."""
        _default_page = default_page

        class FeedbackHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if _default_page and self.path in ('/', '/index.html'):
                    self.path = '/' + _default_page
                return super().do_GET()

            def do_POST(self):
                if self.path == endpoint:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    with open(selection_file, 'w') as f:
                        f.write(post_data.decode('utf-8'))
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return FeedbackHandler

    def _cleanup(self):
        """Clean up server and restore directory."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self._original_dir:
            os.chdir(self._original_dir)
            self._original_dir = None

    def _get_spec_attr(self, spec, attr: str, default=""):
        """Get attribute from FeatureSpec object or dict."""
        if isinstance(spec, dict):
            return spec.get(attr, default)
        return getattr(spec, attr, default)

    def _generate_feature_specs_html(self, feature_specs: List, index: int) -> str:
        """Generate HTML for detailed feature specifications."""
        if not feature_specs:
            return ""
        
        cards = ""
        for i, spec in enumerate(feature_specs):
            desired_direction = self._get_spec_attr(spec, "desired_direction", "")
            direction_icon = "↓" if desired_direction == "minimize" else "↑"
            direction_color = "#f87171" if desired_direction == "minimize" else "#4ade80"
            
            measurable_signals = self._get_spec_attr(spec, "measurable_signals", [])
            signals = ", ".join(measurable_signals) if measurable_signals else "N/A"
            
            feature_name = self._get_spec_attr(spec, "feature_name", "Unknown")
            weight = self._get_spec_attr(spec, "weight", 1.0)
            intent = self._get_spec_attr(spec, "intent", "")
            proxy_metric = self._get_spec_attr(spec, "proxy_metric", "")
            failure_mode = self._get_spec_attr(spec, "typical_failure_mode", "")
            
            cards += f"""
            <div class="feature-card">
                <div class="feature-header">
                    <span class="feature-name">{feature_name}</span>
                    <span class="feature-weight">weight: {weight}</span>
                    <span class="feature-direction" style="color: {direction_color};">
                        {direction_icon} {desired_direction}
                    </span>
                </div>
                <div class="feature-intent">
                    <strong>Intent:</strong> {intent}
                </div>
                <div class="feature-details">
                    <div class="detail-row">
                        <span class="detail-label">Signals:</span>
                        <code>{signals}</code>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Metric:</span>
                        <code>{proxy_metric}</code>
                    </div>
                </div>
                <div class="feature-failure">
                    <strong>Failure mode:</strong> {failure_mode}
                </div>
            </div>
            """
        
        return f"""
        <div class="feature-specs-section">
            <h4 onclick="toggleSection('features-{index}')" class="section-toggle">
                Feature Specifications <span class="toggle-icon" id="icon-features-{index}">▼</span>
            </h4>
            <div class="section-content" id="features-{index}">
                {cards}
            </div>
        </div>
        """

    def _generate_reward_section(self, reward_info: RewardInfo, index: int) -> str:
        """Generate HTML for reward info section."""
        if reward_info is None:
            return ""
        
        sections = []
        
        # Description
        if reward_info.description:
            sections.append(f'<p class="reward-desc">{reward_info.description}</p>')
        
        # Detailed feature specs
        if reward_info.feature_specs:
            sections.append(self._generate_feature_specs_html(reward_info.feature_specs, index))
        
        # Simple features table (if no detailed specs)
        if reward_info.features and not reward_info.feature_specs:
            features_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in reward_info.features.items()
            )
            sections.append(f"""
            <div class="simple-table-section">
                <h5>Features</h5>
                <table>
                    <thead><tr><th>Feature</th><th>Value</th></tr></thead>
                    <tbody>{features_rows}</tbody>
                </table>
            </div>
            """)
        
        # Metrics table
        # if reward_info.metrics:
        #     def format_metric(v):
        #         return f"{v:.4f}" if isinstance(v, float) else str(v)
        #     metrics_rows = "".join(
        #         f"<tr><td>{k}</td><td>{format_metric(v)}</td></tr>"
        #         for k, v in reward_info.metrics.items()
        #     )
        #     sections.append(f"""
        #     <div class="simple-table-section">
        #         <h5>Metrics</h5>
        #         <table>
        #             <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        #             <tbody>{metrics_rows}</tbody>
        #         </table>
        #     </div>
        #     """)
        
        if not sections:
            return ""
        
        title = reward_info.name if reward_info.name else "Reward Information"
        return f"""
        <div class="reward-info-container">
            <h4 class="reward-title">{title}</h4>
            {"".join(sections)}
        </div>
        """

    def _generate_html(
        self,
        video_paths: List[str],
        descriptions: List[str],
        task_description: str = None,
        reward_infos: List[RewardInfo] = None,
        allow_text_feedback: bool = True,
        allow_rating: bool = False,
    ) -> str:
        """Generate the HTML page for video/reward selection."""
        video_cards = ""
        num_options = len(descriptions)
        
        for i in range(num_options):
            desc = descriptions[i]
            
            # Video section (only if valid video provided, not "NONE")
            video_section = ""
            if video_paths and i < len(video_paths):
                video_path = video_paths[i]
                if video_path and video_path.upper() != "NONE":
                    video_name = os.path.basename(video_path)
                    video_section = f"""
                    <video controls preload="metadata">
                        <source src="{video_name}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    """
                else:
                    video_section = """
                    <div class="no-video">
                        <span class="no-video-icon">🎬</span>
                        <p>No video available</p>
                    </div>
                    """
            
            # Reward info section
            reward_section = ""
            if reward_infos and i < len(reward_infos) and reward_infos[i]:
                reward_section = self._generate_reward_section(reward_infos[i], i)
            
            video_cards += f"""
            <div class="video-card" id="card-{i}">
                <h3>{desc}</h3>
                {video_section}
                {reward_section}
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
                <textarea id="feedback" placeholder="What did you like or dislike? Any suggestions for improvement?"></textarea>
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
        
        # Determine title and subtitle based on content
        valid_videos = [p for p in video_paths if p and p.upper() != "NONE"]
        has_any_videos = bool(valid_videos)
        has_all_videos = has_any_videos and len(valid_videos) == num_options
        
        if has_all_videos:
            page_title = "Human Feedback - Video Selection"
            page_subtitle = "Watch the videos and select your preferred option"
        elif has_any_videos:
            page_title = "Human Feedback - Comparison"
            page_subtitle = "Review the options and select your preferred one"
        else:
            page_title = "Human Feedback - Reward Comparison"
            page_subtitle = "Review the reward functions and select your preferred option"
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{page_title}</title>
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
            max-width: 900px;
            border-radius: 0 10px 10px 0;
        }}
        .task-info h2 {{
            color: #4cc9f0;
            margin-top: 0;
        }}
        .container {{ 
            display: flex; 
            flex-wrap: nowrap;
            justify-content: center; 
            align-items: flex-start;
            gap: 30px; 
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .video-card {{ 
            background: rgba(255,255,255,0.05); 
            padding: 20px; 
            border-radius: 15px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            flex: 1;
            min-width: 400px;
            max-width: 700px;
        }}
        @media (max-width: 900px) {{
            .container {{
                flex-wrap: wrap;
            }}
            .video-card {{
                min-width: 100%;
            }}
        }}
        .video-card:hover {{
            border-color: #4cc9f0;
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
            max-width: 550px;
            border-radius: 10px; 
            margin: 10px 0;
            background: #000;
        }}
        .no-video {{
            width: 100%;
            max-width: 550px;
            height: 300px;
            border-radius: 10px;
            margin: 10px auto;
            background: rgba(0, 0, 0, 0.3);
            border: 2px dashed #555;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #888;
        }}
        .no-video-icon {{
            font-size: 48px;
            margin-bottom: 10px;
            opacity: 0.5;
        }}
        .no-video p {{
            margin: 0;
            font-size: 14px;
        }}
        
        /* Reward Info Styles */
        .reward-info-container {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin: 15px 0;
            padding: 15px;
            text-align: left;
        }}
        .reward-title {{
            color: #4cc9f0;
            margin: 0 0 10px 0;
            font-size: 16px;
        }}
        .reward-desc {{
            color: #aaa;
            font-size: 13px;
            margin-bottom: 15px;
            line-height: 1.5;
        }}
        
        /* Section Toggle Styles */
        .section-toggle {{
            background: rgba(76, 201, 240, 0.15);
            padding: 10px 15px;
            margin: 10px 0 0 0;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            color: #4cc9f0;
            border-radius: 8px;
            transition: background 0.3s;
        }}
        .section-toggle:hover {{
            background: rgba(76, 201, 240, 0.25);
        }}
        .toggle-icon {{
            font-size: 12px;
            transition: transform 0.3s;
        }}
        .section-content {{
            padding: 15px 0;
            max-height: 600px;
            overflow-y: auto;
        }}
        .section-content.collapsed {{
            display: none;
        }}
        
        /* Feature Card Styles */
        .feature-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 12px;
        }}
        .feature-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .feature-name {{
            font-weight: bold;
            color: #fff;
            font-size: 15px;
        }}
        .feature-weight {{
            background: rgba(76, 201, 240, 0.2);
            color: #4cc9f0;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
        }}
        .feature-direction {{
            font-weight: bold;
            font-size: 13px;
        }}
        .feature-intent {{
            color: #ccc;
            font-size: 13px;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        .feature-details {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .detail-row {{
            display: flex;
            margin-bottom: 5px;
            font-size: 12px;
        }}
        .detail-row:last-child {{
            margin-bottom: 0;
        }}
        .detail-label {{
            color: #888;
            min-width: 70px;
        }}
        .detail-row code {{
            background: rgba(76, 201, 240, 0.1);
            color: #7dd3fc;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
        }}
        .feature-failure {{
            color: #f87171;
            font-size: 12px;
            font-style: italic;
            padding: 8px;
            background: rgba(248, 113, 113, 0.1);
            border-radius: 6px;
            border-left: 3px solid #f87171;
        }}
        
        /* Simple Table Styles */
        .simple-table-section {{
            margin: 15px 0;
        }}
        .simple-table-section h5 {{
            color: #4cc9f0;
            margin: 0 0 8px 0;
            font-size: 13px;
        }}
        .simple-table-section table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        .simple-table-section th, .simple-table-section td {{
            padding: 8px 12px;
            border: 1px solid #333;
            text-align: left;
        }}
        .simple-table-section th {{
            background: rgba(76, 201, 240, 0.1);
            color: #4cc9f0;
        }}
        .simple-table-section td {{
            color: #ccc;
        }}
        
        /* Code Block Styles */
        .code-block {{
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            line-height: 1.5;
            color: #a8d8a8;
            max-height: 400px;
            overflow-y: auto;
            margin: 0;
        }}
        .code-block code {{
            white-space: pre;
        }}
        
        /* Button Styles */
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
            margin-top: 15px;
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
    <p class="subtitle">{page_subtitle}</p>
    
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
            
            document.querySelectorAll('.video-card').forEach((card, i) => {{
                card.classList.toggle('selected', i === index);
            }});
            
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('selection-hint').textContent = 
                'Option ' + (index + 1) + ' selected';
            document.getElementById('selection-hint').style.color = '#4ade80';
        }}
        
        function toggleSection(id) {{
            const content = document.getElementById(id);
            const icon = document.getElementById('icon-' + id);
            if (content) {{
                content.classList.toggle('collapsed');
                if (icon) {{
                    icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
                }}
            }}
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


# Legacy alias for backwards compatibility
HumanVideoSelector = HumanFeedbackManager
