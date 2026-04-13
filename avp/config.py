"""
Lightweight configuration loader for Active Video Perception (AVP / AAVP).

Supports a single JSON file.  All fields are optional — unrecognised keys are
ignored and every field has a safe default so existing configs continue to work
without modification.

Core fields (unchanged from AVP):
{
  "project":                  "your-gcp-project",
  "location":                 ["us-central1", "us-east1", "global"],
  "model":                    "gemini-2.5-pro",
  "plan_replan_model":        "",
  "execute_model":            "",
  "annotation_path":          "/path/to/eval.json",
  "output_dir":               "/path/to/out",
  "default_media_resolution": "medium",
  "prefer_compressed":        true,
  "debug":                    false
}

AAVP audio enrichment fields (all default to off/disabled):
{
  "audio_enabled":                false,
  "audio_sample_rate":            16000,
  "audio_snippet_halfwidth_sec":  2.5,
  "audio_max_snippets_per_round": 15,
  "audio_gap_probes":             5,
  "audio_closed_tags":            ["SILENCE", "SPEECH", ...]
}

Notes:
- location can be a single string or a list; config.get_random_location() picks
  one randomly per sample for load-balanced Vertex AI requests.
- audio_enabled is the master switch: when False the Observer never calls
  audio_utils regardless of what the planner selects.
- audio_closed_tags must stay in sync with the enum list in prompt.py
  (_ACOUSTIC_TAGS) and AUDIO_ENRICHMENT_SCHEMA.

Env overrides (applied after JSON, if set):
  VERTEX_PROJECT, VERTEX_LOCATION, GEMINI_MODEL, GEMINI_API_KEY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os
import random
from pathlib import Path


@dataclass
class AVPConfig:
    project: str = "your-gcp-project"
    location: List[str] = field(default_factory=lambda: ["global"])  # List of locations to randomly select from
    model: str = "gemini-2.5-pro"  # Legacy field - used as fallback if plan_replan_model/execute_model not set
    plan_replan_model: str = ""  # Model for planning and replanning (empty = use model field)
    execute_model: str = ""  # Model for video inference/execution (empty = use model field)
    api_key: str = ""  # If set, use Google AI API key (e.g. from Google AI Studio); else use Vertex AI. Prefer GEMINI_API_KEY env.
    annotation_path: str = ""
    output_dir: str = ""
    default_media_resolution: str = "medium"  # low|medium|high
    prefer_compressed: bool = True
    debug: bool = False
    
    # Max frame settings for media resolution
    max_frame_low: int = 512
    max_frame_medium: int = 128
    max_frame_high: int = 128

    # ------------------------------------------------------------------
    # AAVP audio enrichment settings
    # All default to off / conservative values so existing AVP configs
    # are completely unaffected.
    # ------------------------------------------------------------------

    # Master switch.  When False the Observer skips audio enrichment entirely,
    # even if the planner sets audio_enrichment != "off".
    audio_enabled: bool = False

    # PCM sample rate for ffmpeg WAV extraction.  16 000 Hz is the Gemini
    # minimum and produces ~64 KB per 5 s snippet.
    audio_sample_rate: int = 16000

    # Half-width of each audio window in seconds (total window = 2 × this).
    # 2.5 s → 5 s window, centred on the key_evidence midpoint.
    audio_snippet_halfwidth_sec: float = 2.5

    # Maximum WAV snippets sent in a single enrichment API call per round.
    # Evidence-source snippets are always kept; gap probes are trimmed first
    # when the cap is exceeded.
    audio_max_snippets_per_round: int = 15

    # Maximum number of gap probes added when audio_enrichment="evidence_plus_gaps".
    audio_gap_probes: int = 5

    # Closed-vocabulary acoustic event tags.  Must stay in sync with
    # prompt._ACOUSTIC_TAGS and AUDIO_ENRICHMENT_SCHEMA.
    audio_closed_tags: List[str] = field(default_factory=lambda: [
        "SILENCE", "SPEECH", "MUSIC", "CHEER", "APPLAUSE",
        "WHISTLE", "BUZZER", "CRASH", "DOOR", "FOOTSTEPS",
        "ENGINE", "SIREN", "BELL", "TYPING", "LAUGHTER",
        "ANIMAL", "WATER", "WIND", "AMBIENT", "OTHER",
    ])

    def __post_init__(self):
        """Initialize location as list if it's a string."""
        if isinstance(self.location, str):
            self.location = [self.location]
        elif not isinstance(self.location, list):
            raise ValueError(f"location must be a string or list of strings, got {type(self.location)}")
    
    def get_random_location(self) -> str:
        """Randomly select a location from the location list."""
        if not self.location:
            return "global"
        return random.choice(self.location)

    def get_plan_replan_model(self) -> str:
        """Get the model for planning/replanning operations."""
        if self.plan_replan_model:
            return self.plan_replan_model
        return self.model
    
    def get_execute_model(self) -> str:
        """Get the model for execution/inference operations."""
        if self.execute_model:
            return self.execute_model
        return self.model
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AVPConfig":
        cfg = AVPConfig()
        for k in d:
            if hasattr(cfg, k):
                setattr(cfg, k, d[k])
        # Env overrides
        cfg.project = os.getenv("VERTEX_PROJECT", cfg.project)
        env_location = os.getenv("VERTEX_LOCATION")
        if env_location:
            # If env var is set, convert to list (handle comma-separated values)
            cfg.location = [loc.strip() for loc in env_location.split(",") if loc.strip()]
        # Legacy model env var applies to both if not separately specified
        env_model = os.getenv("GEMINI_MODEL")
        if env_model:
            if not cfg.plan_replan_model:
                cfg.plan_replan_model = env_model
            if not cfg.execute_model:
                cfg.execute_model = env_model
            cfg.model = env_model  # Also set legacy field
        # API key: env overrides config so production can use env-only
        cfg.api_key = os.getenv("GEMINI_API_KEY", cfg.api_key or "")
        # Ensure location is properly initialized as a list
        cfg.__post_init__()
        return cfg


def load_config(path: Optional[str]) -> AVPConfig:
    """Load config from JSON file if provided; else env/defaults.

    Args:
        path: Optional path to a JSON config file
    """
    if path is None or str(path).strip() == "":
        return AVPConfig.from_dict({})

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object")
    return AVPConfig.from_dict(data)


