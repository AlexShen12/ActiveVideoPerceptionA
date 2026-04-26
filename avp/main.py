"""
Agentic Video Understanding Framework (Gemini API) — Plan-Observe-Reflect Implementation
========================================================================================

This module implements a single-action plan → observe → verification loop for video understanding.
Given a query Q and a long video V, the framework runs an iterative process:

1. Plan (single observation action):
   - The Planner drafts one observation configuration for this round (no multi-step):
     - load_mode: "uniform" (full video) or "region" (specific time span[s])
     - fps: sampling rate
     - spatial resolution: low/medium
   - The sub_query equals the original query (including options if provided).

2. Observe:
   - The Observer watches the video using the planned configuration and the query,
     feeds the video directly to the model, and collects a query-related evidence list.
   - Evidence is a list of timestamp ranges with descriptions, normalized to full seconds.

3. Verification (decision):
   - The verifier re-watches the specific evidence regions (cropped clips or offsets)
     to confirm correctness and sufficiency to answer the query.
   - If evidence is sufficient: synthesize the final answer.
   - If insufficient: return a justification and loop back to the Planner with the
     current evidence list and full interaction history to draft a new observation.

The framework leverages Gemini's native video understanding capabilities:
videos are passed directly to the API with metadata (fps, time ranges, resolution)
rather than extracting individual frames.

Video metadata (duration, fps) is loaded from JSON dataset files rather than
extracting from video files, eliminating codec dependencies and improving speed.

External deps: `google-genai` (Gemini SDK) and `tqdm` (optional).
No OpenCV or video codec dependencies required!

CLI subcommands:
  - `plan`          : Create initial plan only
  - `run`           : Orchestrate plan→observe→verify/reflect loop until done
  - `show`          : Pretty-print current plan/evidence summary

Usage examples:
  python agentic_video_framework.py plan --run-dir runs/demo --video /path/v.mp4 --query "When does the person enter the red car?"
  python agentic_video_framework.py run --run-dir runs/demo --max-rounds 3
  python agentic_video_framework.py show --run-dir runs/demo
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import pathlib
import re
import time
import hashlib

# Gemini / Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Part, Blob, VideoMetadata
import os
from pathlib import Path

# Import prompt management
from .prompt import (
    PromptManager,
    parse_json_response,
    validate_against_schema,
    PLAN_SCHEMA,
    EVIDENCE_SCHEMA,
    FINAL_ANSWER_SCHEMA,
    MCQ_SCHEMA,
    AUDIO_ENRICHMENT_SCHEMA,
    REFLECTION_SCHEMA,
)

# AAVP audio extraction utilities (ffmpeg-based, no local ASR)
from .audio_utils import (
    extract_audio_snippet,
    generate_gap_probes,
    cleanup_audio_artifacts,
)

# Import video utilities
from .video_utils import (
    VideoMetadataExtractor,
    get_mime_type,
    find_compressed_video_fallback,
    create_video_clip,  # For creating video clips in region mode
    round_intervals_full_seconds,
)

# -----------------------------
# Optional deps
# -----------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x


# Max frame settings moved to config - use AVPConfig.max_frame_* instead


# ======================================================
# API retry helper
# ======================================================

def _generate_with_backoff(models_client, *, model: str, contents, config=None, max_retries: int = 6):
    """Call ``models_client.generate_content`` with exponential back-off.

    Retries on HTTP 429 (quota / rate-limit) and transient 5xx errors.
    Raises the last exception if all retries are exhausted.

    Delay schedule (seconds): 5, 10, 20, 40, 80, 160  (doubles each time, cap 160 s).
    """
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            if config is not None:
                return models_client.generate_content(
                    model=model, contents=contents, config=config
                )
            return models_client.generate_content(model=model, contents=contents)
        except Exception as exc:
            msg = str(exc)
            # Retry on quota / rate-limit errors and transient server errors.
            is_retryable = (
                "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "quota" in msg.lower()
                or "rate" in msg.lower()
                or "503" in msg
                or "500" in msg
            )
            if not is_retryable or attempt == max_retries:
                raise
            wait = min(delay, 160.0)
            print(
                f"[retry] generate_content hit retryable error (attempt {attempt + 1}/{max_retries}): "
                f"{msg[:120]} — waiting {wait:.0f}s"
            )
            time.sleep(wait)
            delay *= 2


# ======================================================
# Contracts & Schemas
# ======================================================
class SpatialTokenRate(str, Enum):
    low = "low"
    medium = "medium"


class AudioEnrichmentScope(str, Enum):
    """Controls whether and how audio enrichment runs after the visual observation.

    off               — No audio; original AVP behaviour (default).
    evidence_only     — Extract audio around each key_evidence timestamp returned
                        by the visual pass and send to Gemini for interpretation.
    evidence_plus_gaps — Same as evidence_only, plus sparse probes at the
                        midpoints of large uncovered timeline gaps, catching
                        off-screen narration or sounds missed by the visual pass.
    """
    off = "off"
    evidence_only = "evidence_only"
    evidence_plus_gaps = "evidence_plus_gaps"


class AudioMode(str, Enum):
    """Steers the audio interpretation prompt toward speech or sound events.

    balanced       — Both verbatim ASR transcription and acoustic event tagging.
    asr_focus      — Prioritise verbatim speech transcription (reasoning queries).
    acoustic_focus — Prioritise non-speech event detection (event-tracking queries).
    """
    balanced = "balanced"
    asr_focus = "asr_focus"
    acoustic_focus = "acoustic_focus"


@dataclass
class WatchConfig:
    """Configuration for one observation round.

    Visual controls (unchanged from AVP):
    - load_mode:          "uniform" (full video) | "region" (specific spans)
    - fps:                Temporal sampling rate in frames per second
    - spatial_token_rate: Spatial resolution ("low" | "medium")
    - regions:            [(start_sec, end_sec)] used only when load_mode="region"

    AAVP audio enrichment controls (post-observation, default-off):
    - audio_enrichment:         Scope of post-visual audio extraction
    - audio_mode:               Focus of the audio interpretation prompt
    - audio_snippet_halfwidth_sec: Half-width of the WAV window per timestamp (s)
    """
    load_mode: str                        # "uniform" | "region"
    fps: float
    spatial_token_rate: SpatialTokenRate
    regions: List[Tuple[float, float]] = field(default_factory=list)
    # --- AAVP additions ---
    audio_enrichment: AudioEnrichmentScope = AudioEnrichmentScope.off
    audio_mode: AudioMode = AudioMode.balanced
    audio_snippet_halfwidth_sec: float = 2.5  # 5 s total window per evidence timestamp


@dataclass
class PlanSpec:
    """Single observation action plan.
    
    Specifies one observation with:
    - query: The user's question
    - watch: Observation config (load_mode, fps, spatial_token_rate, regions)
    - description: Goal/reasoning objective for this observation
    """
    plan_version: str
    query: str
    watch: WatchConfig  # Observation config: region, fps, spatial resolution
    description: str = ""  # Goal/reasoning objective
    completion_criteria: str = ""
    final_answer: Optional[str] = None
    complete: bool = False


@dataclass
class AudioEnrichment:
    """Audio evidence gathered for a single timestamp from the visual pass.

    One ``AudioEnrichment`` corresponds to one WAV snippet extracted by
    ``audio_utils.extract_audio_snippet`` and interpreted by Gemini.

    Fields:
        center_sec:       Midpoint of the audio window in original video seconds.
        window_start_sec: Actual extraction start (= center_sec - halfwidth, clamped).
        window_end_sec:   Actual extraction end  (= center_sec + halfwidth).
        speech_evidence:  Verbatim word-bounded transcript; empty string if inaudible.
        acoustic_evidence: Closed-tag acoustic labels drawn from the config vocabulary.
        source:           "evidence" (from key_evidence) or "gap_probe".
    """
    center_sec: float
    window_start_sec: float
    window_end_sec: float
    speech_evidence: str = ""
    acoustic_evidence: List[str] = field(default_factory=list)
    source: str = "evidence"  # "evidence" | "gap_probe"


@dataclass
class Evidence:
    """Evidence gathered from one observation round.

    Visual fields (unchanged from AVP):
    - detailed_response: Full analysis text from the visual model call
    - key_evidence:      Timestamp intervals [timestamp_start, timestamp_end, description]
    - reasoning:         Model's explanation of findings
    - frames_used:       Video segments submitted to the model
    - model_call:        API call metadata
    - timestamp:         ISO timestamp of collection
    - round_id:          Which iteration round produced this evidence

    AAVP addition:
    - audio_enrichments: Per-timestamp audio results from the post-visual enrichment
                         step (empty list when audio_enrichment=off).
    """
    detailed_response: str = ""
    key_evidence: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    frames_used: List[Dict[str, Any]] = field(default_factory=list)
    model_call: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    round_id: int = 0
    # --- AAVP addition ---
    audio_enrichments: List[AudioEnrichment] = field(default_factory=list)

    @property
    def step_id(self) -> str:
        """Compatibility property: returns round_id as string."""
        return str(self.round_id)


@dataclass
class ReflectionOutput:
    """Machine-readable output from the LLM-based AAVP reflector.

    Used by ``Reflector._reflect_aavp`` when audio enrichments are present.
    The Controller reads ``zoom_region`` and ``required_modalities`` to override
    the next plan when the reflector finds insufficient multimodal alignment.

    Fields:
        sufficient:              True if evidence is enough to answer the query.
        query_support:           Per-modality confidence scores {visual, speech, acoustic}.
        citations:               Grounded evidence citations with timestamps and quotes.
        contradiction_with_query: True if evidence directly contradicts the query.
        reason_code:             One of SUFFICIENT | MODALITY_MISMATCH | TEMPORAL_GAP |
                                 LOW_CONFIDENCE | NO_EVIDENCE.
        zoom_region:             [t_start, t_end] for targeted re-observation (None if
                                 not applicable).
        required_modalities:     Escalation hints, e.g.
                                 {"audio_enrichment": "evidence_plus_gaps",
                                  "spatial_token_rate": "medium"}.
        reasoning:               Human-readable explanation of the reflection decision.
        confidence:              Reflector's confidence in its own decision (0–1).
        query_confidence:        Estimated probability the query can be answered (0–1).
    """
    sufficient: bool
    query_support: Dict[str, float] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    contradiction_with_query: bool = False
    reason_code: str = ""
    zoom_region: Optional[Tuple[float, float]] = None
    required_modalities: Dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    query_confidence: float = 0.0


@dataclass
class Blackboard:
    video_path: str
    duration_sec: Optional[float] = None
    evidences: List[Evidence] = field(default_factory=list)  # List of evidence from each round
    meta: Dict[str, Any] = field(default_factory=dict)
    query_confidence: Optional[float] = None  # Query-level confidence (0.0 to 1.0) for answering the query

    def add_evidence(self, ev: Evidence) -> None:
        """Add evidence from current round."""
        self.evidences.append(ev)

    def get_evidence_list(self) -> List[Evidence]:
        """Get evidence as a list, sorted temporally by earliest timestamp.
        
        Evidence is ordered by the earliest timestamp_start in each evidence's key_evidence.
        If evidence has no timestamps, it's placed at the end (sorted by round_id as tiebreaker).
        
        Returns:
            List of Evidence objects sorted temporally (by timestamp_start)
        """
        def get_earliest_timestamp(ev: Evidence) -> float:
            """Get the earliest timestamp_start from evidence's key_evidence."""
            earliest = None
            for kev in ev.key_evidence:
                if isinstance(kev, dict):
                    ts_start = kev.get("timestamp_start")
                    if ts_start is not None:
                        if earliest is None or ts_start < earliest:
                            earliest = ts_start
            # If no timestamps found, return a large number to place at end
            if earliest is None:
                return 999999.0 + ev.round_id  # Place at end, sorted by round_id
            return earliest
        
        evidence_list = list(self.evidences)
        # Sort by earliest timestamp, then by round_id as tiebreaker
        evidence_list.sort(key=lambda ev: (get_earliest_timestamp(ev), ev.round_id))
        return evidence_list

    def audio_enrichment_summary_text(self) -> str:
        """Format all audio enrichment evidence for use in prompts.

        Produces a compact timestamped block suitable for the reflector and
        synthesis prompts.  Example output line::

            t=45.0s [evidence] | Speech: "Welcome back everyone" | Acoustic: [SPEECH, MUSIC]
            t=75.0s [gap_probe] | Speech: "" | Acoustic: [AMBIENT]

        Returns empty string when no audio enrichments exist (audio_enrichment=off
        or audio step produced no results), so callers can test truthiness.
        """
        lines = []
        for ev in self.evidences:
            for ae in ev.audio_enrichments:
                parts = [f"t={ae.center_sec:.1f}s [{ae.source}]"]
                if ae.speech_evidence:
                    parts.append(f'Speech: "{ae.speech_evidence}"')
                else:
                    parts.append('Speech: ""')
                if ae.acoustic_evidence:
                    parts.append(f"Acoustic: [{', '.join(ae.acoustic_evidence)}]")
                lines.append(" | ".join(parts))
        return "\n".join(lines)

    def summary_text(self) -> str:
        """Generate a condensed text summary of all accumulated evidence.

        Includes visual evidence from every round plus, when present, a
        structured Audio Enrichment Findings block produced by the AAVP
        post-observation step.
        """
        lines = []
        for idx, e in enumerate(self.evidences, 1):
            main_text = e.detailed_response

            evidence_lines = []
            for ev in e.key_evidence:
                if isinstance(ev, dict):
                    ts_start = ev.get("timestamp_start")
                    ts_end = ev.get("timestamp_end")
                    desc = ev.get("description", "")
                    if ts_start is not None and ts_end is not None:
                        if desc:
                            evidence_lines.append(f"  • {ts_start:.1f}s - {ts_end:.1f}s: {desc}")
                        else:
                            evidence_lines.append(f"  • {ts_start:.1f}s - {ts_end:.1f}s")

            entry = f"[Round {idx}] {main_text}"
            if evidence_lines:
                entry += "\nKey observations:\n" + "\n".join(evidence_lines)

            lines.append(entry)

        # Append audio enrichment block when any enrichments exist
        audio_text = self.audio_enrichment_summary_text()
        if audio_text:
            lines.append(f"\n[Audio Enrichment Findings]\n{audio_text}")

        if self.query_confidence is not None:
            lines.append(f"\n[Overall Query Confidence: {self.query_confidence:.2f}]")

        return "\n\n".join(lines)


# ======================================================
# Persistence Layer
# ======================================================
class Store:
    def __init__(self, run_dir: str):
        self.root = pathlib.Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "evidence").mkdir(exist_ok=True)

    # Files
    @property
    def plan_initial(self) -> pathlib.Path:
        return self.root / "plan.initial.json"

    def plan_updated(self, k: int) -> pathlib.Path:
        return self.root / f"plan.updated.{k}.json"

    @property
    def history(self) -> pathlib.Path:
        return self.root / "history.jsonl"

    def evidence_dir(self, round_id: int) -> pathlib.Path:
        """Get evidence directory for a round."""
        p = self.root / "evidence" / f"round_{round_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def evidence_json(self, round_id: int) -> pathlib.Path:
        """Get evidence JSON path for a round."""
        return self.evidence_dir(round_id) / "evidence.json"

    @property
    def meta(self) -> pathlib.Path:
        return self.root / "meta.json"

    @property
    def final_answer(self) -> pathlib.Path:
        return self.root / "final_answer.json"
    
    @property
    def conversation_history(self) -> pathlib.Path:
        return self.root / "conversation_history.json"

    # IO helpers
    def write_json(self, path: pathlib.Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def append_history(self, event: Dict[str, Any]) -> None:
        with open(self.history, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Get interaction history as a structured list.
        
        This provides explicit access to interaction history for the Reflector.
        
        Returns:
            List of interaction history entries (dicts) from history.jsonl
        """
        if not self.history.exists():
            return []
        
        history = []
        with open(self.history, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        history.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return history
    
    def save_conversation_history(self, query: str, plan: PlanSpec, bb: Blackboard, final_answer: Dict[str, Any]) -> None:
        """Save complete conversation history including all interactions."""
        history = {
            "query": query,
            "plan": {
                "reasoning": plan.completion_criteria,
                "description": plan.description,
                "watch_config": {
                    "load_mode": plan.watch.load_mode,
                    "fps": plan.watch.fps,
                    "spatial_token_rate": plan.watch.spatial_token_rate.value if hasattr(plan.watch.spatial_token_rate, 'value') else str(plan.watch.spatial_token_rate),
                    "regions": plan.watch.regions
                }
            },
            "execution_history": [],
            "final_answer": final_answer,
            "summary": {
                "total_rounds": len(bb.evidences),
                "total_evidence_items": sum(len(ev.key_evidence) for ev in bb.evidences),
                "video_duration_sec": bb.duration_sec,
                "query_confidence": bb.query_confidence
            }
        }
        
        # Add execution results from each round
        for round_idx, evidence in enumerate(bb.evidences, 1):
            exec_entry = {
                "round_id": round_idx,
                "evidence": {
                    "detailed_response": evidence.detailed_response,
                    "reasoning": evidence.reasoning,
                    "key_evidence": evidence.key_evidence,
                    "frames_used": evidence.frames_used
                },
                "timestamp": evidence.timestamp
            }
            history["execution_history"].append(exec_entry)
        
        self.write_json(self.conversation_history, history)


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        plan_replan_model: Optional[str] = None,
        execute_model: Optional[str] = None,
        project: str = "my-project",
        location: str = "us-central1",
        api_key: Optional[str] = None,
        debug: bool = False,
        max_frame_low: int = 512,
        max_frame_medium: int = 128,
        max_frame_high: int = 128,
    ):
        # Support separate models for plan/replan vs execute
        # If not specified, use the legacy 'model' parameter for both
        self.plan_replan_model = plan_replan_model if plan_replan_model is not None else model
        self.execute_model = execute_model if execute_model is not None else model
        # Keep legacy model for backward compatibility
        self.model = model
        self.project = project
        self.location = location
        self.api_key = api_key or ""
        self.client = None
        self.debug = debug
        self.max_frame_low = max_frame_low
        self.max_frame_medium = max_frame_medium
        self.max_frame_high = max_frame_high
        self.created_clips = []  # Track clips created during execution
        self.temp_clips_dir = None  # Will be set by Controller to be unique per job/worker

        # AAVP audio config — populated by Controller from AVPConfig when present.
        # Safe defaults mean pure-visual code paths are completely unaffected.
        self._avp_config = None
        self.audio_sample_rate: int = 16000
        self.audio_max_snippets_per_round: int = 15
        self.audio_gap_probes: int = 5
        self.audio_closed_tags: List[str] = []

    def initialize_client(self):
        """Initialize the Gemini client (API key or Vertex AI)."""
        try:
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                if self.debug:
                    print("✅ Initialized Gemini client (API key)")
            else:
                self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
                if self.debug:
                    print(f"✅ Initialized Vertex AI client for project: {self.project} @ {self.location}")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            raise
    
    # ---------- Video helpers ----------
    # Note: These methods now delegate to video_utils for cleaner separation

    def create_video_part(
        self,
        video_path: str,
        fps: Optional[float] = None,
        start_offset: Optional[str] = None,
        end_offset: Optional[str] = None,
        media_resolution: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ) -> Part:
        """Create a Gemini Part object for video with metadata.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to sample
            start_offset: Start time offset (e.g., "10.5s")
            end_offset: End time offset (e.g., "60s")
            media_resolution: Resolution level (MEDIA_RESOLUTION_LOW/MEDIUM/HIGH)
            duration_sec: Optional explicit duration in seconds (useful for clips not in metadata cache)
            
        Returns:
            Gemini Part object with video blob and metadata
        """
        # Try to find compressed version first
        compressed_path = find_compressed_video_fallback(video_path)
        actual_video_path = compressed_path if compressed_path else video_path
        
        if not os.path.exists(actual_video_path):
            raise FileNotFoundError(f"Video file not found: {actual_video_path}")
        
        if self.debug and compressed_path:
            print(f"📹 Using compressed video: {actual_video_path}")
        
        # Use video_utils to get MIME type
        mime_type = get_mime_type(actual_video_path)
        
        with open(actual_video_path, "rb") as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)

        video_metadata = None
        if fps or start_offset or end_offset:
            kwargs = {}
            if fps:
                # Select max frame based on media resolution (from config)
                if media_resolution == "low":
                    max_frame = self.max_frame_low
                elif media_resolution == "medium":
                    max_frame = self.max_frame_medium
                elif media_resolution == "high":
                    max_frame = self.max_frame_high
                else:
                    max_frame = self.max_frame_medium
                
                # Calculate duration in seconds from offsets or explicit duration
                duration = duration_sec  # Use explicit duration if provided
                
                if duration is None or duration <= 0:
                    # Try to calculate from offsets
                    if start_offset and end_offset:
                        try:
                            start_sec = float(start_offset.rstrip("s"))
                            end_sec = float(end_offset.rstrip("s"))
                            duration = max(0, end_sec - start_sec)  # Segment duration
                        except ValueError:
                            duration = None
                    elif end_offset:
                        try:
                            duration = float(end_offset.rstrip("s"))  # Duration from start
                        except ValueError:
                            duration = None
                
                # If we still don't have duration, try to get it from video metadata
                if duration is None or duration <= 0:
                    meta_extractor = VideoMetadataExtractor(actual_video_path)
                    video_duration = meta_extractor.duration
                    if start_offset:
                        try:
                            start_sec = float(start_offset.rstrip("s"))
                            duration = max(0, video_duration - start_sec)
                        except ValueError:
                            duration = video_duration
                    else:
                        duration = video_duration
                
                # Adjust fps if it would exceed max frame limit
                if duration and duration > 0:
                    expected_frames = fps * duration
                    if expected_frames > max_frame:
                        adjusted_fps = max_frame / duration
                        if self.debug:
                            print(f"⚠️  Adjusting FPS: {fps:.2f} -> {adjusted_fps:.2f} (max {max_frame} frames)")
                        fps = adjusted_fps
                
                kwargs["fps"] = fps
            
            if start_offset:
                kwargs["startOffset"] = start_offset
            if end_offset:
                kwargs["endOffset"] = end_offset
            
            if kwargs:
                video_metadata = VideoMetadata(**kwargs)
                if self.debug:
                    print(f"📹 Video metadata: fps={fps}, startOffset={start_offset}, endOffset={end_offset}")

        # Note: mediaResolution should be passed via GenerateContentConfig, not Part
        # The media_resolution parameter is still used above for FPS adjustment calculations
        if self.debug and media_resolution:
            print(f"📹 Media resolution: {media_resolution} (will be set in GenerateContentConfig)")
        
        return Part(inlineData=blob, videoMetadata=video_metadata)

    # ---------- Core calls ----------
    def _map_rate_to_media_res(self, rate: SpatialTokenRate | str) -> str:
        """Map SpatialTokenRate or plain string to simple tokens: low|medium|high."""
        if isinstance(rate, SpatialTokenRate):
            return rate.value
        # assume already a string like "low" | "medium" | "high"
        return str(rate).strip().lower()


    def plan(self, query: str, video_meta: Dict[str, Any] = None, prior: Optional[Blackboard] = None, options: Optional[List[str]] = None) -> PlanSpec:
        """Generate observation action plan using LLM.
        
        For initial planning: generates a single observation action based on query and video metadata.
        For replanning: incorporates all evidence and interaction history to plan the next observation.
        
        Args:
            query: User's question about the video
            video_meta: Video metadata (duration, fps, etc.)
            prior: Optional prior blackboard state (for replanning - contains evidence and history)
            options: Optional list of options for MCQ questions
            
        Returns:
            PlanSpec with single observation action (query + watch config)
        """
        if self.client is None:
            self.initialize_client()
        
        if video_meta is None:
            video_meta = {}
        
        # Check if this is a replan (has prior evidence)
        is_replan = prior is not None and len(prior.evidences) > 0
        
        # Generate prompt using PromptManager
        if is_replan:
            # Replanning: include evidence summary and justification
            evidence_summary = prior.summary_text()
            prompt = PromptManager.get_replanning_prompt(query, video_meta, evidence_summary, options)
        else:
            # Initial planning
            prompt = PromptManager.get_planning_prompt(query, video_meta, options)
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🎯 PLANNER INPUT {'(REPLAN)' if is_replan else '(INITIAL)'}")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Video Duration: {video_meta.get('duration_sec', 'unknown')}s")
            if options:
                print(f"Options: {options}")
            if is_replan:
                print(f"Prior Evidence Rounds: {len(prior.evidences)}")
                print(f"Evidence Summary: {evidence_summary[:200]}...")
            print(f"{'='*80}\n")
        
        # Call LLM using plan_replan_model for planning
        resp = _generate_with_backoff(
            self.client.models,
            model=self.plan_replan_model,
            contents=prompt,
        )
        response_text = getattr(resp, "text", str(resp))
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"📋 PLANNER OUTPUT (Raw)")
            print(f"{'='*80}")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            print(f"{'='*80}\n")
        
        # Parse JSON response
        plan_data = parse_json_response(response_text)
        
        if plan_data is None or not validate_against_schema(plan_data, PLAN_SCHEMA):
            # Fallback to default plan if parsing fails
            if self.debug:
                print("⚠️  Failed to parse LLM plan, using fallback")
            return self._get_fallback_plan(query)
        
        # Convert parsed JSON to PlanSpec (take first step only)
        try:
            steps_list = plan_data.get("steps", [])
            if not steps_list:
                if self.debug:
                    print("⚠️  No steps in plan, using fallback")
                return self._get_fallback_plan(query)
            
            # Take only the first step
            s = steps_list[0]
            
            # Handle case-insensitive spatial_token_rate
            spatial_rate_str = str(s["spatial_token_rate"]).strip().lower()
            try:
                spatial_token_rate = SpatialTokenRate(spatial_rate_str)
            except ValueError as e:
                if self.debug:
                    print(f"⚠️  Invalid spatial_token_rate: '{s['spatial_token_rate']}', using 'low' as default")
                spatial_token_rate = SpatialTokenRate.low
            
            # Parse regions with validation
            regions = []
            raw_regions = s.get("regions", [])
            if raw_regions:
                for r in raw_regions:
                    try:
                        # Validate region format
                        if not isinstance(r, (list, tuple)) or len(r) != 2:
                            if self.debug:
                                print(f"⚠️  Invalid region format (expected [start, end]): {r}, skipping")
                            continue
                        # Try to convert to floats
                        start_val = r[0]
                        end_val = r[1]
                        if isinstance(start_val, str) and isinstance(end_val, str):
                            # Both are strings - might be malformed
                            if self.debug:
                                print(f"⚠️  Region contains non-numeric values: {r}, skipping")
                            continue
                        start = float(start_val)
                        end = float(end_val)
                        regions.append((start, end))
                    except (ValueError, TypeError) as e:
                        if self.debug:
                            print(f"⚠️  Error parsing region {r}: {e}, skipping")
                        continue
            
            # --- 5a: Parse AAVP audio fields (degrade gracefully to defaults) ---
            audio_enrich_str = str(s.get("audio_enrichment", "off")).strip().lower()
            try:
                audio_enrichment = AudioEnrichmentScope(audio_enrich_str)
            except ValueError:
                if self.debug:
                    print(f"⚠️  Invalid audio_enrichment: '{audio_enrich_str}', using 'off'")
                audio_enrichment = AudioEnrichmentScope.off

            audio_mode_str = str(s.get("audio_mode", "balanced")).strip().lower()
            try:
                audio_mode = AudioMode(audio_mode_str)
            except ValueError:
                if self.debug:
                    print(f"⚠️  Invalid audio_mode: '{audio_mode_str}', using 'balanced'")
                audio_mode = AudioMode.balanced

            # --- 5b: Build WatchConfig with AAVP fields ---
            watch = WatchConfig(
                load_mode=s["load_mode"],
                fps=float(s["fps"]),
                spatial_token_rate=spatial_token_rate,
                regions=regions,
                audio_enrichment=audio_enrichment,
                audio_mode=audio_mode,
            )
            
            plan = PlanSpec(
                plan_version="v1",
                query=query,
                watch=watch,
                description=s.get("description", ""),
                completion_criteria=plan_data.get("completion_criteria", ""),
            )
            
            if self.debug:
                print(f"\n{'='*80}")
                print(f"✅ PLANNER OUTPUT (Parsed)")
                print(f"{'='*80}")
                print(f"Generated single observation action")
                try:
                    print(f"Load Mode: {plan.watch.load_mode}")
                    print(f"FPS: {plan.watch.fps}")
                    print(f"Spatial Resolution: {plan.watch.spatial_token_rate.value if hasattr(plan.watch.spatial_token_rate, 'value') else plan.watch.spatial_token_rate}")
                    print(f"Regions: {plan.watch.regions}")
                    print(f"Audio Enrichment: {plan.watch.audio_enrichment.value}")
                    print(f"Audio Mode: {plan.watch.audio_mode.value}")
                    print(f"Description: {plan.description}")
                    print(f"\nFull Plan JSON:")
                    print(json.dumps(dataclasses.asdict(plan), indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"Error printing plan: {e}")
                print(f"{'='*80}\n")
            
            return plan
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error converting plan: {e}, using fallback")
            return self._get_fallback_plan(query)
    
    def _get_fallback_plan(self, query: str) -> PlanSpec:
        """Fallback plan if LLM generation fails — safe visual-only uniform scan."""
        watch = WatchConfig(
            load_mode="uniform",
            fps=0.5,
            spatial_token_rate=SpatialTokenRate.low,
            audio_enrichment=AudioEnrichmentScope.off,
        )
        return PlanSpec(
            plan_version="v1",
            query=query,
            watch=watch,
            description="Uniform scan to gather evidence",
        )

    def infer_on_video(
        self,
        video_path: str,
        duration_sec: float,
        sub_query: str,
        context: str,
        start_sec: float,
        end_sec: float,
        watch_cfg: WatchConfig,
        step_id: str,
        original_query: str = "",
    ) -> Evidence:
        """Query Gemini with video file directly (no frame extraction needed).
        
        Args:
            video_path: Path to video file
            sub_query: Question to answer about the video
            context: Accumulated evidence from previous steps
            start_sec: Start time offset in seconds (used when regions are empty)
            end_sec: End time offset in seconds (used when regions are empty)
            watch_cfg: Watch configuration (fps, spatial resolution)
            step_id: Current step identifier
            original_query: Original user query for context
            
        Returns:
            Evidence object with model response
        """
        if self.client is None:
            self.initialize_client()

        # Get fps and media resolution from watch config
        fps = watch_cfg.fps if watch_cfg.fps > 0 else None
        # Use simple "low|medium|high" tokens directly
        media_res = self._map_rate_to_media_res(watch_cfg.spatial_token_rate)
        
        # Get video duration for prompt
        meta_extractor = VideoMetadataExtractor(video_path)
        video_duration = meta_extractor.duration

        # Handle multiple regions: create clips for each region and pass all to API
        parts = []
        clip_paths = []
        frames_used = []
        
        if watch_cfg.load_mode == "region" and watch_cfg.regions:
            # Multiple regions: create a clip for each region
            # Use unique temp_clips_dir per job/worker (set by Controller)
            clip_dir = self.temp_clips_dir if self.temp_clips_dir else os.path.join(os.path.dirname(video_path), "temp_clips")
            
            for region_idx, (reg_start, reg_end) in enumerate(watch_cfg.regions):
                # Validate time range
                reg_start = max(0.0, float(reg_start))
                reg_end = min(duration_sec, float(reg_end))
                
                if reg_start >= reg_end:
                    if self.debug:
                        print(f"⚠️  Skipping invalid region {region_idx}: {reg_start:.1f}s >= {reg_end:.1f}s")
                    continue
                
                # Create clip for this region
                clip_path = create_video_clip(
                    video_path=video_path,
                    start_time=reg_start,
                    end_time=reg_end,
                    clip_name=f"step_{step_id}_region_{region_idx}",
                    temp_dir=clip_dir,
                    debug=self.debug
                )
                
                if clip_path:
                    # Create video part for this clip (no offsets needed for clips)
                    # Calculate duration from region timestamps for FPS adjustment
                    clip_duration = reg_end - reg_start
                    part = self.create_video_part(
                        video_path=clip_path,
                        fps=fps,
                        start_offset=None,
                        end_offset=None,
                        media_resolution=media_res,
                        duration_sec=clip_duration,
                    )
                    parts.append(part)
                    clip_paths.append(clip_path)
                    frames_used.append({"start": reg_start, "end": reg_end, "fps": fps})
                    # Track clips for cleanup
                    self.created_clips.append(clip_path)
                    if self.debug:
                        print(f"📹 Created clip {region_idx}: {clip_path} (range: {reg_start:.1f}s - {reg_end:.1f}s)")
                else:
                    # Fallback: use offsets if clip creation failed
                    if self.debug:
                        print(f"⚠️  Clip creation failed for region {region_idx}, using metadata offsets")
                    part = self.create_video_part(
                        video_path=video_path,
                        fps=fps,
                        start_offset=f"{reg_start}s" if reg_start > 0 else None,
                        end_offset=f"{reg_end}s" if reg_end > 0 else None,
                        media_resolution=media_res,
                    )
                    parts.append(part)
                    frames_used.append({"start": reg_start, "end": reg_end, "fps": fps})
            
            # Use the first region's range for prompt context (or compute overall range)
            if clip_paths or parts:
                overall_start = min(r[0] for r in watch_cfg.regions)
                overall_end = max(r[1] for r in watch_cfg.regions)
                start_sec = overall_start
                end_sec = overall_end
            else:
                # No valid regions, fall back to single range
                if self.debug:
                    print(f"⚠️  No valid regions created, falling back to single range")
                part = self.create_video_part(
                    video_path=video_path,
                    fps=fps,
                    start_offset=f"{start_sec}s" if start_sec > 0 else None,
                    end_offset=f"{end_sec}s" if end_sec > 0 else None,
                    media_resolution=media_res,
                )
                parts = [part]
                frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
        elif watch_cfg.load_mode == "region" and start_sec is not None and end_sec is not None:
            # Single region mode: create one clip or use offsets
            if start_sec >= end_sec:
                if self.debug:
                    print(f"⚠️  Invalid time range for clip: {start_sec:.1f}s >= {end_sec:.1f}s, falling back to offsets")
                clip_path = None
            else:
                # Use unique temp_clips_dir per job/worker (set by Controller)
                clip_dir = self.temp_clips_dir if self.temp_clips_dir else os.path.join(os.path.dirname(video_path), "temp_clips")
                
                # Try to create a clip
                clip_path = create_video_clip(
                    video_path=video_path,
                    start_time=start_sec,
                    end_time=end_sec,
                    temp_dir=clip_dir,
                    debug=self.debug
                )
            
            if clip_path:
                # Use the clip file instead of full video with offsets
                # Calculate duration from time range for FPS adjustment
                clip_duration = end_sec - start_sec
                part = self.create_video_part(
                    video_path=clip_path,
                    fps=fps,
                    start_offset=None,
                    end_offset=None,
                    media_resolution=media_res,
                    duration_sec=clip_duration,
                )
                parts = [part]
                clip_paths = [clip_path]
                frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
                # Track this clip for cleanup
                self.created_clips.append(clip_path)
                if self.debug:
                    print(f"📹 Using clipped video: {clip_path} (original: {start_sec:.1f}s - {end_sec:.1f}s)")
            else:
                # Fallback to using offsets if clip creation failed
                if self.debug:
                    print(f"⚠️  Clip creation failed, falling back to metadata offsets")
                part = self.create_video_part(
                    video_path=video_path,
                    fps=fps,
                    start_offset=f"{start_sec}s" if start_sec > 0 else None,
                    end_offset=f"{end_sec}s" if end_sec > 0 else None,
                    media_resolution=media_res,
                )
                parts = [part]
                frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
        else:
            # For uniform mode, use offsets as normal
            part = self.create_video_part(
                video_path=video_path,
                fps=fps,
                start_offset=f"{start_sec}s" if start_sec > 0 else None,
                end_offset=f"{end_sec}s" if end_sec > 0 else None,
                media_resolution=media_res,
            )
            parts = [part]
            frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
        
        # Determine if this is a region mode
        is_region = (len(clip_paths) > 0) or (watch_cfg.load_mode == "region")
        
        # Build prompt using PromptManager
        # Pass region information if multiple clips are being sent
        regions_for_prompt = None
        if is_region and watch_cfg.regions and len(watch_cfg.regions) > 1:
            regions_for_prompt = watch_cfg.regions
        
        prompt = PromptManager.get_inference_prompt(
            sub_query=sub_query,
            context=context,
            start_sec=start_sec,
            end_sec=end_sec,
            original_query=original_query,
            video_duration_sec=video_duration,
            is_region=is_region,
            regions=regions_for_prompt
        )
        
        if self.debug:
            if len(parts) > 1:
                print(f"🎬 Querying video with {len(parts)} regions:")
                for i, (reg_start, reg_end) in enumerate(watch_cfg.regions):
                    print(f"   Region {i}: {reg_start:.1f}s - {reg_end:.1f}s")
            else:
                if clip_paths:
                    print(f"🎬 Querying video: {clip_paths[0]} (original range: {start_sec:.1f}s - {end_sec:.1f}s)")
                else:
                    print(f"🎬 Querying video: {video_path}")
                    print(f"   Time range: {start_sec:.1f}s - {end_sec:.1f}s")
            print(f"   FPS: {fps}, Resolution: {media_res}")
        
        # Convert media_resolution to config format (e.g., "MEDIA_RESOLUTION_MEDIUM")
        media_resolution_config = None
        if media_res:
            mr = str(media_res).strip().lower()
            if mr == "low":
                media_resolution_config = "MEDIA_RESOLUTION_LOW"
            elif mr == "medium":
                media_resolution_config = "MEDIA_RESOLUTION_MEDIUM"
            elif mr == "high":
                media_resolution_config = "MEDIA_RESOLUTION_HIGH"
        
        # Create GenerateContentConfig with media_resolution
        config_kwargs = {}
        if media_resolution_config:
            config_kwargs["media_resolution"] = media_resolution_config
        
        generate_content_config = None
        if config_kwargs:
            generate_content_config = types.GenerateContentConfig(**config_kwargs)
        
        # Call Gemini API with all video parts (multiple clips or single video)
        contents = [prompt] + parts
        resp = _generate_with_backoff(
            self.client.models,
            model=self.execute_model,
            contents=contents,
            config=generate_content_config,
        )
        response_text = getattr(resp, "text", str(resp))
        
        # Try to parse structured JSON response
        evidence_data = parse_json_response(response_text)
        
        if evidence_data and validate_against_schema(evidence_data, EVIDENCE_SCHEMA):
            # Use structured response - support both old and new formats
            summary = evidence_data.get("summary") or evidence_data.get("detailed_response", response_text)
            detailed_response = evidence_data.get("detailed_response") or evidence_data.get("summary", "")
            
            # Extract key_evidence (new format only)
            key_evidence = evidence_data.get("key_evidence", [])
            
            if self.debug:
                print(f"✅ Parsed structured evidence: {len(key_evidence)} items")
        else:
            # Fallback: Try to extract fields from malformed JSON using regex
            detailed_response = self._extract_json_field(response_text, "detailed_response")
            
            # If detailed_response contains nested JSON with "description", extract it
            if detailed_response and "description" in detailed_response:
                description_value = self._extract_json_field(detailed_response, "description")
                if description_value:
                    detailed_response = description_value
            
            summary = self._extract_json_field(response_text, "summary") or detailed_response or response_text
            key_evidence = self._extract_key_evidence(response_text)
            
            # If we extracted valid data, use it
            if detailed_response and detailed_response != response_text:
                if self.debug:
                    print(f"✅ Extracted fields from malformed JSON: {len(key_evidence)} items")
            else:
                # Final fallback to regex extraction
                summary = response_text
                detailed_response = response_text
                time_anchors = self._extract_timestamps(response_text)
                # Convert single timestamps to ranges (use ±1 second window)
                key_evidence = [{"timestamp_start": max(0.0, t - 1.0), "timestamp_end": t + 1.0, "description": ""} for t in time_anchors]
                if self.debug:
                    print(f"⚠️  Using fallback parsing: {len(key_evidence)} items")
        
        # Normalize key_evidence to full-second intervals using floor for start and ceil for end
        # Clamp to [0, duration_sec] and drop invalid/zero-length intervals; deduplicate
        try:
            raw_ranges = []
            descs = []
            for ev_item in key_evidence:
                if isinstance(ev_item, dict):
                    ts_start = ev_item.get("timestamp_start")
                    ts_end = ev_item.get("timestamp_end")
                    desc = ev_item.get("description", "")
                    if ts_start is not None and ts_end is not None:
                        raw_ranges.append((float(ts_start), float(ts_end)))
                        descs.append(str(desc))
            rounded = round_intervals_full_seconds(raw_ranges, duration=duration_sec)
            # Map rounded intervals to a representative description (prefer first non-empty)
            interval_to_desc: Dict[Tuple[int, int], str] = {}
            for idx, (rs, re) in enumerate(rounded):
                # Find a source description from the original list that rounds to this interval
                chosen_desc = ""
                for jdx, (orig_s, orig_e) in enumerate(raw_ranges):
                    from math import floor, ceil
                    if (floor(orig_s), ceil(orig_e)) == (rs, re):
                        cand = descs[jdx]
                        if cand and not interval_to_desc.get((rs, re)):
                            chosen_desc = cand
                            break
                interval_to_desc[(rs, re)] = chosen_desc
            key_evidence = [
                {"timestamp_start": int(rs), "timestamp_end": int(re), "description": interval_to_desc.get((rs, re), "")}
                for (rs, re) in rounded
            ]
        except Exception:
            # If rounding fails for any reason, leave key_evidence as-is
            pass
        
        # Build model_call metadata
        model_call_metadata = {
            "model": self.execute_model,
            "fps": fps,
            "media_resolution": media_res,
            "prompt_version": "v2_structured"
        }
        
        # Add region information to metadata
        if len(frames_used) > 1:
            model_call_metadata["regions"] = frames_used
            model_call_metadata["num_regions"] = len(frames_used)
        else:
            if frames_used:
                model_call_metadata["start_offset"] = f"{frames_used[0]['start']}s" if frames_used[0]['start'] > 0 else None
                model_call_metadata["end_offset"] = f"{frames_used[0]['end']}s" if frames_used[0]['end'] > 0 else None
        
        return Evidence(
            detailed_response=detailed_response,
            key_evidence=key_evidence,
            reasoning=summary,  # Use summary as reasoning
            frames_used=frames_used,
            model_call=model_call_metadata,
            timestamp=now_iso(),
            round_id=0  # Will be set by Controller
        )
    
    # verify_evidence method removed - Verifier now analyzes evidence without re-watching video

    # ------------------------------------------------------------------
    # 7b. Audio Part helper
    # ------------------------------------------------------------------

    def _create_audio_part(self, audio_path: str) -> Part:
        """Wrap a local WAV file as a Gemini inline-data Part.

        Args:
            audio_path: Absolute path to the extracted WAV file.

        Returns:
            A ``Part`` with ``inlineData`` blob (``audio/wav`` MIME type).
        """
        with open(audio_path, "rb") as fh:
            data = fh.read()
        return Part(inlineData=Blob(mime_type="audio/wav", data=data))

    # ------------------------------------------------------------------
    # 7a. Post-observation audio enrichment
    # ------------------------------------------------------------------

    def enrich_with_audio(
        self,
        evidence: "Evidence",
        video_path: str,
        query: str,
        watch_cfg: "WatchConfig",
        duration_sec: float,
        visual_summary: str = "",
    ) -> List["AudioEnrichment"]:
        """Post-hoc audio enrichment on model-chosen timestamps.

        Runs after the visual observation.  Extracts short WAV snippets
        centred on the timestamps returned by the visual pass, optionally
        adds sparse gap probes, then sends all snippets in one Gemini call
        for structured speech/acoustic interpretation.

        Steps:
            1. Collect midpoints from ``evidence.key_evidence`` timestamps.
            2. If scope is ``evidence_plus_gaps``, add gap probes.
            3. Cap total snippets at ``audio_max_snippets_per_round``.
            4. Extract WAV for each point via ``audio_utils``.
            5. One ``generate_content`` call with all audio Parts + prompt.
            6. Parse response against ``AUDIO_ENRICHMENT_SCHEMA``.
            7. Cleanup temp files.

        Args:
            evidence:       Evidence from the visual observation
                            (provides key_evidence timestamps).
            video_path:     Path to the source video file.
            query:          Original user query (guides relevance).
            watch_cfg:      Watch config (audio_enrichment scope,
                            audio_mode, snippet halfwidth).
            duration_sec:   Total video duration in seconds.
            visual_summary: Text summary of visual findings for
                            cross-reference in the audio prompt.

        Returns:
            List of ``AudioEnrichment`` objects — one per snippet
            for which the model returned a result.  Empty list on
            complete failure.
        """
        if self.client is None:
            self.initialize_client()

        # ── Step 1: collect evidence midpoints ───────────────────────
        snippet_times: List[Dict[str, Any]] = []
        for kev in evidence.key_evidence:
            if isinstance(kev, dict):
                ts_start = kev.get("timestamp_start")
                ts_end = kev.get("timestamp_end")
                if ts_start is not None and ts_end is not None:
                    mid = (float(ts_start) + float(ts_end)) / 2.0
                    snippet_times.append({"center_sec": mid, "source": "evidence"})

        # ── Step 2: gap probes ────────────────────────────────────────
        if watch_cfg.audio_enrichment == AudioEnrichmentScope.evidence_plus_gaps:
            evidence_intervals = [
                (float(kev["timestamp_start"]), float(kev["timestamp_end"]))
                for kev in evidence.key_evidence
                if isinstance(kev, dict)
                and kev.get("timestamp_start") is not None
                and kev.get("timestamp_end") is not None
            ]
            max_gap_probes = getattr(self, "audio_gap_probes", 5)
            gap_times = generate_gap_probes(
                evidence_timestamps=evidence_intervals,
                duration_sec=duration_sec,
                max_probes=max_gap_probes,
            )
            for t in gap_times:
                snippet_times.append({"center_sec": t, "source": "gap_probe"})

        if not snippet_times:
            if self.debug:
                print("ℹ️  [enrich_with_audio] No timestamps to enrich — skipping audio step")
            return []

        # ── Step 3: cap total snippets (evidence first, then gaps) ───
        max_snippets = getattr(self, "audio_max_snippets_per_round", 15)
        if len(snippet_times) > max_snippets:
            evidence_snips = [s for s in snippet_times if s["source"] == "evidence"]
            gap_snips = [s for s in snippet_times if s["source"] == "gap_probe"]
            remaining = max(0, max_snippets - len(evidence_snips))
            snippet_times = evidence_snips + gap_snips[:remaining]
            if self.debug:
                print(
                    f"ℹ️  [enrich_with_audio] Capped to {len(snippet_times)} snippets "
                    f"({len(evidence_snips)} evidence + {len(gap_snips[:remaining])} gap probes)"
                )

        # ── Step 4: extract WAV snippets ─────────────────────────────
        halfwidth = getattr(watch_cfg, "audio_snippet_halfwidth_sec", 2.5)
        sample_rate = getattr(self, "audio_sample_rate", 16000)
        temp_clips_dir = getattr(self, "temp_clips_dir", None)

        audio_parts: List[Part] = []
        temp_files: List[str] = []
        snippet_metadata: List[Dict[str, Any]] = []

        for snip in snippet_times:
            center = snip["center_sec"]
            start = max(0.0, center - halfwidth)
            end = min(duration_sec, center + halfwidth)
            audio_path = extract_audio_snippet(
                video_path,
                center,
                halfwidth,
                sample_rate=sample_rate,
                temp_dir=temp_clips_dir,
                debug=self.debug,
            )
            if audio_path:
                audio_parts.append(self._create_audio_part(audio_path))
                temp_files.append(audio_path)
                snippet_metadata.append({
                    "center_sec": center,
                    "start_sec": start,
                    "end_sec": end,
                    "source": snip["source"],
                })
            elif self.debug:
                print(f"⚠️  [enrich_with_audio] Failed to extract audio at t={center:.1f}s — skipping")

        # ── Step 5: single Gemini call ───────────────────────────────
        if not audio_parts:
            cleanup_audio_artifacts(temp_files, debug=self.debug)
            if self.debug:
                print("⚠️  [enrich_with_audio] All WAV extractions failed — no audio enrichment")
            return []

        closed_tags: List[str] = getattr(
            self, "audio_closed_tags",
            [item for item in AUDIO_ENRICHMENT_SCHEMA["properties"]["audio_results"]
             ["items"]["properties"]["acoustic_evidence"]["items"]["enum"]]
        )

        prompt = PromptManager.get_audio_enrichment_prompt(
            query=query,
            audio_mode=watch_cfg.audio_mode.value,
            snippet_times=snippet_metadata,
            visual_evidence_summary=visual_summary,
            video_duration_sec=duration_sec,
            closed_tags=closed_tags,
        )

        if self.debug:
            print(
                f"\n{'='*80}\n🔊 AUDIO ENRICHMENT — {len(audio_parts)} snippets "
                f"({watch_cfg.audio_mode.value} mode)\n{'='*80}"
            )

        resp = _generate_with_backoff(
            self.client.models,
            model=self.execute_model,
            contents=[prompt] + audio_parts,
        )
        response_text = getattr(resp, "text", str(resp))

        # ── Step 6: parse → AudioEnrichment objects ──────────────────
        enrichments: List[AudioEnrichment] = []
        try:
            data = parse_json_response(response_text)
            if data and validate_against_schema(data, AUDIO_ENRICHMENT_SCHEMA):
                audio_results = data.get("audio_results", [])
                for i, ar in enumerate(audio_results):
                    # Match back to snippet_metadata by position
                    meta = snippet_metadata[i] if i < len(snippet_metadata) else {}
                    center = ar.get("center_sec", meta.get("center_sec", 0.0))
                    enrichments.append(AudioEnrichment(
                        center_sec=float(center),
                        window_start_sec=float(meta.get("start_sec", center - halfwidth)),
                        window_end_sec=float(meta.get("end_sec", center + halfwidth)),
                        speech_evidence=str(ar.get("speech_evidence", "")),
                        acoustic_evidence=list(ar.get("acoustic_evidence", [])),
                        source=str(meta.get("source", "evidence")),
                    ))
                if self.debug:
                    print(
                        f"✅ [enrich_with_audio] Parsed {len(enrichments)} audio enrichments. "
                        f"Summary: {data.get('overall_audio_summary', '')[:150]}"
                    )
            else:
                if self.debug:
                    print("⚠️  [enrich_with_audio] Response failed schema validation — no enrichments stored")
        except Exception as exc:
            if self.debug:
                print(f"⚠️  [enrich_with_audio] Parse error: {exc}")

        # ── Step 7: cleanup ───────────────────────────────────────────
        cleanup_audio_artifacts(temp_files, debug=self.debug)
        return enrichments

    def _extract_timestamps(self, text: str) -> List[float]:
        """Extract timestamps from model response text."""
        import re
        # Look for patterns like "10.5s", "at 23 seconds", "1:30", etc.
        patterns = [
            r'(\d+\.?\d*)\s*(?:seconds?|secs?|s\b)',  # "10.5 seconds" or "10s"
            r'at\s+(\d+\.?\d*)',  # "at 10.5"
            r'(\d+):(\d+)',  # "1:30" (minutes:seconds)
        ]
        timestamps = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if ':' in match.group(0):
                    # Handle mm:ss format
                    mins, secs = match.groups()
                    timestamps.append(float(mins) * 60 + float(secs))
                else:
                    timestamps.append(float(match.group(1)))
        return sorted(list(set(timestamps)))  # Remove duplicates and sort
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from model response text."""
        import re
        # Look for patterns like "confidence: 0.8", "0.7 confidence", etc.
        patterns = [
            r'confidence[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+confidence',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                return conf if conf <= 1.0 else conf / 100.0  # Handle percentage
        return 0.5  # Default medium confidence
    
    def _extract_json_field(self, text: str, field_name: str) -> str:
        """Extract a specific field value from malformed JSON using regex.
        
        Args:
            text: The response text (may contain malformed JSON)
            field_name: The field name to extract (e.g., "detailed_response", "summary", "description")
            
        Returns:
            The extracted field value, or empty string if not found
        """
        import re
        
        # Try to find the field value in the text
        # Pattern: "field_name": "value"
        # Handle escaped quotes and multiline strings
        # Define patterns (avoid raw f-string issues with braces)
        nested_pattern = f'"{field_name}"' + r'\s*:\s*\{' + r'\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        direct_pattern = rf'"{field_name}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        single_quote_pattern = rf'"{field_name}"\s*:\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\''
        nested_direct_pattern = f'"{field_name}"' + r'[^}]*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        patterns = [
            # Nested format: "field_name": { "description": "value" }
            nested_pattern,
            # Direct format: "field_name": "value"
            direct_pattern,
            # Single quotes: "field_name": 'value'
            single_quote_pattern,
            # Also try to extract nested description field directly
            nested_direct_pattern,
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1)
                # Unescape common escape sequences
                value = value.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
                # Clean up any remaining escape sequences
                value = value.replace('\\', '')
                return value
        
        return ""
    
    def _extract_key_evidence(self, text: str) -> List[Dict[str, Any]]:
        """Extract key_evidence array from malformed JSON.
        
        Args:
            text: The response text (may contain malformed JSON)
            
        Returns:
            List of key_evidence items (timestamp_start, timestamp_end, description)
        """
        import re
        
        # Try to extract key_evidence array
        # First try to match new format with timestamp ranges
        key_evidence = []
        
        # Pattern to match new format: "timestamp_start": X, "timestamp_end": Y, "description": "text"
        pattern_range = r'"timestamp_start"\s*:\s*(\d+\.?\d*)\s*,\s*"timestamp_end"\s*:\s*(\d+\.?\d*)\s*,\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        matches = re.finditer(pattern_range, text, re.DOTALL)
        for match in matches:
            ts_start = float(match.group(1))
            ts_end = float(match.group(2))
            description = match.group(3)
            # Unescape common escape sequences
            description = description.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            
            key_evidence.append({
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "description": description
            })
        
        # If no matches found, try legacy format (single timestamp)
        if not key_evidence:
            pattern_legacy = r'"timestamp_sec"\s*:\s*(\d+\.?\d*)\s*,\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
            matches = re.finditer(pattern_legacy, text, re.DOTALL)
            for match in matches:
                timestamp = float(match.group(1))
                description = match.group(2)
                # Unescape common escape sequences
                description = description.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                # Convert single timestamp to range (use ±1 second window)
                key_evidence.append({
                    "timestamp_start": max(0.0, timestamp - 1.0),
                    "timestamp_end": timestamp + 1.0,
                    "description": description
                })
        
        return key_evidence

    # Legacy replan helpers removed in simplified loop

    def synthesize_final_answer(self, plan: PlanSpec, bb: Blackboard) -> Dict[str, Any]:
        """Synthesize final answer from all evidence.
        
        Args:
            plan: Executed plan
            bb: Blackboard with all evidence
            
        Returns:
            Dictionary with final answer and metadata (MCQ format)
        """
        if self.client is None:
            self.initialize_client()
        
        # Get options from blackboard meta if available, normalize to empty list if None
        options = bb.meta.get("options", None)
        options_list = options if options else []
        
        # Generate prompt using PromptManager (always uses MCQ format)
        prompt = PromptManager.get_synthesis_prompt(
            original_query=plan.query,
            all_evidence=bb.summary_text(),
            video_duration=bb.duration_sec or 0.0,
            options=options_list
        )
        
        if self.debug:
            print(f"🔄 Synthesizing final answer...")
            if options_list:
                print(f"📝 MCQ format with {len(options_list)} options")
            else:
                print(f"📝 MCQ format (open-ended question)")
        
        # Call LLM using plan_replan_model for final answer synthesis (reasoning task)
        resp = _generate_with_backoff(
            self.client.models,
            model=self.plan_replan_model,
            contents=prompt,
        )
        response_text = getattr(resp, "text", str(resp))
        
        # Try to parse structured JSON response
        answer_data = parse_json_response(response_text)
        
        # Always use MCQ_SCHEMA for validation
        if answer_data and validate_against_schema(answer_data, MCQ_SCHEMA):
            # Use structured response
            # Add query-level confidence if available from Blackboard
            if bb.query_confidence is not None:
                answer_data["query_confidence"] = bb.query_confidence
            if self.debug:
                print(f"✅ Parsed structured final answer (MCQ format)")
            return answer_data
        else:
            # Fallback: return text as-is in MCQ format
            if self.debug:
                print(f"⚠️  Using unstructured final answer (converted to MCQ format)")
            # Use query-level confidence from Blackboard if available
            query_confidence = bb.query_confidence if bb.query_confidence is not None else 0.5
            return {
                "selected_option": "A",
                "confidence": query_confidence,
                "query_confidence": query_confidence,
                "reasoning": response_text[:500],
                "selected_option_text": response_text[:200] if response_text else ""
            }


# ======================================================
# Validation Utilities
# ======================================================
def clamp_regions(regions: List[List[float]], duration: float) -> List[List[float]]:
    out = []
    for start, end in regions:
        s = max(0.0, float(start))
        e = min(duration, float(end))
        if e > s:
            # Take the largest region: floor start, ceil end
            s_rounded = int(s)  # floor
            e_rounded = int(e) + 1 if e > int(e) else int(e)  # ceil
            out.append([s_rounded, e_rounded])
    return out


# ======================================================
# Planner / Observer / Reflector / Controller
# ======================================================
class Planner:
    """Planner component of the plan-observe-reflect framework.
    
    The Planner generates multi-step action plans that guide the agent
    to perceive key query-related evidence from the video.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def initial_plan(self, query: str, video_meta: Dict[str, Any] = None, options: Optional[List[str]] = None) -> PlanSpec:
        """Generate initial observation action plan.
        
        This is the "plan" phase of the plan-observe-verify framework.
        Creates a single observation action plan (query + watch config).
        """
        return self.client.plan(query, video_meta, options=options)

    # Legacy update_plan removed in simplified loop


class Observer:
    """Observer component of the plan-observe-reflect framework.
    
    The Observer executes actions by observing the video and gathering
    query-related evidence into the evidence list.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def _compute_time_range(self, watch: WatchConfig, duration: float, bb: Blackboard = None) -> Tuple[Optional[float], Optional[float]]:
        """Compute start and end time offsets based on watch configuration.
        
        For region mode with multiple regions, this returns the overall range (min start, max end)
        for prompt context. The actual region processing happens in infer_on_video.
        
        Args:
            watch: Watch configuration
            duration: Video duration in seconds
            bb: Blackboard with accumulated evidence (unused in simplified loop)
        
        Returns:
            (start_sec, end_sec) tuple. None values mean use full video.
        """
        if watch.load_mode == "uniform":
            # Full video scan
            return (0.0, duration)
        elif watch.load_mode == "region":
            # If multiple regions exist, return overall range for prompt context
            # The actual processing will handle each region separately in infer_on_video
            if watch.regions:
                regions = [(float(s), float(e)) for s, e in watch.regions]
                # Return overall range (min start, max end) for prompt context
                start = max(0.0, min(r[0] for r in regions))
                end = min(duration, max(r[1] for r in regions))
                return (start, end)
            else:
                # No region info available, scan full video
                return (0.0, duration)
        else:
            raise ValueError(f"Unknown load_mode: {watch.load_mode}")

    def observe(self, plan: PlanSpec, bb: Blackboard) -> Evidence:
        """Observe the video segment specified by the plan and gather evidence.
        
        This is the "observe" phase of the plan-observe-verify framework.
        Executes the observation action and collects query-related evidence.
        
        Args:
            plan: Plan with query and watch config
            bb: Blackboard with video path and accumulated evidence
            
        Returns:
            Evidence from this observation
        """
        # Get video metadata
        meta_extractor = VideoMetadataExtractor(bb.video_path)
        duration = meta_extractor.duration
        
        # Compute time range
        start_sec, end_sec = self._compute_time_range(plan.watch, duration, bb)
        
        # If querying full video, automatically use uniform mode instead of region mode
        is_full_video = (start_sec is not None and end_sec is not None and 
                        start_sec <= 1.0 and abs(end_sec - duration) <= 1.0)
        
        # Create a copy of watch_cfg to modify if needed
        watch_cfg = plan.watch
        if is_full_video and watch_cfg.load_mode == "region":
            # Switch to uniform mode for full video queries
            watch_cfg = dataclasses.replace(
                watch_cfg,
                load_mode="uniform",
                regions=[]  # Clear regions when using uniform mode
            )
            if self.client.debug:
                print(f"🔄 Detected full video query (0.0s - {duration:.1f}s), switching to uniform mode")
        
        # Get accumulated context
        context = bb.summary_text()
        
        # Build the complete query with options if available
        query_to_use = plan.query
        options = bb.meta.get("options", [])
        if options:
            options_text = "\n".join([f"- {opt}" for opt in options])
            query_to_use = f"{plan.query}\n\nOptions:\n{options_text}"
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"👁️  OBSERVER INPUT")
            print(f"{'='*80}")
            print(f"Query: {query_to_use[:200]}...")
            print(f"Video: {bb.video_path}")
            print(f"Time Range: {start_sec:.1f}s - {end_sec:.1f}s")
            print(f"Load Mode: {watch_cfg.load_mode}")
            print(f"FPS: {watch_cfg.fps}")
            print(f"Spatial Resolution: {watch_cfg.spatial_token_rate}")
            if watch_cfg.regions:
                print(f"Regions: {watch_cfg.regions}")
            print(f"Context: {context[:200]}..." if context else "Context: None")
            print(f"{'='*80}\n")
        
        ev = self.client.infer_on_video(
                video_path=bb.video_path,
                duration_sec=duration,
                sub_query=query_to_use,
                context=context,
                start_sec=start_sec,
                end_sec=end_sec,
                watch_cfg=watch_cfg,
                step_id="1",  # Always "1" in single-action mode
                original_query=plan.query
            )
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"📊 OBSERVER OUTPUT")
            print(f"{'='*80}")
            print(f"Detailed Response: {ev.detailed_response[:300]}...")
            print(f"Key Evidence Count: {len(ev.key_evidence)}")
            for i, kev in enumerate(ev.key_evidence[:5], 1):  # Show first 5
                if isinstance(kev, dict):
                    ts_start = kev.get("timestamp_start", 0)
                    ts_end = kev.get("timestamp_end", 0)
                    desc = kev.get("description", "")[:100]
                    print(f"  {i}. [{int(ts_start)}, {int(ts_end)}]s: {desc}")
            if len(ev.key_evidence) > 5:
                print(f"  ... and {len(ev.key_evidence) - 5} more")
            print(f"Reasoning: {ev.reasoning[:200]}...")
            print(f"{'='*80}\n")

        # === AAVP audio enrichment (post-observation, runs only when enabled) ===
        # Respect the audio_enabled master switch when an AVPConfig is attached.
        # If no config is present (legacy / unit-test paths) fall through to the
        # planner-decided audio_enrichment scope as before.
        _avp_cfg = getattr(self.client, "_avp_config", None)
        _audio_globally_enabled = (_avp_cfg is None) or bool(_avp_cfg.audio_enabled)
        if _audio_globally_enabled and plan.watch.audio_enrichment != AudioEnrichmentScope.off:
            audio_enrichments = self.client.enrich_with_audio(
                evidence=ev,
                video_path=bb.video_path,
                query=plan.query,
                watch_cfg=plan.watch,
                duration_sec=duration,
                visual_summary=ev.detailed_response,
            )
            ev.audio_enrichments = audio_enrichments
            if self.client.debug and audio_enrichments:
                print(
                    f"🔊 Audio enrichment complete: {len(audio_enrichments)} snippet(s) "
                    f"attached to evidence round {ev.round_id}"
                )

        bb.add_evidence(ev)
        return ev


class Reflector:
    """Reflector component of the plan-observe-reflect framework.
    
    The Reflector reflects over the query, evidence list, and interaction history
    to decide whether the evidence is sufficient to answer the query or whether
    additional actions need to be generated.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def reflect(
        self,
        query: str,
        plan: PlanSpec,
        evidence_list: List[Evidence],
        interaction_history: List[Dict[str, Any]] = None,
        video_path: str = "",
        duration_sec: Optional[float] = None,
        is_last_round: bool = False,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Reflect on the current state and decide on next actions.
        
        This is the "reflect" phase of the plan-observe-reflect framework.
        Analyzes the query, gathered evidence, and interaction history to determine
        if evidence is sufficient or if new actions are needed.
        
        Args:
            query: Original user query
            plan: Current plan
            evidence_list: List of evidence gathered so far
            interaction_history: Optional list of interaction history entries
            video_path: Path to video file
            duration_sec: Video duration in seconds
            is_last_round: If True, directly generate final answer using synthesis prompt
            options: Optional list of multiple choice options
            
        Returns:
            Dictionary with:
            - sufficient: bool - whether evidence is sufficient to answer
            - should_update: bool - whether plan should be updated
            - updates: List[Dict] - proposed plan updates
            - reasoning: str - explanation of the reflection decision
            - confidence: float - confidence in the reflection decision
            - final_answer: Optional[Dict] - final answer data (if is_last_round=True)
        """
        # Build blackboard for summary helpers
        bb = Blackboard(video_path=video_path or "")
        for ev in evidence_list:
            bb.add_evidence(ev)
        context_text = bb.summary_text()
        audio_text = bb.audio_enrichment_summary_text()

        # If last round, directly generate final answer using synthesis prompt
        if is_last_round:
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"🎯 LAST ROUND - GENERATING FINAL ANSWER DIRECTLY")
                print(f"{'='*80}")
                print(f"Query: {query}")
                print(f"Evidence Count: {len(evidence_list)}")
                print(f"Video Duration: {duration_sec:.1f}s" if duration_sec else "Duration: unknown")
                if options:
                    print(f"Options: {options}")
                if audio_text:
                    print(f"Audio Enrichments: {audio_text.count(chr(10)) + 1} snippet(s)")
                print(f"{'='*80}\n")

            options_list = options if options else []

            # Generate synthesis prompt — pass audio enrichment block when present
            prompt = PromptManager.get_synthesis_prompt(
                original_query=query,
                all_evidence=context_text,
                video_duration=duration_sec or 0.0,
                options=options_list,
                audio_enrichment_summary=audio_text,
            )
            
            # Call LLM using plan_replan_model for final answer synthesis
            if self.client.client is None:
                self.client.initialize_client()
            
            resp = _generate_with_backoff(
                self.client.client.models,
                model=self.client.plan_replan_model,
                contents=prompt,
            )
            response_text = getattr(resp, "text", str(resp))
            
            # Try to parse structured JSON response
            answer_data = parse_json_response(response_text)
            
            # Always use MCQ_SCHEMA for validation
            if answer_data and validate_against_schema(answer_data, MCQ_SCHEMA):
                # Use structured response
                query_confidence = answer_data.get("confidence", 0.5)
                if self.client.debug:
                    print(f"✅ Parsed structured final answer (MCQ format)")
            else:
                # Fallback: return text as-is in MCQ format
                if self.client.debug:
                    print(f"⚠️  Using unstructured final answer (converted to MCQ format)")
                query_confidence = 0.5
                answer_data = {
                    "selected_option": "A",
                    "confidence": query_confidence,
                    "reasoning": response_text[:500],
                    "selected_option_text": response_text[:200] if response_text else ""
                }
            
            # Add query_confidence to answer_data
            answer_data["query_confidence"] = query_confidence
            
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"✅ FINAL ANSWER GENERATED IN REFLECTION")
                print(f"{'='*80}")
                print(f"Selected Option: {answer_data.get('selected_option', 'N/A')}")
                print(f"Option Text: {answer_data.get('selected_option_text', '')}")
                print(f"Reasoning: {answer_data.get('reasoning', '')[:300]}...")
                print(f"Confidence: {answer_data.get('confidence', 0.0):.2f}")
                print(f"{'='*80}\n")
            
            return {
                "sufficient": True,
                "should_update": False,
                "updates": [],
                "reasoning": f"Final round: Generated answer directly from all evidence.",
                "confidence": 0.9,
                "query_confidence": query_confidence,
                "event": "FINAL_ANSWER_GENERATED",
                "final_answer": answer_data
            }
        
        # === 8a: AAVP LLM reflection when audio enrichments are present ===
        has_audio_evidence = any(ev.audio_enrichments for ev in evidence_list)
        if has_audio_evidence:
            return self._reflect_aavp(
                query=query,
                evidence_list=evidence_list,
                video_path=video_path,
                duration_sec=duration_sec,
                options=options,
                audio_text=audio_text,
                context_text=context_text,
            )

        # === Legacy heuristic path (video-only evidence) ===
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"🔍 VERIFIER INPUT")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Evidence Count: {len(evidence_list)}")
            print(f"Video: {video_path}")
            print(f"Duration: {duration_sec:.1f}s" if duration_sec else "Duration: unknown")
            print(f"{'='*80}\n")

        # Collect regions from current evidence
        regions: List[Tuple[float, float]] = []
        for ev in evidence_list:
            for kev in ev.key_evidence:
                if isinstance(kev, dict):
                    ts_start = kev.get("timestamp_start")
                    ts_end = kev.get("timestamp_end")
                    if ts_start is not None and ts_end is not None:
                        regions.append((float(ts_start), float(ts_end)))
        
        # Deduplicate exact duplicates
        regions = list(set(regions))
        
        # Normalize to full seconds
        if duration_sec:
            regions = round_intervals_full_seconds(regions, duration=duration_sec)
        
        # Merge overlapping/adjacent regions (within 5 seconds)
        if regions:
            regions_sorted = sorted(regions, key=lambda x: x[0])
            merged = []
            for start, end in regions_sorted:
                if not merged:
                    merged.append([start, end])
                else:
                    last_start, last_end = merged[-1]
                    if start <= last_end + 5:  # Overlap or close (within 5s)
                        merged[-1] = [last_start, max(last_end, end)]
                    else:
                        merged.append([start, end])
            regions = [(float(s), float(e)) for s, e in merged]
        
        # Limit to max 10 regions
        if len(regions) > 10:
            if self.client.debug:
                print(f"⚠️  Limiting regions from {len(regions)} to 10")
            regions = regions[:10]
        
        if self.client.debug:
            print(f"Total evidence items: {sum(len(ev.key_evidence) for ev in evidence_list)}")
            print(f"Unique regions (after dedup/merge): {len(regions)}")
            for i, (start, end) in enumerate(regions[:5], 1):
                print(f"  {i}. [{int(start)}, {int(end)}]s")
            if len(regions) > 5:
                print(f"  ... and {len(regions) - 5} more")
        
        # If no regions, insufficient
        if not regions:
            if self.client.debug:
                print(f"\n⚠️  No evidence regions - insufficient")
            return {
                "sufficient": False,
                "should_update": True,
                "updates": [],
                "reasoning": "No evidence regions found; need to replan with different observation strategy.",
                "confidence": 0.7,
                "query_confidence": 0.2,
                "event": "VERIFICATION"
            }
        
        # Assess sufficiency based on evidence quality (no re-watching video)
        has_evidence = any(ev.detailed_response for ev in evidence_list)
        
        # Compute query confidence based on evidence completeness
        if has_evidence:
            # Check if evidence seems complete
            total_evidence_items = sum(len(ev.key_evidence) for ev in evidence_list)
            has_detailed_responses = all(ev.detailed_response for ev in evidence_list)
            
            # Higher confidence if we have multiple evidence items with detailed responses
            if total_evidence_items >= 3 and has_detailed_responses:
                query_confidence = 0.8
            elif total_evidence_items >= 1 and has_detailed_responses:
                query_confidence = 0.6
            else:
                query_confidence = 0.4
        else:
            query_confidence = 0.2
        
        # Sufficient if confidence > 0.5
        sufficient = query_confidence > 0.5
        
        # Build reasoning summary
        reasoning = f"Evidence analysis: {len(evidence_list)} round(s), {len(regions)} unique region(s) after dedup/merge, {sum(len(ev.key_evidence) for ev in evidence_list)} total evidence items. Query confidence: {query_confidence:.2f}"
        
        result = {
            "sufficient": sufficient,
            "should_update": not sufficient,
            "updates": [],
            "reasoning": reasoning,
            "confidence": 0.8,
            "query_confidence": query_confidence,
            "event": "VERIFICATION"
        }
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"✅ VERIFIER OUTPUT")
            print(f"{'='*80}")
            print(f"Sufficient: {result['sufficient']}")
            print(f"Query Confidence: {result['query_confidence']:.2f}")
            print(f"Total Evidence Items: {sum(len(ev.key_evidence) for ev in evidence_list)}")
            print(f"Unique Regions (after dedup/merge): {len(regions)}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Decision: {'✅ SUFFICIENT - Generate Answer' if result['sufficient'] else '❌ INSUFFICIENT - Replan'}")
            print(f"{'='*80}\n")
        
        return result

    # Legacy confidence/decision helpers removed in simplified loop

    # ------------------------------------------------------------------
    # 8b. LLM-based multimodal reflection (AAVP)
    # ------------------------------------------------------------------

    def _reflect_aavp(
        self,
        query: str,
        evidence_list: List[Evidence],
        video_path: str,
        duration_sec: Optional[float],
        options: Optional[List[str]],
        audio_text: str,
        context_text: str,
    ) -> Dict[str, Any]:
        """Query-conditioned alignment reflection using the LLM.

        Called when at least one ``Evidence`` object carries audio enrichments.
        Builds a structured reflection by calling ``get_reflection_prompt`` and
        parsing the response against ``REFLECTION_SCHEMA``.

        On parse failure, falls through transparently to the legacy heuristic
        path so no round is lost.

        Returns a dict compatible with ``Controller.run`` expectations:
            sufficient, should_update, updates, reasoning, confidence,
            query_confidence, event, zoom_region, reason_code,
            required_modalities.
        """
        if self.client.client is None:
            self.client.initialize_client()

        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"🎙️  AAVP REFLECTOR — LLM-based multimodal reflection")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Evidence rounds: {len(evidence_list)}")
            if audio_text:
                print(f"Audio snippets: {audio_text.count(chr(10)) + 1}")
            print(f"{'='*80}\n")

        prompt = PromptManager.get_reflection_prompt(
            query=query,
            evidence_summary=context_text,
            audio_enrichment_summary=audio_text,
            video_duration_sec=duration_sec or 0.0,
            options=options,
        )

        resp = _generate_with_backoff(
            self.client.client.models,
            model=self.client.plan_replan_model,
            contents=prompt,
        )
        response_text = getattr(resp, "text", str(resp))

        reflection_data = parse_json_response(response_text)
        if not (reflection_data and validate_against_schema(reflection_data, REFLECTION_SCHEMA)):
            # ── Fallback: heuristic path ──────────────────────────────
            if self.client.debug:
                print("⚠️  [_reflect_aavp] Schema validation failed — falling back to heuristic")
            # Build a minimal temporary blackboard just to reuse heuristic logic
            bb_tmp = Blackboard(video_path=video_path or "")
            for ev in evidence_list:
                bb_tmp.add_evidence(ev)
            has_evidence = any(ev.detailed_response for ev in evidence_list)
            total_items = sum(len(ev.key_evidence) for ev in evidence_list)
            if has_evidence and total_items >= 1:
                query_confidence = 0.6
            else:
                query_confidence = 0.3
            sufficient = query_confidence > 0.5
            return {
                "sufficient": sufficient,
                "should_update": not sufficient,
                "updates": [],
                "reasoning": "AAVP reflection parse failed; heuristic fallback used.",
                "confidence": 0.5,
                "query_confidence": query_confidence,
                "event": "AAVP_REFLECTION_FALLBACK",
            }

        # ── Parse structured reflection ───────────────────────────────
        sufficient = bool(reflection_data.get("sufficient", False))
        reason_code = str(reflection_data.get("reason_code", "LOW_CONFIDENCE"))
        reasoning = str(reflection_data.get("reasoning", ""))
        confidence = float(reflection_data.get("confidence", 0.5))

        query_support = reflection_data.get("query_support", {})
        query_confidence = float(
            max(query_support.values()) if query_support else (0.8 if sufficient else 0.3)
        )

        zoom_region = reflection_data.get("zoom_region")
        required_modalities = reflection_data.get("required_modalities", {})
        citations = reflection_data.get("citations", [])

        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"✅ AAVP REFLECTOR OUTPUT")
            print(f"{'='*80}")
            print(f"Sufficient:       {sufficient}")
            print(f"Reason Code:      {reason_code}")
            print(f"Confidence:       {confidence:.2f}")
            print(f"Query Support:    {query_support}")
            print(f"Citations:        {len(citations)}")
            print(f"Zoom Region:      {zoom_region}")
            print(f"Required Mods:    {required_modalities}")
            print(f"Reasoning:        {reasoning[:200]}...")
            print(f"{'='*80}\n")

        return {
            "sufficient": sufficient,
            "should_update": not sufficient,
            "updates": [],
            "reasoning": reasoning,
            "confidence": confidence,
            "query_confidence": query_confidence,
            "event": "AAVP_REFLECTION",
            "reason_code": reason_code,
            "zoom_region": zoom_region,
            "required_modalities": required_modalities,
            "citations": citations,
        }


class Controller:
    """Controller orchestrates the plan-observe-reflect framework.
    
    The Controller manages the iterative loop:
    1. Plan: Generate action plans
    2. Observe: Execute actions and gather evidence
    3. Reflect: Assess evidence sufficiency and decide on next actions
    """
    def __init__(self, run_dir: str, video_path: str, client: GeminiClient, options: Optional[List[str]] = None):
        self.store = Store(run_dir)
        self.client = client
        # Find compressed video if available
        compressed_path = find_compressed_video_fallback(video_path)
        actual_video_path = compressed_path if compressed_path else video_path
        if client.debug and compressed_path:
            print(f"✅ Using compressed video: {compressed_path}")
        self.bb = Blackboard(video_path=actual_video_path)
        if options:
            self.bb.meta["options"] = options
        # Track clips created by this controller's client
        self.client.created_clips = []
        # Set unique temp_clips directory per job/worker using run_dir
        # This prevents multiple workers from overwriting each other's temp files
        self.client.temp_clips_dir = os.path.join(run_dir, "temp_clips")
        os.makedirs(self.client.temp_clips_dir, exist_ok=True)
        if client.debug:
            print(f"📁 Using unique temp_clips directory: {self.client.temp_clips_dir}")
        self._init_meta()
        self._propagate_audio_config()

    # ---------- AAVP config propagation ----------
    def _propagate_audio_config(self) -> None:
        """Copy AAVP fields from client._avp_config into client runtime attrs."""
        cfg = getattr(self.client, "_avp_config", None)
        if cfg is None:
            return
        self.client.audio_sample_rate = cfg.audio_sample_rate
        self.client.audio_max_snippets_per_round = cfg.audio_max_snippets_per_round
        self.client.audio_gap_probes = cfg.audio_gap_probes
        self.client.audio_closed_tags = list(cfg.audio_closed_tags)

    # ---------- meta ----------
    def _init_meta(self) -> None:
        # Use VideoMetadataExtractor to get video info (from JSON cache - only duration needed!)
        meta_extractor = VideoMetadataExtractor(self.bb.video_path)
        self.bb.duration_sec = meta_extractor.duration
        
        meta = {
            "video_path": self.bb.video_path,
            "duration_sec": self.bb.duration_sec,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.client.model,  # Legacy field
            "plan_replan_model": self.client.plan_replan_model,
            "execute_model": self.client.execute_model,
            "prompt_versions": {"plan": "v2_json_metadata", "infer": "v2_structured", "update": "v2_structured", "synthesize": "v2_structured"}
        }
        self.store.write_json(self.store.meta, meta)

    # ---------- planning ----------
    def plan(self, query: str) -> PlanSpec:
        """Generate initial plan with video metadata."""
        planner = Planner(self.client)
        # Prepare video metadata for planning (only duration is needed!)
        video_meta = {
            "duration_sec": self.bb.duration_sec,
        }
        # Get options from blackboard if available
        options = self.bb.meta.get("options", None)
        plan = planner.initial_plan(query, video_meta, options=options)
        
        self.store.write_json(self.store.plan_initial, dataclasses.asdict(plan))
        self.store.append_history({
            "ts": now_iso(), "event": "PLAN_INITIAL", "plan_path": str(self.store.plan_initial), "query": query
        })
        return plan

    # ---------- execute observation ----------
    def execute_observation(self, plan: PlanSpec) -> Evidence:
        """Execute the observation action from the plan."""
        # Use round_id = len(evidences) + 1
        round_id = len(self.bb.evidences) + 1
        self.store.append_history({"ts": now_iso(), "event": "EXEC_OBSERVATION_START", "round_id": round_id})
        observer = Observer(self.client)
        ev = observer.observe(plan, self.bb)
        ev.round_id = round_id
        # Persist evidence with interval_map for readability
        ev_dict = dataclasses.asdict(ev)
        try:
            interval_map = {}
            for item in ev.key_evidence:
                if isinstance(item, dict):
                    s = item.get("timestamp_start")
                    e = item.get("timestamp_end")
                    d = item.get("description", "")
                    if s is not None and e is not None:
                        interval_map[f"[{int(s)},{int(e)}]"] = d or ""
            ev_dict["interval_map"] = interval_map
        except Exception:
            pass
        self.store.write_json(self.store.evidence_json(round_id), ev_dict)
        # Extract timestamps from key_evidence (use midpoint for logging)
        timestamps = []
        for e in ev.key_evidence:
            if isinstance(e, dict):
                ts_start = e.get("timestamp_start")
                ts_end = e.get("timestamp_end")
                if ts_start is not None and ts_end is not None:
                    timestamps.append((ts_start + ts_end) / 2.0)
                elif ts_start is not None:
                    timestamps.append(ts_start)
        self.store.append_history({
            "ts": now_iso(), "event": "EXEC_OBSERVATION_END", "round_id": round_id,
            "evidence_path": str(self.store.evidence_json(round_id)),
            "timestamps": timestamps
        })
        return ev

    # ---------- reflection ----------
    # Replanning is driven by the Planner; verifier only verifies evidence and signals sufficiency.
    # Legacy placeholder autofill removed in simplified loop
    
    # Legacy region-merging helpers removed in simplified loop

    # ---------- run loop (plan-observe-reflect framework) ----------
    def run(self, query: str, max_rounds: int = 3) -> Dict[str, Any]:
        """Run the iterative plan-observe-reflect loop.
        
        The framework follows this process:
        1. Plan (single observation action)
        2. Observe (gather evidence)
        3. Reflect & Verify (re-watch regions). If sufficient → answer; else → replan.
        
        Args:
            query: User's question
            max_rounds: Maximum number of plan-observe-reflect cycles
        """
        # Plan phase: Generate initial plan if not present
        plan = self._load_latest_plan() or self.plan(query)
        
        # Track final answer if generated in reflection
        final_answer_from_reflection = None

        for round_idx in range(max_rounds):
            if self.client.debug:
                print(f"\n{'#'*80}")
                print(f"🔄 ROUND {round_idx + 1} / {max_rounds}")
                print(f"{'#'*80}\n")
            
            # Observe: execute the observation action
            observer = Observer(self.client)
            ev = observer.observe(plan, self.bb)
            ev.round_id = round_idx + 1  # Set round number
            self.bb.add_evidence(ev)
            ev_dict = dataclasses.asdict(ev)
            try:
                interval_map = {}
                for item in ev.key_evidence:
                    if isinstance(item, dict):
                        s = item.get("timestamp_start")
                        e = item.get("timestamp_end")
                        d = item.get("description", "")
                        if s is not None and e is not None:
                            interval_map[f"[{int(s)},{int(e)}]"] = d or ""
                ev_dict["interval_map"] = interval_map
            except Exception:
                pass
            self.store.write_json(self.store.evidence_json(ev.round_id), ev_dict)
            # Log timestamps (midpoints)
            timestamps = []
            for e in ev.key_evidence:
                if isinstance(e, dict):
                    ts_start = e.get("timestamp_start")
                    ts_end = e.get("timestamp_end")
                    if ts_start is not None and ts_end is not None:
                        timestamps.append((ts_start + ts_end) / 2.0)
                    elif ts_start is not None:
                        timestamps.append(ts_start)
            self.store.append_history({
                "ts": now_iso(), "event": "OBSERVE_ROUND_END", "round_id": ev.round_id,
                "evidence_path": str(self.store.evidence_json(ev.round_id)),
                "timestamps": timestamps
            })
            
            # Reflect & verify
            reflector = Reflector(self.client)
            evidence_list = self.bb.get_evidence_list()
            interaction_history = self.store.get_interaction_history()
            is_last_round = (round_idx == max_rounds - 1)
            options = self.bb.meta.get("options", None)
            
            reflection = reflector.reflect(
                query=query,
                plan=plan,
                evidence_list=evidence_list,
                interaction_history=interaction_history,
                video_path=self.bb.video_path,
                duration_sec=self.bb.duration_sec,
                is_last_round=is_last_round,
                options=options,
            )
            
            # Check if final answer was generated in reflection (last round)
            if reflection.get("final_answer"):
                final_answer_from_reflection = reflection.get("final_answer")
                if self.client.debug:
                    print(f"✅ Final answer generated in last round reflection.")
                plan.complete = True
                break
            
            query_confidence = reflection.get("query_confidence")
            if query_confidence is not None:
                self.bb.query_confidence = query_confidence
                if self.client.debug:
                    print(f"📊 Query-level confidence: {query_confidence:.2f}")
            self.store.append_history({
                "ts": now_iso(),
                "event": reflection.get("event", "VERIFICATION"),
                "reflection": reflection,
                "query_confidence": query_confidence
            })
            if reflection.get("sufficient", False):
                if self.client.debug:
                    print(f"✅ Evidence sufficient after verification (query_confidence={query_confidence:.2f}).")
                plan.complete = True
                break
            
            # Not sufficient → replan with all evidence and history (only if not last round)
            if not is_last_round:
                if self.client.debug:
                    print(f"❌ Evidence insufficient. Replanning with all evidence and history...")

                video_meta = {"duration_sec": self.bb.duration_sec}
                plan = self.client.plan(query, video_meta=video_meta, prior=self.bb, options=options)

                # Override plan with targeted instructions from the structured reflector.
                reflection_zoom = reflection.get("zoom_region")
                reflection_mods = reflection.get("required_modalities") or {}

                if reflection_zoom:
                    plan.watch.load_mode = "region"
                    plan.watch.regions = [tuple(reflection_zoom)]
                    if self.client.debug:
                        print(f"🔍 Reflector zoom override → region {reflection_zoom}")

                if "audio_enrichment" in reflection_mods:
                    try:
                        plan.watch.audio_enrichment = AudioEnrichmentScope(
                            reflection_mods["audio_enrichment"]
                        )
                        if self.client.debug:
                            print(
                                f"🔊 Reflector audio override → {plan.watch.audio_enrichment}"
                            )
                    except ValueError:
                        pass

                if "spatial_token_rate" in reflection_mods:
                    try:
                        plan.watch.spatial_token_rate = SpatialTokenRate(
                            reflection_mods["spatial_token_rate"]
                        )
                        if self.client.debug:
                            print(
                                f"🖼  Reflector spatial rate override → {plan.watch.spatial_token_rate}"
                            )
                    except ValueError:
                        pass

        # final answer
        if final_answer_from_reflection is not None:
            # Use final answer generated in last round reflection
            final = final_answer_from_reflection
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"✅ USING FINAL ANSWER FROM LAST ROUND REFLECTION")
                print(f"{'='*80}")
                print(f"Total Rounds: {len(self.bb.evidences)}")
                print(f"Total Evidence Items: {sum(len(ev.key_evidence) for ev in self.bb.evidences)}")
                print(f"{'='*80}\n")
        else:
            # Generate final answer using synthesis
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"🎯 SYNTHESIZING FINAL ANSWER")
                print(f"{'='*80}")
                print(f"Total Rounds: {len(self.bb.evidences)}")
                print(f"Total Evidence Items: {sum(len(ev.key_evidence) for ev in self.bb.evidences)}")
                print(f"{'='*80}\n")
            
            final = self.client.synthesize_final_answer(plan, self.bb)
        
        # Extract final answer text from MCQ format
        # Prefer selected_option_text, fallback to reasoning
        final_answer_text = final.get("selected_option_text", "") or final.get("reasoning", "")
        plan.final_answer = final_answer_text
        final["query"] = query  # Add query to final result
        self.store.write_json(self.store.final_answer, final)
        self.store.append_history({"ts": now_iso(), "event": "SYNTHESIZE_ANSWER_END", "final_path": str(self.store.final_answer)})
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"✅ FINAL ANSWER")
            print(f"{'='*80}")
            print(f"Selected Option: {final.get('selected_option', 'N/A')}")
            print(f"Option Text: {final.get('selected_option_text', '')}")
            print(f"Reasoning: {final.get('reasoning', '')[:300]}...")
            print(f"Confidence: {final.get('confidence', 0.0):.2f}")
            # Print evidence timestamps if available
            evidence_timestamps = final.get('evidence_timestamps', [])
            if evidence_timestamps:
                print(f"Evidence Timestamps ({len(evidence_timestamps)} range(s)):")
                for i, ts in enumerate(evidence_timestamps, 1):
                    ts_start = ts.get('timestamp_start', 'N/A')
                    ts_end = ts.get('timestamp_end', 'N/A')
                    desc = ts.get('description', '')
                    if isinstance(ts_start, (int, float)) and isinstance(ts_end, (int, float)):
                        if desc:
                            print(f"  {i}. [{ts_start:.1f}s - {ts_end:.1f}s]: {desc}")
                        else:
                            print(f"  {i}. [{ts_start:.1f}s - {ts_end:.1f}s]")
            print(f"{'='*80}\n")
        
        # Save full conversation history
        self.store.save_conversation_history(query, plan, self.bb, final)

        return {"plan": dataclasses.asdict(plan), "final": final}

    # ---------- helpers ----------
    def _load_latest_plan(self) -> Optional[PlanSpec]:
        if self.store.plan_initial.exists():
            data = json.loads(self.store.plan_initial.read_text())
            plan = plan_from_dict(data)
            # apply the last updated snapshot if present
            k = last_plan_index(self.store)
            if k is not None:
                up = json.loads(self.store.plan_updated(k).read_text())
                plan = plan_from_dict(up)
            return plan
        return None


# ======================================================
# Helpers & Builders
# ======================================================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _enum_val(raw: Any, default: str) -> str:
    """Return the canonical string value of *raw*.

    Works for two cases that arise in ``plan_from_dict``:
    - In-memory round-trip via ``dataclasses.asdict``: ``raw`` is the original
      enum instance; its ``.value`` is already the plain string we need.
    - JSON deserialization: ``raw`` is a plain ``str`` from ``json.loads``.

    Using ``str(enum_instance)`` is intentionally avoided because Python's
    ``str(SomeStrEnum.member)`` returns ``"ClassName.member"`` on some Python
    versions rather than just the value.
    """
    if raw is None:
        return default
    if hasattr(raw, "value"):   # enum instance
        return str(raw.value)
    return str(raw).strip()


def plan_from_dict(d: Dict[str, Any]) -> PlanSpec:
    """Parse PlanSpec from dict (supports both old steps format and new watch format)."""
    # New format: watch directly in plan
    if "watch" in d:
        watch_data = d["watch"]

        try:
            spatial_token_rate = SpatialTokenRate(
                _enum_val(watch_data.get("spatial_token_rate"), "low").lower()
            )
        except ValueError:
            spatial_token_rate = SpatialTokenRate.low

        try:
            audio_enrichment = AudioEnrichmentScope(
                _enum_val(watch_data.get("audio_enrichment"), "off").lower()
            )
        except ValueError:
            audio_enrichment = AudioEnrichmentScope.off

        try:
            audio_mode = AudioMode(
                _enum_val(watch_data.get("audio_mode"), "balanced").lower()
            )
        except ValueError:
            audio_mode = AudioMode.balanced

        watch = WatchConfig(
            load_mode=watch_data["load_mode"],
            fps=float(watch_data["fps"]),
            spatial_token_rate=spatial_token_rate,
            regions=[(float(a), float(b)) for a, b in watch_data.get("regions", [])],
            audio_enrichment=audio_enrichment,
            audio_mode=audio_mode,
            audio_snippet_halfwidth_sec=float(
                watch_data.get("audio_snippet_halfwidth_sec", 2.5)
            ),
        )

        return PlanSpec(
            plan_version=d.get("plan_version", "v1"),
            query=d.get("query", ""),
            watch=watch,
            description=d.get("description", ""),
            completion_criteria=d.get("completion_criteria", ""),
            final_answer=d.get("final_answer"),
            complete=bool(d.get("complete", False)),
        )

    # Legacy format: steps array (take first step)
    elif "steps" in d and d["steps"]:
        s = d["steps"][0]  # Take first step only
        watch_data = s["watch"]

        try:
            spatial_token_rate = SpatialTokenRate(
                _enum_val(watch_data.get("spatial_token_rate"), "low").lower()
            )
        except ValueError:
            spatial_token_rate = SpatialTokenRate.low

        watch = WatchConfig(
            load_mode=watch_data["load_mode"],
            fps=float(watch_data["fps"]),
            spatial_token_rate=spatial_token_rate,
            regions=[(float(a), float(b)) for a, b in watch_data.get("regions", [])],
            # Legacy plans never had audio fields — AudioEnrichmentScope.off is the safe default
        )
        
        return PlanSpec(
            plan_version=d.get("plan_version", "v1"),
            query=d.get("query", ""),
            watch=watch,
            description=s.get("description", ""),
            completion_criteria=d.get("completion_criteria", ""),
            final_answer=d.get("final_answer"),
            complete=bool(d.get("complete", False)),
        )
    
    # Fallback: empty plan
    return PlanSpec(
        plan_version="v1",
        query=d.get("query", ""),
        watch=WatchConfig(load_mode="uniform", fps=0.5, spatial_token_rate=SpatialTokenRate.low),
        description=""
    )


def last_plan_index(store: Store) -> Optional[int]:
    pattern = re.compile(r"plan\.updated\.(\d+)\.json$")
    max_k = None
    for p in store.root.glob("plan.updated.*.json"):
        m = pattern.search(p.name)
        if m:
            k = int(m.group(1))
            max_k = k if (max_k is None or k > max_k) else max_k
    return max_k


def next_plan_index(store: Store) -> int:
    k = last_plan_index(store)
    return 1 if k is None else k + 1


# ======================================================
# CLI
# ======================================================
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Agentic Video Understanding (Separated Loop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("plan")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--video", required=True)
    a.add_argument("--query", required=True)
    a.add_argument("--model", default="gemini-2.0-flash-exp")

    b = sub.add_parser("run")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--video", required=True)
    b.add_argument("--query", required=True)
    b.add_argument("--max-rounds", type=int, default=3)

    e = sub.add_parser("show")
    e.add_argument("--run-dir", required=True)

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "plan":
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=args.video, client=client)
        plan = ctl.plan(args.query)
        print(json.dumps(dataclasses.asdict(plan), indent=2))
        return

    if args.cmd == "run":
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=args.video, client=client)
        result = ctl.run(query=args.query, max_rounds=args.max_rounds)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "show":
        store = Store(args.run_dir)
        if not store.meta.exists():
            raise SystemExit("Run not initialized.")
        meta = json.loads(store.meta.read_text())
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=meta["video_path"], client=client)
        plan = ctl._load_latest_plan()
        plan_dict = dataclasses.asdict(plan) if plan else None
        ev_summary = [dataclasses.asdict(v) for v in ctl.bb.evidences]
        print(json.dumps({"meta": meta, "plan": plan_dict, "evidences": ev_summary}, indent=2))
        return


if __name__ == "__main__":
    main()
