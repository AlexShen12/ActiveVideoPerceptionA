# AAVP Implementation Plan — Active Audio-Visual Perception

> Concrete code changes to evolve **AVP-A** into **AAVP**: a post-hoc audio
> enrichment pipeline that keeps the existing uniform/region video observation
> path and adds a lightweight audio interpretation step using timestamps
> returned by the visual pass.
>
> **Key design decision:** Gemini already returns timestamped `key_evidence`
> from its own sampling of the video blob. Instead of defining an independent
> anchor grid, AAVP extracts audio **around those model-chosen timestamps**
> after the visual observation, then sends the audio snippets to Gemini for
> structured speech/acoustic interpretation. This preserves the full 128K
> token budget for the visual pass and avoids a parallel observation path.
>
> **ASR and sound event detection are performed by Gemini, not locally.**
> Local ffmpeg is used only to extract raw WAV snippets from the video
> container; all semantic interpretation (transcription, acoustic tagging)
> is done by the model via structured prompts.

---

## Table of Contents

1. [New File: `avp/audio_utils.py`](#1-new-file-avpaudio_utilspy)
2. [Schema & Dataclass Changes (`avp/main.py` — Contracts)](#2-schema--dataclass-changes-avpmainpy--contracts)
3. [JSON Schema Changes (`avp/prompt.py` — Schemas)](#3-json-schema-changes-avppromptpy--schemas)
4. [Config Changes (`avp/config.py`)](#4-config-changes-avpconfigpy)
5. [Planner Changes (`avp/main.py` — `GeminiClient.plan`)](#5-planner-changes-avpmainpy--geminiclientplan)
6. [Prompt Changes (`avp/prompt.py` — `PromptManager`)](#6-prompt-changes-avppromptpy--promptmanager)
7. [Observer Changes (`avp/main.py` — `Observer` + `GeminiClient`)](#7-observer-changes-avpmainpy--observer--geminiclient)
8. [Reflector Changes (`avp/main.py` — `Reflector`)](#8-reflector-changes-avpmainpy--reflector)
9. [Controller Changes (`avp/main.py` — `Controller.run`)](#9-controller-changes-avpmainpy--controllerrun)
10. [Storage Changes (`avp/main.py` — `Store` + `Blackboard`)](#10-storage-changes-avpmainpy--store--blackboard)
11. [Export Changes (`avp/__init__.py`)](#11-export-changes-avp__init__py)
12. [Migration & Backwards Compatibility](#12-migration--backwards-compatibility)
13. [Testing Checklist](#13-testing-checklist)

---

## Architecture Overview

```
Round N flow (when audio enrichment is enabled):

  ┌─────────────────────────────────────────────────────────┐
  │  1. PLAN  (text-only LLM call)                          │
  │     → PlanSpec: load_mode, fps, spatial_token_rate,     │
  │       audio_enrichment, audio_mode                      │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  2. OBSERVE  (video blob → Gemini, same as today)       │
  │     → Evidence with key_evidence[{timestamp_start,      │
  │       timestamp_end, description}]                      │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  3. AUDIO ENRICH  (new step — only if plan says so)     │
  │     a. Read timestamps from key_evidence                │
  │     b. ffmpeg: extract WAV snippet per timestamp        │
  │        (+ optional sparse gap probes)                   │
  │     c. One Gemini call: audio Parts + prompt            │
  │        → speech_evidence (verbatim quotes)              │
  │        → acoustic_evidence (closed tags)                │
  │     d. Merge into Evidence.audio_enrichments            │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  4. REFLECT  (LLM or heuristic)                         │
  │     → sufficient? → synthesize final answer             │
  │     → insufficient? → replan (may escalate audio scope) │
  └─────────────────────────────────────────────────────────┘
```

---

## 1. New File: `avp/audio_utils.py`

**Purpose:** Local audio extraction via ffmpeg (parallel to `video_utils.py`).
No ASR or sound classification here — only raw WAV/frame extraction.

### Functions to Create

```python
def check_ffmpeg_audio_support() -> bool:
    """Check that ffmpeg can extract audio (aac/pcm codecs)."""

def extract_audio_snippet(
    video_path: str,
    center_sec: float,
    half_width_sec: float = 2.5,
    output_format: str = "wav",
    sample_rate: int = 16000,
    temp_dir: str | None = None,
    debug: bool = False,
) -> str | None:
    """Extract a short audio clip centered at `center_sec`.

    Produces: {temp_dir}/{stem}_audio_{center_sec:.1f}s.wav
    Uses ffmpeg:
        ffmpeg -ss <start> -to <end> -i <video> -vn -acodec pcm_s16le
               -ar <sample_rate> -ac 1 <output>

    Returns path to WAV file, or None on failure.
    """

def extract_audio_region(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_format: str = "wav",
    sample_rate: int = 16000,
    temp_dir: str | None = None,
    debug: bool = False,
) -> str | None:
    """Extract audio for an arbitrary [start, end] region.

    Same as snippet but with explicit start/end instead of center ± half_width.
    Used when the reflector requests audio over a specific zoom_region.
    """

def generate_gap_probes(
    evidence_timestamps: list[tuple[float, float]],
    duration_sec: float,
    max_probes: int = 5,
    min_gap_sec: float = 10.0,
) -> list[float]:
    """Generate sparse probe times in gaps between evidence timestamps.

    Identifies the largest gaps (≥ min_gap_sec) between consecutive
    evidence intervals, places one probe at each gap midpoint,
    returns up to max_probes times sorted ascending.

    Used to catch off-screen narration or sounds that produced no
    visual key_evidence in the initial observation.
    """

def cleanup_audio_artifacts(
    paths: list[str],
    debug: bool = False,
) -> None:
    """Remove temporary WAV files."""
```

### Notes

- All extraction uses **ffmpeg** (like existing `create_video_clip` in `video_utils.py`).
- WAV at 16 kHz mono: ~64 KB per 2 s snippet — tiny per-snippet cost.
- No local ASR, no local sound classifier. Gemini does all interpretation.

---

## 2. Schema & Dataclass Changes (`avp/main.py` — Contracts)

All changes are in the **Contracts & Schemas** section (lines ~100–240).

### 2a. New Enums

```python
# After SpatialTokenRate (line 106)

class AudioEnrichmentScope(str, Enum):
    """Controls whether and how audio enrichment runs after visual observation."""
    off = "off"                      # No audio (original AVP behavior)
    evidence_only = "evidence_only"  # Extract audio around key_evidence timestamps only
    evidence_plus_gaps = "evidence_plus_gaps"  # Above + sparse probes in timestamp gaps

class AudioMode(str, Enum):
    """Steers the audio interpretation prompt toward speech or sound events."""
    balanced = "balanced"            # Both ASR + sound recognition
    asr_focus = "asr_focus"          # Prioritize verbatim speech transcription
    acoustic_focus = "acoustic_focus"  # Prioritize non-speech event detection
```

### 2b. Extend `WatchConfig`

```python
@dataclass
class WatchConfig:
    load_mode: str                        # "uniform" | "region" (unchanged)
    fps: float
    spatial_token_rate: SpatialTokenRate
    regions: list[tuple[float, float]] = field(default_factory=list)
    # --- AAVP additions ---
    audio_enrichment: AudioEnrichmentScope = AudioEnrichmentScope.off
    audio_mode: AudioMode = AudioMode.balanced
    audio_snippet_halfwidth_sec: float = 2.5  # Half-width of audio window per evidence timestamp
```

**No new `load_mode` values.** `"uniform"` and `"region"` work exactly as before.
Audio enrichment is an **add-on step** after the standard visual observation, not
a different observation path.

### 2c. New `AudioEnrichment` Dataclass

```python
@dataclass
class AudioEnrichment:
    """Audio evidence gathered for a single timestamp from the visual pass."""
    center_sec: float                   # Midpoint of the audio window
    window_start_sec: float             # Audio extraction start
    window_end_sec: float               # Audio extraction end
    speech_evidence: str = ""           # Verbatim quoted transcript (empty if inaudible)
    acoustic_evidence: list[str] = field(default_factory=list)  # Closed-tag labels
    source: str = "evidence"            # "evidence" (from key_evidence) or "gap_probe"
```

### 2d. Extend `Evidence`

```python
@dataclass
class Evidence:
    detailed_response: str = ""
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    frames_used: list[dict[str, Any]] = field(default_factory=list)
    model_call: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    round_id: int = 0
    # --- AAVP addition ---
    audio_enrichments: list[AudioEnrichment] = field(default_factory=list)
```

### 2e. New `ReflectionOutput` Dataclass

```python
@dataclass
class ReflectionOutput:
    """Machine-readable output from the Reflector."""
    sufficient: bool
    query_support: dict[str, float] = field(default_factory=dict)
        # {"visual": 0.8, "speech": 0.3, "acoustic": 0.0}
    citations: list[dict[str, Any]] = field(default_factory=list)
        # [{"timestamp": 45.0, "quote": "and the coach says...", "modality": "speech"}]
    contradiction_with_query: bool = False
    reason_code: str = ""
        # "SUFFICIENT" | "MODALITY_MISMATCH" | "TEMPORAL_GAP" | "LOW_CONFIDENCE" | "NO_EVIDENCE"
    zoom_region: tuple[float, float] | None = None
        # Targeted [t_start, t_end] for replan (with margin)
    required_modalities: dict[str, str] = field(default_factory=dict)
        # {"audio_enrichment": "evidence_plus_gaps", "spatial_token_rate": "medium"}
    reasoning: str = ""
    confidence: float = 0.0
    query_confidence: float = 0.0
```

---

## 3. JSON Schema Changes (`avp/prompt.py` — Schemas)

### 3a. Extend `PLAN_SCHEMA`

Add two new properties to the `steps[*]` item (after `"regions"`, line ~45):

```python
"audio_enrichment": {
    "type": "string",
    "enum": ["off", "evidence_only", "evidence_plus_gaps"],
    "default": "off",
    "description": "Post-observation audio enrichment scope: off | evidence_only (audio around visual timestamps) | evidence_plus_gaps (+ sparse gap probes)"
},
"audio_mode": {
    "type": "string",
    "enum": ["balanced", "asr_focus", "acoustic_focus"],
    "default": "balanced",
    "description": "Audio interpretation focus: balanced | asr_focus | acoustic_focus"
}
```

### 3b. New `AUDIO_ENRICHMENT_SCHEMA`

Add a new schema constant (after `EVIDENCE_SCHEMA`, line ~76):

```python
AUDIO_ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "audio_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "center_sec": {
                        "type": "number",
                        "description": "Midpoint of the audio window in original video seconds"
                    },
                    "speech_evidence": {
                        "type": "string",
                        "description": "Verbatim word-bounded transcript (empty string if inaudible/no speech)"
                    },
                    "acoustic_evidence": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "SILENCE", "SPEECH", "MUSIC", "CHEER", "APPLAUSE",
                                "WHISTLE", "BUZZER", "CRASH", "DOOR", "FOOTSTEPS",
                                "ENGINE", "SIREN", "BELL", "TYPING", "LAUGHTER",
                                "ANIMAL", "WATER", "WIND", "AMBIENT", "OTHER"
                            ]
                        },
                        "description": "Closed-tag acoustic labels for this window"
                    }
                },
                "required": ["center_sec", "speech_evidence", "acoustic_evidence"]
            }
        },
        "overall_audio_summary": {
            "type": "string",
            "description": "Brief summary of all audio findings relevant to the query"
        }
    },
    "required": ["audio_results", "overall_audio_summary"]
}
```

### 3c. New `REFLECTION_SCHEMA`

Add after the enrichment schema:

```python
REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "query_support": {
            "type": "object",
            "properties": {
                "visual": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "speech": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "acoustic": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "number"},
                    "quote_or_tag": {"type": "string"},
                    "modality": {"type": "string", "enum": ["visual", "speech", "acoustic"]}
                },
                "required": ["timestamp", "quote_or_tag", "modality"]
            }
        },
        "contradiction_with_query": {"type": "boolean"},
        "reason_code": {
            "type": "string",
            "enum": ["SUFFICIENT", "MODALITY_MISMATCH", "TEMPORAL_GAP", "LOW_CONFIDENCE", "NO_EVIDENCE"]
        },
        "zoom_region": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[start_sec, end_sec] for targeted re-observation"
        },
        "required_modalities": {
            "type": "object",
            "properties": {
                "audio_enrichment": {
                    "type": "string",
                    "enum": ["off", "evidence_only", "evidence_plus_gaps"]
                },
                "spatial_token_rate": {"type": "string", "enum": ["low", "medium"]}
            }
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["sufficient", "reason_code", "reasoning", "confidence"]
}
```

---

## 4. Config Changes (`avp/config.py`)

### 4a. Extend `AVPConfig`

Add after `max_frame_high` (line 50):

```python
# AAVP audio settings
audio_enabled: bool = False               # Master switch for AAVP audio enrichment
audio_sample_rate: int = 16000            # WAV extraction sample rate (Hz)
audio_snippet_halfwidth_sec: float = 2.5  # Default half-width of audio window (5s total)
audio_max_snippets_per_round: int = 15    # Cap on audio snippets per enrichment call
audio_gap_probes: int = 5                 # Max sparse gap probes when using evidence_plus_gaps
audio_closed_tags: list[str] = field(default_factory=lambda: [
    "SILENCE", "SPEECH", "MUSIC", "CHEER", "APPLAUSE",
    "WHISTLE", "BUZZER", "CRASH", "DOOR", "FOOTSTEPS",
    "ENGINE", "SIREN", "BELL", "TYPING", "LAUGHTER",
    "ANIMAL", "WATER", "WIND", "AMBIENT", "OTHER",
])
```

### 4b. Config JSON Example

```json
{
  "audio_enabled": true,
  "audio_snippet_halfwidth_sec": 2.5,
  "audio_max_snippets_per_round": 15,
  "audio_gap_probes": 5
}
```

---

## 5. Planner Changes (`avp/main.py` — `GeminiClient.plan`)

### Location: `GeminiClient.plan` method (lines ~529–682)

### 5a. Audio-Aware Plan Parsing

After parsing `spatial_token_rate` (line ~617), add parsing for the new fields:

```python
# Parse audio fields (AAVP)
audio_enrich_str = str(s.get("audio_enrichment", "off")).strip().lower()
try:
    audio_enrichment = AudioEnrichmentScope(audio_enrich_str)
except ValueError:
    audio_enrichment = AudioEnrichmentScope.off

audio_mode_str = str(s.get("audio_mode", "balanced")).strip().lower()
try:
    audio_mode = AudioMode(audio_mode_str)
except ValueError:
    audio_mode = AudioMode.balanced
```

### 5b. Include New Fields in `WatchConfig` Construction

Modify the `WatchConfig` construction (lines ~646–650):

```python
watch = WatchConfig(
    load_mode=s["load_mode"],
    fps=float(s["fps"]),
    spatial_token_rate=spatial_token_rate,
    regions=regions,
    # AAVP fields
    audio_enrichment=audio_enrichment,
    audio_mode=audio_mode,
)
```

### 5c. Fallback Plan

Update `_get_fallback_plan` (line ~684) to include audio defaults:

```python
def _get_fallback_plan(self, query: str) -> PlanSpec:
    watch = WatchConfig(
        load_mode="uniform",
        fps=0.5,
        spatial_token_rate=SpatialTokenRate.low,
        audio_enrichment=AudioEnrichmentScope.off,
    )
    return PlanSpec(plan_version="v1", query=query, watch=watch,
                    description="Uniform scan to gather evidence")
```

---

## 6. Prompt Changes (`avp/prompt.py` — `PromptManager`)

### 6a. Extend `get_planning_prompt` (lines ~151–312)

Add to the **Planning Framework** section (after line ~170, the sampling granularity bullet):

```
4. **Audio Enrichment** (AAVP — post-observation audio analysis):
   - "audio_enrichment": Controls whether audio is analyzed after the visual pass
     * "off": No audio enrichment (pure visual, default for backward compatibility)
     * "evidence_only": After visual observation, extract audio around each
       returned key_evidence timestamp and send to model for speech/acoustic analysis
     * "evidence_plus_gaps": Same as above, plus sparse audio probes in timeline
       gaps between evidence timestamps (catches off-screen narration/sounds)
   - "audio_mode": Steers the audio interpretation prompt
     * "balanced": Both speech transcription and sound recognition
     * "asr_focus": Prioritize verbatim speech extraction (for reasoning-heavy queries)
     * "acoustic_focus": Prioritize non-speech event detection (for event-tracking queries)

**Audio Trigger Heuristics:**
- If query contains words like "said," "told," "narrator," "dialogue," "speaks," "voice,"
  "heard," "listen" → set audio_enrichment to "evidence_only" with audio_mode "asr_focus"
- If query references sounds: "whistle," "buzzer," "crash," "music," "applause," "bell" →
  set audio_enrichment to "evidence_only" with audio_mode "acoustic_focus"
- If query is purely visual ("color of," "how many," "identify the object") →
  keep audio_enrichment as "off"
- When in doubt, use "evidence_only" with "balanced" — it adds only one extra API call
```

Add a **new few-shot exemplar** for audio-enriched plans:

```
- Audio-enriched uniform scan (narrator query):
{{
"reasoning": "Query asks what the narrator said. Need audio enrichment after visual scan.",
"steps": [
    {{
    "step_id": "1",
    "description": "Uniform scan to locate key visual moments, then enrich with audio",
    "sub_query": "What does the narrator say about the recipe ingredient?",
    "load_mode": "uniform",
    "fps": 0.5,
    "spatial_token_rate": "low",
    "regions": [],
    "audio_enrichment": "evidence_plus_gaps",
    "audio_mode": "asr_focus"
    }}
],
"completion_criteria": "Locate key moments and capture narrator speech"
}}
```

### 6b. Extend `get_replanning_prompt` (lines ~437–521)

Add to **Replanning Strategy** bullet list (after line ~486):

```
- If previous visual-only scan was insufficient and query involves speech/sound →
  enable audio_enrichment "evidence_only" or "evidence_plus_gaps"
- If reflector flagged MODALITY_MISMATCH → use the zoom_region with
  audio_enrichment "evidence_plus_gaps" and spatial_token_rate "medium"
- If reflector flagged TEMPORAL_GAP → widen the observation window or
  enable "evidence_plus_gaps" to catch off-screen cues in timeline gaps
```

### 6c. New Method: `get_audio_enrichment_prompt`

Add a new static method to `PromptManager`:

```python
@staticmethod
def get_audio_enrichment_prompt(
    query: str,
    audio_mode: str,
    snippet_times: list[dict],  # [{"center_sec": 45.0, "start_sec": 42.5, "end_sec": 47.5, "source": "evidence"}, ...]
    visual_evidence_summary: str,
    video_duration_sec: float,
    closed_tags: list[str],
) -> str:
    """Generate prompt for post-observation audio enrichment.

    The model receives multiple audio WAV Parts (one per snippet) alongside
    this text prompt. It must return structured JSON with speech transcripts
    and acoustic tags per snippet.

    Args:
        query: Original user query
        audio_mode: "balanced" | "asr_focus" | "acoustic_focus"
        snippet_times: Metadata for each audio snippet being sent
        visual_evidence_summary: Text summary of visual evidence from this round
        video_duration_sec: Total video duration
        closed_tags: Allowed acoustic event tags
    """
```

The prompt body should:

1. List each snippet with its time window and source (evidence vs gap probe).
2. Instruct the model to produce `speech_evidence` as **verbatim word-bounded
   quotes only** (empty string `""` if nothing audible / no speech).
3. Instruct the model to produce `acoustic_evidence` as a list drawn **only**
   from the `closed_tags` enum.
4. Include the query so evidence is query-relevant.
5. Include the visual evidence summary so the model can cross-reference.
6. If `audio_mode == "asr_focus"`, emphasize speech extraction in instructions.
7. If `audio_mode == "acoustic_focus"`, emphasize sound event tagging.
8. Require output as JSON matching `AUDIO_ENRICHMENT_SCHEMA`.

### 6d. New Method: `get_reflection_prompt`

Add a new static method to `PromptManager`:

```python
@staticmethod
def get_reflection_prompt(
    query: str,
    evidence_summary: str,
    audio_enrichment_summary: str,
    video_duration_sec: float,
    options: list[str] | None = None,
) -> str:
    """Generate prompt for the query-conditioned alignment reflector.

    Instructs the model to:
    1. Evaluate visual ∩ audio alignment against Q.
    2. Cite specific timestamps and quotes from evidence.
    3. Flag MODALITY_MISMATCH or TEMPORAL_GAP if found.
    4. Produce machine-readable JSON matching REFLECTION_SCHEMA.
    """
```

### 6e. Extend `get_synthesis_prompt` (lines ~523–605)

Add to the **Evidence from All Observation Rounds** section: if audio
enrichment data exists, format it as a structured block:

```
**Audio Enrichment Findings:**
- t=45.0s [evidence] | Speech: "Welcome back everyone" | Acoustic: [SPEECH, MUSIC]
- t=120.0s [evidence] | Speech: "" | Acoustic: [WHISTLE, CHEER]
- t=75.0s [gap_probe] | Speech: "Now add the butter" | Acoustic: [SPEECH]
...
```

This is generated by a helper on `Blackboard` (see Section 10).

---

## 7. Observer Changes (`avp/main.py` — `Observer` + `GeminiClient`)

### 7a. New Method: `GeminiClient.enrich_with_audio`

Add after `infer_on_video` (around line ~1047):

```python
def enrich_with_audio(
    self,
    evidence: Evidence,
    video_path: str,
    query: str,
    watch_cfg: WatchConfig,
    duration_sec: float,
    visual_summary: str = "",
) -> list[AudioEnrichment]:
    """Post-hoc audio enrichment on model-chosen timestamps.

    1. Collect midpoints from evidence.key_evidence timestamps.
    2. If scope is evidence_plus_gaps, also generate gap probes.
    3. Cap total snippets at audio_max_snippets_per_round.
    4. For each point, extract WAV via audio_utils.extract_audio_snippet.
    5. Build one generate_content call with all audio Parts + prompt.
    6. Parse response against AUDIO_ENRICHMENT_SCHEMA.
    7. Return list of AudioEnrichment objects.

    Args:
        evidence: Evidence from the visual observation (contains key_evidence timestamps)
        video_path: Path to video file
        query: Original user query
        watch_cfg: Current watch config (for audio_mode, snippet halfwidth)
        duration_sec: Video duration in seconds
        visual_summary: Text summary of visual findings (for cross-reference in prompt)

    Returns:
        List of AudioEnrichment objects (one per snippet)
    """
```

**Implementation outline:**

```python
from .audio_utils import extract_audio_snippet, generate_gap_probes, cleanup_audio_artifacts

# Step 1: Collect evidence midpoints
snippet_times = []
for kev in evidence.key_evidence:
    if isinstance(kev, dict):
        ts_start = kev.get("timestamp_start")
        ts_end = kev.get("timestamp_end")
        if ts_start is not None and ts_end is not None:
            mid = (ts_start + ts_end) / 2.0
            snippet_times.append({
                "center_sec": mid,
                "source": "evidence",
            })

# Step 2: Add gap probes if scope is evidence_plus_gaps
if watch_cfg.audio_enrichment == AudioEnrichmentScope.evidence_plus_gaps:
    evidence_intervals = [
        (kev["timestamp_start"], kev["timestamp_end"])
        for kev in evidence.key_evidence
        if isinstance(kev, dict)
        and kev.get("timestamp_start") is not None
        and kev.get("timestamp_end") is not None
    ]
    gap_times = generate_gap_probes(
        evidence_timestamps=evidence_intervals,
        duration_sec=duration_sec,
        max_probes=self.audio_gap_probes,
    )
    for t in gap_times:
        snippet_times.append({"center_sec": t, "source": "gap_probe"})

# Step 3: Cap total snippets
max_snippets = getattr(self, 'audio_max_snippets_per_round', 15)
if len(snippet_times) > max_snippets:
    # Keep all evidence snippets, trim gap probes first
    evidence_snips = [s for s in snippet_times if s["source"] == "evidence"]
    gap_snips = [s for s in snippet_times if s["source"] == "gap_probe"]
    remaining = max_snippets - len(evidence_snips)
    snippet_times = evidence_snips + gap_snips[:max(0, remaining)]

# Step 4: Extract audio
halfwidth = watch_cfg.audio_snippet_halfwidth_sec
temp_files = []
audio_parts = []
snippet_metadata = []

for snip in snippet_times:
    center = snip["center_sec"]
    start = max(0.0, center - halfwidth)
    end = min(duration_sec, center + halfwidth)
    audio_path = extract_audio_snippet(
        video_path, center, halfwidth,
        sample_rate=self.audio_sample_rate,
        temp_dir=self.temp_clips_dir,
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

# Step 5: One Gemini call
if not audio_parts:
    cleanup_audio_artifacts(temp_files)
    return []

closed_tags = getattr(self, 'audio_closed_tags', AUDIO_ENRICHMENT_SCHEMA[...])

prompt = PromptManager.get_audio_enrichment_prompt(
    query=query,
    audio_mode=watch_cfg.audio_mode.value,
    snippet_times=snippet_metadata,
    visual_evidence_summary=visual_summary,
    video_duration_sec=duration_sec,
    closed_tags=closed_tags,
)

resp = self.client.models.generate_content(
    model=self.execute_model,
    contents=[prompt] + audio_parts,
)

# Step 6: Parse response → AudioEnrichment objects
# Step 7: Cleanup temp files
cleanup_audio_artifacts(temp_files)
# Return list of AudioEnrichment
```

### 7b. New Helper Method on `GeminiClient`

```python
def _create_audio_part(self, audio_path: str) -> Part:
    """Create a Gemini Part from a WAV audio file."""
    with open(audio_path, "rb") as f:
        data = f.read()
    return Part(inlineData=Blob(mime_type="audio/wav", data=data))
```

### 7c. Modify `Observer.observe` (lines ~1328–1419)

The visual observation path is **unchanged**. Add audio enrichment **after**
the visual observation returns:

```python
def observe(self, plan: PlanSpec, bb: Blackboard) -> Evidence:
    # ... existing visual observation code (unchanged) ...
    # ev = self.client.infer_on_video(...)
    # bb.add_evidence(ev)

    # === AAVP audio enrichment (post-observation) ===
    if plan.watch.audio_enrichment != AudioEnrichmentScope.off:
        meta_extractor = VideoMetadataExtractor(bb.video_path)
        duration = meta_extractor.duration
        audio_enrichments = self.client.enrich_with_audio(
            evidence=ev,
            video_path=bb.video_path,
            query=plan.query,
            watch_cfg=plan.watch,
            duration_sec=duration,
            visual_summary=ev.detailed_response,
        )
        ev.audio_enrichments = audio_enrichments

    return ev
```

---

## 8. Reflector Changes (`avp/main.py` — `Reflector`)

### Location: `Reflector.reflect` method (lines ~1432–1670)

### 8a. Upgrade to LLM-Based Reflection When Audio Evidence Exists

The current non-last-round path (lines ~1553–1670) is **purely heuristic**.
When audio enrichments are present, use an **LLM call** instead:

```python
# In Reflector.reflect, before the heuristic path:

has_audio_evidence = any(ev.audio_enrichments for ev in evidence_list)

if has_audio_evidence and not is_last_round:
    return self._reflect_aavp(query, evidence_list, interaction_history,
                               video_path, duration_sec, options)

# ... existing heuristic path for legacy video-only evidence ...
```

### 8b. New Method: `Reflector._reflect_aavp`

```python
def _reflect_aavp(
    self,
    query: str,
    evidence_list: list[Evidence],
    interaction_history: list[dict],
    video_path: str,
    duration_sec: float,
    options: list[str] | None,
) -> dict[str, Any]:
    """LLM-based query-conditioned alignment reflection (AAVP).

    1. Build structured summary of visual evidence + audio enrichments.
    2. Call plan_replan_model with get_reflection_prompt.
    3. Parse against REFLECTION_SCHEMA.
    4. Return dict compatible with Controller expectations:
       - sufficient, should_update, updates, reasoning, confidence,
         query_confidence, event, zoom_region, reason_code, required_modalities
    """
```

**Key behaviors:**

- **Quote-span enforcement:** The prompt requires `citations[]` with `quote_or_tag`
  and `timestamp` — grounded in actual evidence.
- **Contradiction logic:** If `reason_code == "MODALITY_MISMATCH"`, the output
  includes `zoom_region` and `required_modalities` with escalated scope
  (e.g. `evidence_plus_gaps` + higher spatial).
- **Fallback:** If LLM parsing fails, fall through to the existing heuristic path.

### 8c. Preserve Legacy Path

The existing heuristic reflector remains as-is for rounds that only have
visual evidence (no `audio_enrichments`). Full backward compatibility.

---

## 9. Controller Changes (`avp/main.py` — `Controller.run`)

### Location: `Controller.run` method (lines ~1787–1930)

### 9a. Pass Config to Client

In `Controller.__init__` (line ~1683), propagate AAVP config:

```python
if hasattr(client, '_avp_config') and client._avp_config:
    cfg = client._avp_config
    self.client.audio_sample_rate = cfg.audio_sample_rate
    self.client.audio_max_snippets_per_round = cfg.audio_max_snippets_per_round
    self.client.audio_gap_probes = cfg.audio_gap_probes
    self.client.audio_closed_tags = cfg.audio_closed_tags
```

### 9b. Handle Structured Replan from Reflector

In the replan branch (lines ~1889–1899), if the reflector returned
`zoom_region` and `required_modalities`, use them to **override** the
planner's next plan:

```python
if not is_last_round:
    reflection_zoom = reflection.get("zoom_region")
    reflection_mods = reflection.get("required_modalities", {})

    video_meta = {"duration_sec": self.bb.duration_sec}
    plan = self.client.plan(query, video_meta=video_meta, prior=self.bb, options=options)

    # Override with reflector's targeted instructions
    if reflection_zoom:
        plan.watch.load_mode = "region"
        plan.watch.regions = [tuple(reflection_zoom)]
    if "audio_enrichment" in reflection_mods:
        plan.watch.audio_enrichment = AudioEnrichmentScope(reflection_mods["audio_enrichment"])
    if "spatial_token_rate" in reflection_mods:
        plan.watch.spatial_token_rate = SpatialTokenRate(reflection_mods["spatial_token_rate"])
```

### 9c. No Observation Dispatch Changes

The existing observer call (`observer.observe(plan, self.bb)`) handles
audio enrichment internally when `audio_enrichment != off`. No controller
loop changes needed.

---

## 10. Storage Changes (`avp/main.py` — `Store` + `Blackboard`)

### 10a. `Blackboard.audio_enrichment_summary_text`

Add a new method to `Blackboard`:

```python
def audio_enrichment_summary_text(self) -> str:
    """Format all audio enrichment evidence for prompts (reflector, synthesis)."""
    lines = []
    for ev in self.evidences:
        for ae in ev.audio_enrichments:
            parts = [f"t={ae.center_sec:.1f}s [{ae.source}]"]
            if ae.speech_evidence:
                parts.append(f'Speech: "{ae.speech_evidence}"')
            else:
                parts.append('Speech: ""')
            if ae.acoustic_evidence:
                tags = ", ".join(ae.acoustic_evidence)
                parts.append(f"Acoustic: [{tags}]")
            lines.append(" | ".join(parts))
    return "\n".join(lines)
```

Update `Blackboard.summary_text` to include audio enrichment when present:

```python
def summary_text(self) -> str:
    # ... existing code builds `lines` ...

    audio_text = self.audio_enrichment_summary_text()
    if audio_text:
        lines.append(f"\n[Audio Enrichment Findings]\n{audio_text}")

    return "\n\n".join(lines)
```

### 10b. `Store` — Persist Audio Enrichment Data

`Store.evidence_json(round_id)` already saves `dataclasses.asdict(ev)`.
Since `AudioEnrichment` is a dataclass and `Evidence.audio_enrichments`
is a list of dataclasses, `dataclasses.asdict` serializes them automatically.
**No Store changes needed** — just verify the round-trip.

---

## 11. Export Changes (`avp/__init__.py`)

Add the new public symbols:

```python
from .main import (
    # ... existing ...
    AudioEnrichmentScope,
    AudioMode,
    AudioEnrichment,
    ReflectionOutput,
)

from .audio_utils import (
    extract_audio_snippet,
    extract_audio_region,
    generate_gap_probes,
    cleanup_audio_artifacts,
    check_ffmpeg_audio_support,
)

from .prompt import (
    # ... existing ...
    AUDIO_ENRICHMENT_SCHEMA,
    REFLECTION_SCHEMA,
)
```

Update `__all__` accordingly.

---

## 12. Migration & Backwards Compatibility

| Concern | Mitigation |
|---------|-----------|
| Existing configs without `audio_enabled` | `AVPConfig.audio_enabled` defaults to `False`; all audio fields have safe defaults. |
| Existing plans without `audio_enrichment` | `WatchConfig.audio_enrichment` defaults to `AudioEnrichmentScope.off`; plan parsing treats missing field as `"off"`. |
| `load_mode` values unchanged | `"uniform"` and `"region"` work exactly as before. No new load modes. |
| Reflector heuristic path | Preserved for rounds with no `audio_enrichments`; LLM reflector only activates when audio evidence exists. |
| `Evidence.audio_enrichments` on old data | Defaults to empty list; `summary_text()` skips empty enrichments. |
| `dataclasses.asdict` serialization | New fields serialize cleanly as they are all primitives / lists / nested dataclasses. |

---

## 13. Testing Checklist

### Unit Tests

- [ ] `audio_utils.extract_audio_snippet` — produces valid WAV, correct duration, clamps to video bounds
- [ ] `audio_utils.extract_audio_region` — produces valid WAV for explicit start/end
- [ ] `audio_utils.generate_gap_probes` — finds correct gap midpoints, respects max_probes and min_gap
- [ ] `WatchConfig` with new fields round-trips through `dataclasses.asdict` / reconstruction
- [ ] `PLAN_SCHEMA` validates plans with and without audio fields
- [ ] `AUDIO_ENRICHMENT_SCHEMA` validates well-formed and malformed responses
- [ ] `REFLECTION_SCHEMA` validates reflector output
- [ ] `Blackboard.audio_enrichment_summary_text` — formats correctly with mixed evidence

### Integration Tests

- [ ] **Pure visual query** (audio_enabled=False): pipeline unchanged, no audio extraction
- [ ] **Pure visual query** (audio_enabled=True): planner sets `off`, no enrichment occurs
- [ ] **Speech query** (audio_enabled=True): planner selects `evidence_only` + `asr_focus`, audio extracted around key_evidence timestamps, speech transcripts returned
- [ ] **Sound query** (audio_enabled=True): planner selects `evidence_only` + `acoustic_focus`, closed tags returned
- [ ] **Gap probe query**: planner selects `evidence_plus_gaps`, gap probes fire in timeline gaps, narrator speech captured in gaps
- [ ] **Reflector MODALITY_MISMATCH**: acoustic tag present, visual empty → replan with `evidence_plus_gaps` + `medium`
- [ ] **Reflector SUFFICIENT**: visual + speech agree → answer generated
- [ ] **Snippet cap**: round with 20 key_evidence items → capped to 15 snippets (evidence prioritized over gaps)

### Cost / Quality Ablation (Manual)

- [ ] Compare token usage: visual-only round vs visual + audio enrichment round
- [ ] Compare accuracy on VideoMME audio-heavy subset (with and without enrichment)
- [ ] Measure latency delta: one extra API call per round for audio enrichment
- [ ] Verify gap probes catch off-screen narration on sample cases
