"""
Active Video Perception Framework (AVP)
================================================

A multi-step video analysis framework using Gemini API with:
- Intelligent planning and replanning
- Coarse-to-fine progressive exploration
- Structured JSON outputs with validation
- Centralized prompt management
- Post-hoc audio enrichment (AAVP)
"""

from .main import (
    # Data structures
    PlanSpec,
    Evidence,
    Blackboard,
    WatchConfig,
    SpatialTokenRate,

    # AAVP data structures
    AudioEnrichmentScope,
    AudioMode,
    AudioEnrichment,
    ReflectionOutput,

    # Core components
    GeminiClient,
    Planner,
    Observer,
    Reflector,
    Controller,
    VideoMetadataExtractor,

    # Storage
    Store,
)

from .prompt import (
    PromptManager,
    parse_json_response,
    validate_against_schema,
    PLAN_SCHEMA,
    EVIDENCE_SCHEMA,
    FINAL_ANSWER_SCHEMA,

    # AAVP schemas
    AUDIO_ENRICHMENT_SCHEMA,
    REFLECTION_SCHEMA,
)

from .video_utils import (
    VideoMetadataExtractor,
    sha256_file,
    get_mime_type,
    find_compressed_video_fallback,
    get_video_path,
    get_video_info,
    print_video_info,
    validate_video_file,
    format_duration,
    set_metadata_source,
    load_video_metadata_from_json,
)

from .audio_utils import (
    check_ffmpeg_audio_support,
    extract_audio_snippet,
    extract_audio_region,
    generate_gap_probes,
    cleanup_audio_artifacts,
)

from .timbre_segmentation import compute_timbre_boundaries
from .local_asr import transcribe_wav
from .timbre_anchor_cache import (
    TimbreAnchor,
    preplan_anchors,
    filter_anchors_in_regions,
    load_cache as load_timbre_cache,
    save_cache as save_timbre_cache,
)

from .config import (
    AVPConfig,
    load_config,
)

__version__ = "1.0.0"
__all__ = [
    # Data structures
    "PlanSpec",
    "Evidence",
    "Blackboard",
    "WatchConfig",
    "SpatialTokenRate",

    # AAVP data structures
    "AudioEnrichmentScope",
    "AudioMode",
    "AudioEnrichment",
    "ReflectionOutput",

    # Core components
    "GeminiClient",
    "Planner",
    "Observer",
    "Reflector",
    "Controller",
    "Store",

    # Video utilities
    "VideoMetadataExtractor",
    "sha256_file",
    "get_mime_type",
    "find_compressed_video_fallback",
    "get_video_path",
    "get_video_info",
    "print_video_info",
    "validate_video_file",
    "format_duration",
    "set_metadata_source",
    "load_video_metadata_from_json",

    # Audio utilities (AAVP)
    "check_ffmpeg_audio_support",
    "extract_audio_snippet",
    "extract_audio_region",
    "generate_gap_probes",
    "cleanup_audio_artifacts",

    # Timbre boundaries + local ASR (AAVP2)
    "compute_timbre_boundaries",
    "transcribe_wav",
    "TimbreAnchor",
    "preplan_anchors",
    "filter_anchors_in_regions",
    "load_timbre_cache",
    "save_timbre_cache",

    # Prompt management
    "PromptManager",
    "parse_json_response",
    "validate_against_schema",
    "PLAN_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FINAL_ANSWER_SCHEMA",
    "AUDIO_ENRICHMENT_SCHEMA",
    "REFLECTION_SCHEMA",

    # Configuration
    "AVPConfig",
    "load_config",
]
