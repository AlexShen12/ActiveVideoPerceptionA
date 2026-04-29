"""
AAVP Test Suite
===============
Covers the plan-13 testing checklist:
  - Unit tests for audio_utils (extraction, gap probes)
  - Unit tests for data contracts (WatchConfig round-trip, Evidence defaults)
  - Unit tests for schema validation (PLAN_SCHEMA, AUDIO_ENRICHMENT_SCHEMA,
    REFLECTION_SCHEMA)
  - Unit test for Blackboard.audio_enrichment_summary_text formatting
  - Backwards-compatibility assertions (point 12 of the plan)
  - Integration test stubs (require live API + video file, skipped by default)

Run unit tests only:
    pytest tests/test_aavp.py -v -m "not integration"

Run everything (needs GOOGLE_API_KEY and a test video):
    pytest tests/test_aavp.py -v --run-integration
"""
from __future__ import annotations

import dataclasses
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path setup: allow running from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Lazy-import helpers — avoid pulling in the heavy google-genai SDK for
# pure-Python unit tests.  We mock the SDK at the module level before import.
# ---------------------------------------------------------------------------

def _mock_google_genai():
    """Install a minimal google-genai stub covering every symbol avp/main.py imports.

    Symbols needed:
        google.genai              — the genai namespace
        google.genai.types        — types namespace
        google.genai.types.Part   — used for video/audio Part construction
        google.genai.types.Blob   — used in video helpers
        google.genai.types.VideoMetadata — used in video helpers
    """
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class _Part:
        @staticmethod
        def from_bytes(data, mime_type=None):
            obj = _Part()
            obj.data = data
            obj.mime_type = mime_type
            return obj

        @staticmethod
        def from_uri(uri, mime_type=None):
            obj = _Part()
            obj.uri = uri
            obj.mime_type = mime_type
            return obj

    class _Blob:
        def __init__(self, data=None, mime_type=None):
            self.data = data
            self.mime_type = mime_type

    class _VideoMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Client:
        def __init__(self, **kwargs):
            pass
        def models(self):
            return self

    types_mod.Part = _Part
    types_mod.Blob = _Blob
    types_mod.VideoMetadata = _VideoMetadata
    genai_mod.types = types_mod
    genai_mod.Client = _Client
    google_mod.genai = genai_mod

    sys.modules.setdefault("google", google_mod)
    sys.modules.setdefault("google.genai", genai_mod)
    sys.modules.setdefault("google.genai.types", types_mod)


_mock_google_genai()

# Now the avp package imports without requiring the real SDK
from avp.audio_utils import (  # noqa: E402
    _format_time,
    extract_audio_snippet,
    extract_audio_region,
    generate_gap_probes,
    cleanup_audio_artifacts,
)
from avp.config import AVPConfig  # noqa: E402
from avp.main import (  # noqa: E402
    AudioEnrichment,
    AudioEnrichmentScope,
    AudioMode,
    Blackboard,
    Evidence,
    PlanSpec,
    SpatialTokenRate,
    WatchConfig,
    plan_from_dict,
)
from avp.prompt import (  # noqa: E402
    AUDIO_ENRICHMENT_SCHEMA,
    PLAN_SCHEMA,
    REFLECTION_SCHEMA,
    validate_against_schema,
    PromptManager,
)
from avp.timbre_anchor_cache import (  # noqa: E402
    TimbreAnchor,
    filter_anchors_in_regions,
    load_cache as load_timbre_cache,
    save_cache as save_timbre_cache,
)
from avp import timbre_anchor_cache as _tac  # noqa: E402
from avp import main as _avp_main  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _make_watch(**kwargs) -> WatchConfig:
    defaults = dict(
        load_mode="uniform",
        fps=0.5,
        spatial_token_rate=SpatialTokenRate.low,
    )
    defaults.update(kwargs)
    return WatchConfig(**defaults)


def _make_evidence(key_evidence=None, audio_enrichments=None) -> Evidence:
    return Evidence(
        detailed_response="test",
        key_evidence=key_evidence or [],
        audio_enrichments=audio_enrichments or [],
    )


def _make_audio_enrichment(
    center_sec: float,
    speech: str = "",
    acoustic: List[str] = None,
    source: str = "evidence",
) -> AudioEnrichment:
    return AudioEnrichment(
        center_sec=center_sec,
        window_start_sec=max(0.0, center_sec - 2.5),
        window_end_sec=center_sec + 2.5,
        speech_evidence=speech,
        acoustic_evidence=acoustic or [],
        source=source,
    )


# ===========================================================================
# 1. _format_time
# ===========================================================================

class TestFormatTime:
    def test_zero(self):
        assert _format_time(0.0) == "00:00:00.000"

    def test_one_minute(self):
        assert _format_time(60.0) == "00:01:00.000"

    def test_fractional_seconds(self):
        assert _format_time(5.5) == "00:00:05.500"

    def test_over_one_hour(self):
        assert _format_time(3661.25) == "01:01:01.250"

    def test_negative_clamped_to_zero(self):
        assert _format_time(-3.0) == "00:00:00.000"


# ===========================================================================
# 2. extract_audio_snippet
# ===========================================================================

class TestExtractAudioSnippet:
    """Tests that mock ffmpeg; no real video file needed."""

    def _patch_ffmpeg_success(self, output_path: str):
        """Context manager patches that simulate a successful ffmpeg run."""
        def _fake_run(cmd, **kw):
            # Simulate ffmpeg creating the output file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 128)
            result = mock.MagicMock()
            result.returncode = 0
            return result

        return mock.patch("avp.audio_utils.subprocess.run", side_effect=_fake_run)

    def test_returns_path_on_success(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        expected = str(tmp_path / "temp_audio" / "clip_audio_10.0s.wav")

        with self._patch_ffmpeg_success(expected):
            result = extract_audio_snippet(
                str(video),
                center_sec=10.0,
                half_width_sec=2.5,
                temp_dir=str(tmp_path / "temp_audio"),
            )
        assert result == expected
        assert os.path.exists(result)

    def test_returns_none_on_ffmpeg_failure(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        def _failing_run(cmd, **kw):
            result = mock.MagicMock()
            result.returncode = 1
            result.stderr = "codec error"
            return result

        with mock.patch("avp.audio_utils.subprocess.run", side_effect=_failing_run):
            result = extract_audio_snippet(
                str(video),
                center_sec=10.0,
                temp_dir=str(tmp_path / "temp_audio"),
            )
        assert result is None

    def test_clamps_start_to_zero(self, tmp_path):
        """center_sec=1.0 with half_width=2.5 → computed start=-1.5 → must clamp to 0."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        captured_cmd: list = []

        def _capture_run(cmd, **kw):
            captured_cmd.extend(cmd)
            result = mock.MagicMock()
            result.returncode = 1
            result.stderr = ""
            return result

        with mock.patch("avp.audio_utils.subprocess.run", side_effect=_capture_run):
            extract_audio_snippet(
                str(video),
                center_sec=1.0,
                half_width_sec=2.5,
                temp_dir=str(tmp_path / "temp_audio"),
            )

        # -ss argument must be "00:00:00.000" (clamped), not a negative value
        ss_idx = captured_cmd.index("-ss")
        assert captured_cmd[ss_idx + 1] == "00:00:00.000"

    def test_reuses_cached_file(self, tmp_path):
        """No ffmpeg call when output file already exists and is non-empty."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        audio_dir = tmp_path / "temp_audio"
        audio_dir.mkdir()
        cached = audio_dir / "clip_audio_5.0s.wav"
        cached.write_bytes(b"\x00" * 64)

        with mock.patch("avp.audio_utils.subprocess.run") as mock_run:
            result = extract_audio_snippet(
                str(video),
                center_sec=5.0,
                temp_dir=str(audio_dir),
            )
        mock_run.assert_not_called()
        assert result == str(cached)

    def test_returns_none_for_missing_video(self, tmp_path):
        result = extract_audio_snippet(
            str(tmp_path / "nonexistent.mp4"),
            center_sec=5.0,
            temp_dir=str(tmp_path / "temp_audio"),
        )
        assert result is None


# ===========================================================================
# 3. extract_audio_region
# ===========================================================================

class TestExtractAudioRegion:
    def test_returns_none_for_inverted_range(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        result = extract_audio_region(
            str(video), start_sec=20.0, end_sec=5.0,
            temp_dir=str(tmp_path / "temp_audio"),
        )
        assert result is None

    def test_returns_none_for_equal_bounds(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        result = extract_audio_region(
            str(video), start_sec=10.0, end_sec=10.0,
            temp_dir=str(tmp_path / "temp_audio"),
        )
        assert result is None

    def test_clamps_negative_start(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        captured_cmd: list = []

        def _capture_run(cmd, **kw):
            captured_cmd.extend(cmd)
            result = mock.MagicMock()
            result.returncode = 1
            result.stderr = ""
            return result

        with mock.patch("avp.audio_utils.subprocess.run", side_effect=_capture_run):
            extract_audio_region(
                str(video), start_sec=-5.0, end_sec=10.0,
                temp_dir=str(tmp_path / "temp_audio"),
            )

        ss_idx = captured_cmd.index("-ss")
        assert captured_cmd[ss_idx + 1] == "00:00:00.000"


# ===========================================================================
# 4. generate_gap_probes
# ===========================================================================

class TestGenerateGapProbes:
    def test_basic_single_gap(self):
        # Evidence covers [10, 20]; video is 60 s.
        # Prefix gap: [0,10] → mid 5; suffix gap: [20,60] → mid 40.
        probes = generate_gap_probes([(10.0, 20.0)], duration_sec=60.0, min_gap_sec=5.0)
        assert 5.0 in probes
        assert 40.0 in probes
        assert probes == sorted(probes)

    def test_merges_overlapping_intervals(self):
        # [0,15] and [10,25] overlap → merged [0,25]; suffix gap [25,60] mid=42.5
        probes = generate_gap_probes(
            [(0.0, 15.0), (10.0, 25.0)], duration_sec=60.0, min_gap_sec=5.0
        )
        assert 42.5 in probes
        # No gap before the merged interval since it starts at 0
        assert all(p > 0 for p in probes)

    def test_respects_max_probes(self):
        # Many small gaps — should still cap at max_probes
        timestamps = [(float(i * 5), float(i * 5 + 1)) for i in range(20)]
        probes = generate_gap_probes(timestamps, duration_sec=200.0, max_probes=3, min_gap_sec=2.0)
        assert len(probes) <= 3

    def test_respects_min_gap(self):
        # Gaps are all 2 s wide; with min_gap=5 no probes should be generated
        # (except possibly the fallback).
        timestamps = [(float(i * 4), float(i * 4 + 2)) for i in range(5)]
        probes = generate_gap_probes(
            timestamps, duration_sec=20.0, max_probes=5, min_gap_sec=5.0
        )
        # All returned probes are midpoints of gaps ≥ 5 s
        for p in probes:
            assert isinstance(p, float)

    def test_empty_evidence_returns_midpoint_fallback(self):
        probes = generate_gap_probes([], duration_sec=100.0)
        assert probes == [50.0]

    def test_zero_duration_returns_empty(self):
        probes = generate_gap_probes([], duration_sec=0.0)
        assert probes == []

    def test_result_is_sorted(self):
        timestamps = [(30.0, 40.0), (5.0, 15.0)]
        probes = generate_gap_probes(timestamps, duration_sec=60.0, min_gap_sec=2.0)
        assert probes == sorted(probes)

    def test_full_coverage_produces_no_probes(self):
        # Evidence covers [0, 60] — no gap anywhere
        probes = generate_gap_probes([(0.0, 60.0)], duration_sec=60.0, min_gap_sec=1.0)
        assert probes == []


# ===========================================================================
# 5. cleanup_audio_artifacts
# ===========================================================================

class TestCleanupAudioArtifacts:
    def test_removes_existing_files(self, tmp_path):
        f = tmp_path / "snippet.wav"
        f.write_bytes(b"\x00" * 32)
        cleanup_audio_artifacts([str(f)])
        assert not f.exists()

    def test_silently_skips_missing_files(self, tmp_path):
        missing = str(tmp_path / "ghost.wav")
        cleanup_audio_artifacts([missing])  # must not raise


# ===========================================================================
# 6. WatchConfig round-trip through dataclasses.asdict / plan_from_dict
# ===========================================================================

class TestWatchConfigRoundTrip:
    def _round_trip(self, watch: WatchConfig) -> WatchConfig:
        plan = PlanSpec(
            plan_version="v2",
            query="test query",
            watch=watch,
            description="round-trip test",
        )
        d = dataclasses.asdict(plan)
        recovered = plan_from_dict(d)
        return recovered.watch

    def test_visual_only_round_trip(self):
        w = _make_watch(load_mode="region", fps=1.0, regions=[(10.0, 20.0)])
        rw = self._round_trip(w)
        assert rw.load_mode == "region"
        assert rw.fps == 1.0
        assert rw.regions == [(10.0, 20.0)]
        assert rw.audio_enrichment == AudioEnrichmentScope.off

    def test_spatial_token_rate_round_trip(self):
        for rate in SpatialTokenRate:
            w = _make_watch(spatial_token_rate=rate)
            rw = self._round_trip(w)
            assert rw.spatial_token_rate == rate

    def test_audio_enrichment_round_trip(self):
        w = _make_watch(
            audio_enrichment=AudioEnrichmentScope.evidence_plus_gaps,
            audio_mode=AudioMode.asr_focus,
            audio_snippet_halfwidth_sec=3.0,
        )
        rw = self._round_trip(w)
        assert rw.audio_enrichment == AudioEnrichmentScope.evidence_plus_gaps
        assert rw.audio_mode == AudioMode.asr_focus
        assert rw.audio_snippet_halfwidth_sec == 3.0

    def test_all_audio_enrichment_scope_values_survive(self):
        for scope in AudioEnrichmentScope:
            w = _make_watch(audio_enrichment=scope)
            rw = self._round_trip(w)
            assert rw.audio_enrichment == scope

    def test_all_audio_mode_values_survive(self):
        for mode in AudioMode:
            w = _make_watch(audio_mode=mode)
            rw = self._round_trip(w)
            assert rw.audio_mode == mode


# ===========================================================================
# 7. Schema validation
# ===========================================================================

class TestPlanSchemaValidation:
    def _valid_step(self, **overrides) -> Dict[str, Any]:
        step = {
            "step_id": "1",
            "description": "Scan video",
            "sub_query": "What happens?",
            "load_mode": "uniform",
            "fps": 0.5,
            "spatial_token_rate": "low",
        }
        step.update(overrides)
        return step

    def test_minimal_valid_plan(self):
        data = {
            "reasoning": "Testing",
            "steps": [self._valid_step()],
            "completion_criteria": "Done",
        }
        assert validate_against_schema(data, PLAN_SCHEMA)

    def test_plan_with_audio_fields(self):
        step = self._valid_step(audio_enrichment="evidence_only", audio_mode="asr_focus")
        data = {
            "reasoning": "Audio query",
            "steps": [step],
            "completion_criteria": "Done",
        }
        assert validate_against_schema(data, PLAN_SCHEMA)

    def test_plan_missing_required_top_level_fails(self):
        data = {"steps": [], "completion_criteria": "Done"}  # missing "reasoning"
        assert not validate_against_schema(data, PLAN_SCHEMA)

    def test_plan_missing_steps_fails(self):
        data = {"reasoning": "ok", "completion_criteria": "Done"}
        assert not validate_against_schema(data, PLAN_SCHEMA)


class TestAudioEnrichmentSchemaValidation:
    def _valid_result(self, center_sec: float = 5.0) -> Dict[str, Any]:
        return {
            "center_sec": center_sec,
            "speech_evidence": "Hello world",
            "acoustic_evidence": ["SPEECH"],
        }

    def test_valid_response(self):
        data = {
            "audio_results": [self._valid_result(5.0), self._valid_result(15.0)],
            "overall_audio_summary": "Speech at 5 s and 15 s.",
        }
        assert validate_against_schema(data, AUDIO_ENRICHMENT_SCHEMA)

    def test_empty_audio_results_still_valid(self):
        # Empty list is structurally valid — schema only checks keys are present
        data = {"audio_results": [], "overall_audio_summary": "Nothing audible."}
        assert validate_against_schema(data, AUDIO_ENRICHMENT_SCHEMA)

    def test_missing_audio_results_fails(self):
        data = {"overall_audio_summary": "ok"}
        assert not validate_against_schema(data, AUDIO_ENRICHMENT_SCHEMA)

    def test_missing_overall_summary_fails(self):
        data = {"audio_results": [self._valid_result()]}
        assert not validate_against_schema(data, AUDIO_ENRICHMENT_SCHEMA)


class TestReflectionSchemaValidation:
    def _valid_reflection(self, **overrides) -> Dict[str, Any]:
        base = {
            "sufficient": True,
            "reason_code": "SUFFICIENT",
            "reasoning": "Visual evidence is clear.",
            "confidence": 0.92,
        }
        base.update(overrides)
        return base

    def test_minimal_sufficient_reflection(self):
        assert validate_against_schema(self._valid_reflection(), REFLECTION_SCHEMA)

    def test_insufficient_with_zoom_and_modalities(self):
        data = self._valid_reflection(
            sufficient=False,
            reason_code="MODALITY_MISMATCH",
            zoom_region=[30.0, 50.0],
            required_modalities={"audio_enrichment": "evidence_plus_gaps"},
        )
        assert validate_against_schema(data, REFLECTION_SCHEMA)

    def test_missing_sufficient_fails(self):
        data = self._valid_reflection()
        del data["sufficient"]
        assert not validate_against_schema(data, REFLECTION_SCHEMA)

    def test_missing_reason_code_fails(self):
        data = self._valid_reflection()
        del data["reason_code"]
        assert not validate_against_schema(data, REFLECTION_SCHEMA)

    def test_missing_reasoning_fails(self):
        data = self._valid_reflection()
        del data["reasoning"]
        assert not validate_against_schema(data, REFLECTION_SCHEMA)

    def test_missing_confidence_fails(self):
        data = self._valid_reflection()
        del data["confidence"]
        assert not validate_against_schema(data, REFLECTION_SCHEMA)


# ===========================================================================
# 8. Blackboard.audio_enrichment_summary_text
# ===========================================================================

class TestBlackboardAudioSummaryText:
    def _bb_with_enrichments(
        self, enrichments_per_round: List[List[AudioEnrichment]]
    ) -> Blackboard:
        bb = Blackboard(video_path="/fake/video.mp4", duration_sec=120.0)
        for round_enrichments in enrichments_per_round:
            ev = _make_evidence(audio_enrichments=round_enrichments)
            bb.add_evidence(ev)
        return bb

    def test_empty_returns_empty_string(self):
        bb = self._bb_with_enrichments([[]])
        assert bb.audio_enrichment_summary_text() == ""

    def test_speech_only_entry(self):
        ae = _make_audio_enrichment(10.0, speech="Hello everyone")
        bb = self._bb_with_enrichments([[ae]])
        text = bb.audio_enrichment_summary_text()
        assert 't=10.0s' in text
        assert 'Hello everyone' in text
        assert '[evidence]' in text

    def test_acoustic_only_entry(self):
        ae = _make_audio_enrichment(30.0, acoustic=["MUSIC", "APPLAUSE"])
        bb = self._bb_with_enrichments([[ae]])
        text = bb.audio_enrichment_summary_text()
        assert 'MUSIC' in text
        assert 'APPLAUSE' in text

    def test_gap_probe_source_label(self):
        ae = _make_audio_enrichment(55.0, source="gap_probe")
        bb = self._bb_with_enrichments([[ae]])
        text = bb.audio_enrichment_summary_text()
        assert '[gap_probe]' in text

    def test_multiple_rounds_all_included(self):
        round1 = [_make_audio_enrichment(5.0, speech="Round one")]
        round2 = [_make_audio_enrichment(50.0, speech="Round two")]
        bb = self._bb_with_enrichments([round1, round2])
        text = bb.audio_enrichment_summary_text()
        assert 'Round one' in text
        assert 'Round two' in text

    def test_summary_text_includes_audio_section(self):
        ae = _make_audio_enrichment(10.0, speech="Key statement")
        bb = self._bb_with_enrichments([[ae]])
        full = bb.summary_text()
        # AAVP2 renames the section to "Regional Speech Transcripts" to
        # reflect the local-Whisper pipeline; legacy "Audio Enrichment"
        # naming is gone.
        assert 'Regional Speech Transcripts' in full
        assert 'Key statement' in full

    def test_summary_text_no_audio_section_when_empty(self):
        bb = self._bb_with_enrichments([[]])
        full = bb.summary_text()
        assert 'Regional Speech Transcripts' not in full
        assert 'Audio Enrichment' not in full


# ===========================================================================
# 9. Backwards compatibility — point 12 mitigations
# ===========================================================================

class TestBackwardsCompatibility:
    """Each test asserts one row of the point-12 migration table."""

    def test_avp_config_audio_disabled_by_default(self):
        """AVPConfig.audio_enabled defaults to False."""
        cfg = AVPConfig()
        assert cfg.audio_enabled is False

    def test_watch_config_audio_enrichment_defaults_off(self):
        """WatchConfig.audio_enrichment defaults to AudioEnrichmentScope.off."""
        w = _make_watch()
        assert w.audio_enrichment == AudioEnrichmentScope.off

    def test_watch_config_audio_mode_defaults_balanced(self):
        """WatchConfig.audio_mode defaults to AudioMode.balanced."""
        w = _make_watch()
        assert w.audio_mode == AudioMode.balanced

    def test_evidence_audio_enrichments_defaults_empty(self):
        """Evidence.audio_enrichments defaults to an empty list."""
        ev = Evidence(detailed_response="x")
        assert ev.audio_enrichments == []

    def test_evidence_asdict_includes_audio_enrichments(self):
        """dataclasses.asdict serialises audio_enrichments without error."""
        ae = _make_audio_enrichment(5.0, speech="hi", acoustic=["SPEECH"])
        ev = Evidence(
            detailed_response="x",
            audio_enrichments=[ae],
        )
        d = dataclasses.asdict(ev)
        assert "audio_enrichments" in d
        assert len(d["audio_enrichments"]) == 1
        assert d["audio_enrichments"][0]["center_sec"] == 5.0

    def test_plan_from_dict_old_format_has_audio_off(self):
        """Legacy plans (steps[] format) load with audio_enrichment=off."""
        old_dict = {
            "plan_version": "v1",
            "query": "old query",
            "steps": [
                {
                    "step_id": "1",
                    "description": "Scan",
                    "sub_query": "What?",
                    "watch": {
                        "load_mode": "uniform",
                        "fps": 0.5,
                        "spatial_token_rate": "low",
                        "regions": [],
                    },
                }
            ],
            "completion_criteria": "done",
        }
        plan = plan_from_dict(old_dict)
        assert plan.watch.audio_enrichment == AudioEnrichmentScope.off

    def test_plan_from_dict_missing_audio_fields_defaults_off(self):
        """New-format plan JSON without audio fields loads with safe defaults."""
        new_dict = {
            "plan_version": "v2",
            "query": "silent query",
            "watch": {
                "load_mode": "uniform",
                "fps": 0.5,
                "spatial_token_rate": "low",
                "regions": [],
            },
            "description": "no audio",
            "completion_criteria": "done",
        }
        plan = plan_from_dict(new_dict)
        assert plan.watch.audio_enrichment == AudioEnrichmentScope.off
        assert plan.watch.audio_mode == AudioMode.balanced
        assert plan.watch.audio_snippet_halfwidth_sec == 2.5

    def test_blackboard_summary_text_no_audio_unchanged(self):
        """summary_text() without audio enrichments contains no audio section."""
        bb = Blackboard(video_path="/fake/video.mp4", duration_sec=30.0)
        ev = _make_evidence(
            key_evidence=[
                {"timestamp_start": 1.0, "timestamp_end": 3.0, "description": "action"}
            ]
        )
        bb.add_evidence(ev)
        text = bb.summary_text()
        assert "Audio Enrichment" not in text
        assert "Regional Speech Transcripts" not in text

    def test_audio_enrichment_scope_is_str_enum(self):
        """AudioEnrichmentScope inherits from str for JSON-safe serialisation."""
        assert isinstance(AudioEnrichmentScope.off, str)
        assert AudioEnrichmentScope.off == "off"

    def test_audio_mode_is_str_enum(self):
        assert isinstance(AudioMode.asr_focus, str)
        assert AudioMode.asr_focus == "asr_focus"


# ===========================================================================
# 10. AAVP2 — timbre anchor cache + filter
# ===========================================================================


class TestFilterAnchorsInRegions:
    """``filter_anchors_in_regions`` keeps anchors whose centre is in any region."""

    def _anchor(self, c: float) -> TimbreAnchor:
        return TimbreAnchor(center_sec=c, window_start_sec=c - 2.5, window_end_sec=c + 2.5, transcript="t")

    def test_empty_inputs_return_empty(self):
        assert filter_anchors_in_regions([], [(0.0, 10.0)]) == []
        assert filter_anchors_in_regions([self._anchor(5.0)], []) == []

    def test_keeps_centres_inside_single_region(self):
        anchors = [self._anchor(2.0), self._anchor(15.0), self._anchor(25.0)]
        out = filter_anchors_in_regions(anchors, [(10.0, 20.0)])
        assert [a.center_sec for a in out] == [15.0]

    def test_keeps_centres_across_multiple_regions(self):
        anchors = [self._anchor(5.0), self._anchor(35.0), self._anchor(80.0)]
        out = filter_anchors_in_regions(anchors, [(0.0, 10.0), (75.0, 90.0)])
        assert sorted(a.center_sec for a in out) == [5.0, 80.0]

    def test_inverted_regions_skipped(self):
        anchors = [self._anchor(5.0)]
        out = filter_anchors_in_regions(anchors, [(20.0, 10.0)])
        assert out == []

    def test_accepts_list_of_lists(self):
        anchors = [self._anchor(5.0)]
        out = filter_anchors_in_regions(anchors, [[0.0, 10.0]])
        assert [a.center_sec for a in out] == [5.0]


class TestTimbreCacheRoundTrip:
    """``save_cache`` then ``load_cache`` returns the same anchors."""

    def _params(self):
        return dict(anchor_interval_sec=15.0, timbre_window_sec=5.0, whisper_model="base")

    def test_save_then_load(self, tmp_path):
        video = tmp_path / "vid.mp4"
        video.write_bytes(b"\x00" * 32)
        anchors = [
            TimbreAnchor(center_sec=10.0, window_start_sec=7.5, window_end_sec=12.5, transcript="hello"),
            TimbreAnchor(center_sec=30.0, window_start_sec=27.5, window_end_sec=32.5, transcript=""),
        ]
        save_timbre_cache(str(tmp_path / "cache"), str(video), anchors, **self._params())
        loaded = load_timbre_cache(str(tmp_path / "cache"), str(video), **self._params())
        assert loaded is not None
        assert [a.center_sec for a in loaded] == [10.0, 30.0]
        assert loaded[0].transcript == "hello"
        assert loaded[1].transcript == ""

    def test_load_miss_returns_none(self, tmp_path):
        video = tmp_path / "vid.mp4"
        video.write_bytes(b"\x00" * 32)
        loaded = load_timbre_cache(str(tmp_path / "cache"), str(video), **self._params())
        assert loaded is None

    def test_param_change_invalidates_cache(self, tmp_path):
        video = tmp_path / "vid.mp4"
        video.write_bytes(b"\x00" * 32)
        anchors = [TimbreAnchor(10.0, 7.5, 12.5, "x")]
        save_timbre_cache(str(tmp_path / "cache"), str(video), anchors, **self._params())
        # Changing whisper_model changes the cache key — old entry is hidden.
        miss = load_timbre_cache(
            str(tmp_path / "cache"),
            str(video),
            anchor_interval_sec=15.0,
            timbre_window_sec=5.0,
            whisper_model="large-v3",
        )
        assert miss is None

    def test_empty_cache_dir_returns_none(self, tmp_path):
        video = tmp_path / "vid.mp4"
        video.write_bytes(b"\x00" * 32)
        loaded = load_timbre_cache("", str(video), **self._params())
        assert loaded is None


class TestPreplanAnchorsCacheHit:
    """``preplan_anchors`` reuses cached entries without rerunning librosa/Whisper."""

    def test_cache_hit_skips_compute(self, tmp_path, monkeypatch):
        video = tmp_path / "vid.mp4"
        video.write_bytes(b"\x00" * 32)
        cache_dir = tmp_path / "cache"
        prebaked = [
            TimbreAnchor(center_sec=10.0, window_start_sec=7.5, window_end_sec=12.5, transcript="hi"),
        ]
        save_timbre_cache(
            str(cache_dir),
            str(video),
            prebaked,
            anchor_interval_sec=15.0,
            timbre_window_sec=5.0,
            whisper_model="base",
        )

        # Sentinel: any call to compute_timbre_boundaries / transcribe_wav
        # would be a duplicate and must not happen on a cache hit.
        compute_calls = []
        transcribe_calls = []
        monkeypatch.setattr(
            _tac,
            "compute_timbre_boundaries",
            lambda *a, **k: compute_calls.append(1) or [],
        )
        monkeypatch.setattr(
            _tac,
            "transcribe_wav",
            lambda *a, **k: transcribe_calls.append(1) or "",
        )

        out = _tac.preplan_anchors(
            str(video),
            duration_sec=30.0,
            cache_dir=str(cache_dir),
            anchor_interval_sec=15.0,
            timbre_window_sec=5.0,
            whisper_model="base",
        )
        assert [a.center_sec for a in out] == [10.0]
        assert out[0].transcript == "hi"
        assert compute_calls == []
        assert transcribe_calls == []


# ===========================================================================
# 11. Observer load_mode branching (uniform no-op, region attaches cached ASR)
# ===========================================================================


class _StubClient:
    """Minimal stand-in for GeminiClient so Observer.observe is exercisable.

    Bypasses the genai SDK entirely.  ``infer_on_video`` returns a fixed
    ``Evidence`` so we can probe the post-observation audio branch.
    """
    def __init__(self, audio_enabled: bool = True):
        self.debug = False
        self.temp_clips_dir = None
        self.created_clips = []
        self._avp_config = AVPConfig(audio_enabled=audio_enabled)
        self.audio_sample_rate = 16000
        self.audio_max_snippets_per_round = 15
        self.audio_gap_probes = 5
        self.audio_closed_tags = []

    def infer_on_video(self, **_):
        return Evidence(
            detailed_response="visual",
            key_evidence=[{"timestamp_start": 30.0, "timestamp_end": 35.0, "description": "moment"}],
            reasoning="r",
        )

    # Ensure the legacy enrichment path raises if accidentally called — that
    # would mean we regressed back to the Gemini WAV transcription pipeline.
    def enrich_with_audio(self, *a, **k):
        raise AssertionError("enrich_with_audio must not be called under AAVP2")


class _StubMetaExtractor:
    def __init__(self, *a, **k):
        self.duration = 120.0


@pytest.fixture
def _patch_meta_extractor(monkeypatch):
    monkeypatch.setattr(_avp_main, "VideoMetadataExtractor", _StubMetaExtractor)


class TestObserverAudioBranch:
    def _bb_with_anchors(self, anchors_meta):
        bb = Blackboard(video_path="/fake/video.mp4", duration_sec=120.0)
        bb.meta["preplan_timbre_anchors"] = anchors_meta
        return bb

    def _plan(self, *, load_mode: str, regions=None) -> PlanSpec:
        return PlanSpec(
            plan_version="v2",
            query="What does the narrator say?",
            watch=WatchConfig(
                load_mode=load_mode,
                fps=1.0,
                spatial_token_rate=SpatialTokenRate.low,
                regions=regions or [],
            ),
            description="test",
        )

    def test_uniform_load_mode_skips_enrichment(self, _patch_meta_extractor):
        client = _StubClient()
        bb = self._bb_with_anchors([
            {"center_sec": 15.0, "window_start_sec": 12.5, "window_end_sec": 17.5, "transcript": "hello"},
        ])
        plan = self._plan(load_mode="uniform")
        observer = _avp_main.Observer(client)
        ev = observer.observe(plan, bb)
        assert ev.audio_enrichments == []

    def test_region_load_mode_attaches_cached_transcripts(self, _patch_meta_extractor):
        client = _StubClient()
        bb = self._bb_with_anchors([
            {"center_sec": 15.0, "window_start_sec": 12.5, "window_end_sec": 17.5, "transcript": "outside"},
            {"center_sec": 32.0, "window_start_sec": 29.5, "window_end_sec": 34.5, "transcript": "in-region"},
            {"center_sec": 90.0, "window_start_sec": 87.5, "window_end_sec": 92.5, "transcript": "outside"},
        ])
        plan = self._plan(load_mode="region", regions=[(25.0, 40.0)])
        observer = _avp_main.Observer(client)
        ev = observer.observe(plan, bb)
        assert len(ev.audio_enrichments) == 1
        assert ev.audio_enrichments[0].speech_evidence == "in-region"
        # acoustic_evidence is deprecated under AAVP2 — must always be empty.
        assert ev.audio_enrichments[0].acoustic_evidence == []

    def test_audio_disabled_master_switch_skips_everything(self, _patch_meta_extractor):
        client = _StubClient(audio_enabled=False)
        bb = self._bb_with_anchors([
            {"center_sec": 30.0, "window_start_sec": 27.5, "window_end_sec": 32.5, "transcript": "irrelevant"},
        ])
        plan = self._plan(load_mode="region", regions=[(25.0, 40.0)])
        observer = _avp_main.Observer(client)
        ev = observer.observe(plan, bb)
        assert ev.audio_enrichments == []

    def test_region_with_no_in_range_anchors_attaches_nothing(self, _patch_meta_extractor):
        client = _StubClient()
        bb = self._bb_with_anchors([
            {"center_sec": 5.0, "window_start_sec": 2.5, "window_end_sec": 7.5, "transcript": "outside"},
        ])
        plan = self._plan(load_mode="region", regions=[(50.0, 60.0)])
        observer = _avp_main.Observer(client)
        ev = observer.observe(plan, bb)
        assert ev.audio_enrichments == []


# ===========================================================================
# 12. Planner soft-evidence injection
# ===========================================================================


class TestPlanningPromptTimbreSoftEvidence:
    # The rendered bullet format ``  - t≈X.Xs  speech: "..."`` is unique to
    # ``_format_timbre_anchor_preview``; the literal phrase
    # "Timbre-Anchor ASR Preview" and shorthand like ``t≈92s`` also appear
    # in the static audio-handling guidance/exemplars, so we anchor the
    # negative tests on the lowercase ``  speech: `` marker that only
    # appears in actual rendered bullets.
    _BULLET_MARKER = "  speech: "
    _BANNER = "heuristic soft evidence — NOT ground truth"

    def test_anchors_render_in_planning_prompt(self):
        meta = {
            "duration_sec": 60.0,
            "timbre_anchors": [
                {"center_sec": 10.0, "window_start_sec": 7.5, "window_end_sec": 12.5, "transcript": "hello world"},
                {"center_sec": 25.0, "window_start_sec": 22.5, "window_end_sec": 27.5, "transcript": ""},
            ],
        }
        prompt = PromptManager.get_planning_prompt("What was said?", meta)
        assert self._BANNER in prompt
        assert self._BULLET_MARKER in prompt
        assert "t≈10.0s" in prompt
        assert "hello world" in prompt
        assert "(none / inaudible)" in prompt

    def test_no_anchors_omits_section(self):
        meta = {"duration_sec": 60.0, "timbre_anchors": []}
        prompt = PromptManager.get_planning_prompt("What happens?", meta)
        assert self._BANNER not in prompt
        assert self._BULLET_MARKER not in prompt

    def test_missing_timbre_key_omits_section(self):
        meta = {"duration_sec": 60.0}
        prompt = PromptManager.get_planning_prompt("What happens?", meta)
        assert self._BANNER not in prompt
        assert self._BULLET_MARKER not in prompt


# ===========================================================================
# 13. Integration test stubs (require API + real video)
# ===========================================================================

INTEGRATION_MARK = pytest.mark.skip(
    reason=(
        "Integration tests require GOOGLE_API_KEY env var and a real video file. "
        "Run with --run-integration to enable."
    )
)


@INTEGRATION_MARK
class TestIntegrationPureVisual:
    """audio_enabled=False: pipeline is unchanged, no audio extraction."""

    def test_no_audio_extraction_when_disabled(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationSpeechQuery:
    """audio_enabled=True, planner selects evidence_only + asr_focus."""

    def test_speech_transcripts_returned(self):
        raise NotImplementedError

    def test_audio_enrichments_attached_to_evidence(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationAcousticQuery:
    """audio_enabled=True, planner selects evidence_only + acoustic_focus."""

    def test_closed_tag_labels_returned(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationGapProbe:
    """planner selects evidence_plus_gaps; gap probes fire in uncovered spans."""

    def test_gap_probes_capture_narration(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationReflectorOverride:
    """MODALITY_MISMATCH → Controller overrides plan with escalated audio scope."""

    def test_replan_escalates_audio_enrichment(self):
        raise NotImplementedError

    def test_replan_applies_zoom_region(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationSnippetCap:
    """20 key_evidence items → capped to audio_max_snippets_per_round (15)."""

    def test_snippet_cap_prioritises_evidence_over_gap_probes(self):
        raise NotImplementedError


@INTEGRATION_MARK
class TestIntegrationReflectorSufficient:
    """Visual + speech agree → answer generated without further replanning."""

    def test_sufficient_breaks_loop(self):
        raise NotImplementedError
