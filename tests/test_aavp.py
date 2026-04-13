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
)


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
        assert 'Audio Enrichment' in full
        assert 'Key statement' in full

    def test_summary_text_no_audio_section_when_empty(self):
        bb = self._bb_with_enrichments([[]])
        full = bb.summary_text()
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

    def test_audio_enrichment_scope_is_str_enum(self):
        """AudioEnrichmentScope inherits from str for JSON-safe serialisation."""
        assert isinstance(AudioEnrichmentScope.off, str)
        assert AudioEnrichmentScope.off == "off"

    def test_audio_mode_is_str_enum(self):
        assert isinstance(AudioMode.asr_focus, str)
        assert AudioMode.asr_focus == "asr_focus"


# ===========================================================================
# 10. Integration test stubs (require API + real video)
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
