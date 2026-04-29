"""
Local ASR via faster-whisper (replaces Gemini WAV transcription for AAVP).

Loads the Whisper model once per process (lazy singleton keyed by
model/device/compute) and exposes ``transcribe_wav`` for one-shot transcript
extraction on short clips.  Failures degrade to an empty string so the rest
of the pipeline keeps working.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

# Cached models keyed by (model, device, compute_type).  Loading large
# Whisper checkpoints can take seconds and several GB of RAM, so we never
# want to instantiate twice per process.
_MODEL_CACHE: dict[Tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


def _resolve_device(device: str) -> str:
    """Map ``"auto"`` to the best available faster-whisper device.

    Falls back to CPU when CUDA/torch is unavailable or unable to initialise.
    """
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_model(
    model: str,
    device: str,
    compute_type: str,
    *,
    debug: bool = False,
):
    """Return a cached ``WhisperModel`` instance, loading it on first use."""
    resolved_device = _resolve_device(device)
    key = (model, resolved_device, compute_type)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            if debug:
                print(f"[asr] faster-whisper not installed ({exc}); transcripts disabled")
            _MODEL_CACHE[key] = None  # negative cache so we don't retry per call
            return None

        try:
            instance = WhisperModel(
                model,
                device=resolved_device,
                compute_type=compute_type,
            )
        except Exception as exc:
            if debug:
                print(
                    f"[asr] Failed to load WhisperModel(model={model!r}, device={resolved_device!r}, "
                    f"compute_type={compute_type!r}): {exc}"
                )
            _MODEL_CACHE[key] = None
            return None

        if debug:
            print(
                f"[asr] Loaded faster-whisper model={model!r} on {resolved_device!r} "
                f"(compute_type={compute_type!r})"
            )
        _MODEL_CACHE[key] = instance
        return instance


def transcribe_wav(
    wav_path: str,
    *,
    model: str = "base",
    device: str = "auto",
    compute_type: str = "default",
    language: Optional[str] = None,
    debug: bool = False,
) -> str:
    """Transcribe a single WAV file to a plain string.

    Empty string is returned on any failure (model unavailable, decode error,
    no detected speech) so callers can use truthiness without try/except.

    Args:
        wav_path:      Path to a 16 kHz mono PCM WAV (other rates work but
                       the model resamples internally).
        model:         faster-whisper model identifier.
        device:        ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        compute_type:  faster-whisper compute type (e.g. ``"int8"``,
                       ``"float16"``, ``"default"``).
        language:      Optional BCP-47 hint; ``None`` lets Whisper auto-detect.
        debug:         Emit progress/error logs.
    """
    instance = _load_model(model, device, compute_type, debug=debug)
    if instance is None:
        return ""

    try:
        segments, _info = instance.transcribe(wav_path, language=language)
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        if debug:
            print(f"[asr] transcribe failed for {wav_path}: {exc}")
        return ""

    return text
