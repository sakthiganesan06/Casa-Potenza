"""
stt/vad.py — Silero VAD wrapper for local, sub-millisecond Voice Activity Detection.

Silero VAD is a lightweight LSTM model (~1MB) that runs entirely locally.
It detects speech vs. silence on 16kHz PCM audio with <1ms per chunk latency.

Responsibilities:
- Load Silero VAD model at startup (cached in memory)
- Process 30ms or 60ms audio frames
- Emit speech_start / speech_end events
- Record Audio_End timestamp when silence threshold is crossed
"""
import asyncio
import time
from collections import deque
from enum import Enum, auto
from typing import Callable, Awaitable

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# VAD State Machine
# ---------------------------------------------------------------------------

class VADState(Enum):
    SILENT = auto()
    SPEAKING = auto()
    TRAILING = auto()   # speech ended, waiting for silence confirmation


class VADEvent(Enum):
    SPEECH_START = auto()
    SPEECH_END = auto()      # Audio_End milestone


# ---------------------------------------------------------------------------
# Silero VAD Model Loader
# ---------------------------------------------------------------------------

_silero_model = None
_silero_utils = None


def _load_silero_model():
    """Load Silero VAD model. Called once at startup."""
    global _silero_model, _silero_utils
    if _silero_model is not None:
        return _silero_model, _silero_utils

    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
            trust_repo=True,
        )
        _silero_model = model
        _silero_utils = utils
        logger.info("Silero VAD model loaded successfully")
        return model, utils
    except Exception as exc:
        logger.warning(
            f"Could not load Silero VAD ({exc}). "
            "Falling back to energy-based VAD."
        )
        return None, None


# ---------------------------------------------------------------------------
# Energy-based fallback VAD (no torch dependency)
# ---------------------------------------------------------------------------

def _energy_vad(pcm_chunk: bytes, threshold: float = 0.01) -> float:
    """
    Simple RMS energy VAD as fallback when Silero is unavailable.

    Returns speech probability in [0, 1].
    """
    if not pcm_chunk:
        return 0.0
    samples = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    # Sigmoid-like normalization relative to threshold
    prob = min(1.0, rms / max(threshold, 1e-6))
    return prob


# ---------------------------------------------------------------------------
# Main VAD Processor
# ---------------------------------------------------------------------------

class SileroVAD:
    """
    Voice Activity Detector backed by Silero VAD with energy-based fallback.

    Usage:
        vad = SileroVAD()
        vad.load()

        async def on_event(event, timestamp):
            if event == VADEvent.SPEECH_END:
                print(f"Audio_End at {timestamp:.3f}s")

        vad.set_event_callback(on_event)

        # Feed PCM chunks (bytes, 16kHz, 16-bit mono)
        for chunk in audio_stream:
            await vad.process_chunk(chunk)
    """

    # Silero works on 30ms frames at 16kHz = 480 samples
    FRAME_SAMPLES = 480
    FRAME_BYTES   = FRAME_SAMPLES * 2  # 16-bit = 2 bytes/sample

    def __init__(self):
        self._model = None
        self._utils = None
        self._use_silero = False

        self._state = VADState.SILENT
        self._silence_counter_ms = 0.0
        self._speech_start_time: float | None = None
        self._audio_end_time: float | None = None

        # Buffer for incomplete frames
        self._pcm_buffer = bytearray()

        # Smoothing: keep last N speech probabilities
        self._prob_history: deque[float] = deque(maxlen=5)

        # Event callback: async fn(event: VADEvent, timestamp: float)
        self._event_callback: Callable[[VADEvent, float], Awaitable[None]] | None = None

    def load(self) -> None:
        """Load the VAD model. Call once at startup."""
        self._model, self._utils = _load_silero_model()
        self._use_silero = self._model is not None
        if self._use_silero:
            logger.info("VAD: Using Silero model")
        else:
            logger.info("VAD: Using energy-based fallback")

    def reset(self) -> None:
        """Reset state between queries."""
        try:
            if self._use_silero and self._model is not None:
                self._model.reset_states()
        except Exception:
            pass
        self._state = VADState.SILENT
        self._silence_counter_ms = 0.0
        self._speech_start_time = None
        self._audio_end_time = None
        self._pcm_buffer = bytearray()
        self._prob_history.clear()

    def set_event_callback(
        self, callback: Callable[[VADEvent, float], Awaitable[None]]
    ) -> None:
        """Register an async callback for VAD events."""
        self._event_callback = callback

    async def _emit(self, event: VADEvent, timestamp: float) -> None:
        """Fire the event callback if registered."""
        if self._event_callback is not None:
            await self._event_callback(event, timestamp)

    def _get_speech_prob(self, pcm_frame: bytes) -> float:
        """Get speech probability for a single PCM frame."""
        if self._use_silero and self._model is not None:
            try:
                import torch
                samples = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0
                tensor = torch.from_numpy(samples).unsqueeze(0)
                with torch.no_grad():
                    prob = self._model(tensor, config.AUDIO_SAMPLE_RATE).item()
                return float(prob)
            except Exception as exc:
                logger.debug(f"Silero inference error: {exc}, falling back to energy")
                return _energy_vad(pcm_frame)
        else:
            return _energy_vad(pcm_frame)

    async def process_chunk(self, pcm_chunk: bytes) -> None:
        """
        Process an incoming PCM audio chunk.

        Accumulates bytes into frame-sized buffers, then runs VAD per frame.
        Thread-safe with asyncio (single-threaded event loop assumed).

        Args:
            pcm_chunk: Raw PCM bytes (16kHz, 16-bit mono).
        """
        self._pcm_buffer.extend(pcm_chunk)
        now = time.perf_counter()

        # Process complete frames
        while len(self._pcm_buffer) >= self.FRAME_BYTES:
            frame = bytes(self._pcm_buffer[:self.FRAME_BYTES])
            self._pcm_buffer = self._pcm_buffer[self.FRAME_BYTES:]

            prob = self._get_speech_prob(frame)
            self._prob_history.append(prob)

            # Smooth probability over last N frames
            smoothed_prob = sum(self._prob_history) / len(self._prob_history)
            frame_duration_ms = (self.FRAME_SAMPLES / config.AUDIO_SAMPLE_RATE) * 1000.0

            await self._update_state(smoothed_prob, frame_duration_ms, now)

    async def _update_state(
        self, prob: float, frame_duration_ms: float, now: float
    ) -> None:
        """Update VAD state machine and emit events."""

        SPEECH_THRESHOLD = 0.5   # prob above this = speaking
        SILENCE_THRESHOLD = 0.35  # prob below this = silent

        if self._state == VADState.SILENT:
            if prob >= SPEECH_THRESHOLD:
                self._state = VADState.SPEAKING
                self._speech_start_time = now
                self._silence_counter_ms = 0.0
                await self._emit(VADEvent.SPEECH_START, now)
                logger.debug(f"VAD: SPEECH_START (prob={prob:.2f})")

        elif self._state == VADState.SPEAKING:
            if prob < SILENCE_THRESHOLD:
                self._state = VADState.TRAILING
                self._silence_counter_ms = frame_duration_ms
            else:
                self._silence_counter_ms = 0.0

        elif self._state == VADState.TRAILING:
            if prob >= SPEECH_THRESHOLD:
                # False alarm: resume speaking
                self._state = VADState.SPEAKING
                self._silence_counter_ms = 0.0
            else:
                self._silence_counter_ms += frame_duration_ms
                if self._silence_counter_ms >= config.VAD_SILENCE_THRESHOLD_MS:
                    # Confirmed end of speech
                    self._audio_end_time = now
                    self._state = VADState.SILENT
                    self._silence_counter_ms = 0.0
                    await self._emit(VADEvent.SPEECH_END, now)
                    logger.debug(
                        f"VAD: SPEECH_END (silence={self._silence_counter_ms:.0f}ms)"
                    )

    @property
    def audio_end_time(self) -> float | None:
        """Return the timestamp of the most recent Audio_End event."""
        return self._audio_end_time

    @property
    def is_speaking(self) -> bool:
        """True if VAD currently detects active speech."""
        return self._state in (VADState.SPEAKING, VADState.TRAILING)
