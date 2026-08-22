"""
stt/client.py — Sarvam AI Saaras v3-Realtime WebSocket STT Client.

Features:
- Async WebSocket connection to wss://api.sarvam.ai/speech-to-text-realtime/ws
- stream_type=fast for minimum TTFT
- mode=codemix for natural Hinglish / code-mixed Indic language handling
- Partial transcript accumulation (is_final=false) → trigger on is_final=true
- Integrated VAD: VAD emits Audio_End → triggers transcription finalization
- Exponential backoff retry on disconnects (via backoff.py)
- Base64 audio encoding for WebSocket message protocol
"""
import asyncio
import base64
import json
import time
import urllib.parse
from typing import AsyncGenerator, Callable, Awaitable
import httpx

import websockets
import websockets.exceptions
from loguru import logger

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import config
from stt.backoff import WebSocketRetryer
from stt.vad import SileroVAD, VADEvent


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TranscriptCallback = Callable[[str, str, float], Awaitable[None]]
# Called with: (transcript_text, lang_code, text_ready_timestamp)


# ---------------------------------------------------------------------------
# Sarvam WebSocket URL builder
# ---------------------------------------------------------------------------

def _build_ws_url(lang_code: str | None = None) -> str:
    """
    Build the full WebSocket URL with query parameters.

    Args:
        lang_code: BCP-47 code (e.g. "hi-IN"). Defaults to config default.
    """
    params = {
        "model":         config.SARVAM_MODEL,
        "language_code": lang_code or config.SARVAM_DEFAULT_LANG,
        "stream_type":   config.SARVAM_STREAM_TYPE,
        "mode":          config.SARVAM_MODE,
    }
    query_string = urllib.parse.urlencode(params, safe='', quote_via=lambda s, safe, encoding, errors: urllib.parse.quote(s, safe=':'))
    return f"{config.SARVAM_WS_URL}?{query_string}"


# ---------------------------------------------------------------------------
# Sarvam STT Client
# ---------------------------------------------------------------------------

class SarvamSTTClient:
    """
    Streaming STT client for Sarvam Saaras v3-realtime.

    Usage (microphone mode):
        client = SarvamSTTClient()
        client.set_transcript_callback(my_callback)

        async with client.connect(lang_code="hi-IN") as session:
            async for pcm_chunk in mic_stream():
                await session.send_audio(pcm_chunk)

    Usage (file mode):
        async with client.connect() as session:
            await session.send_audio_file("query.wav")
            result = await session.get_final_transcript()
    """

    def __init__(self):
        self._vad = SileroVAD()
        self._vad.load()
        self._transcript_callback: TranscriptCallback | None = None
        self._partial_transcript = ""
        self._final_transcript = ""
        self._text_ready_time: float | None = None
        self._audio_end_time: float | None = None
        self._current_lang: str = config.SARVAM_DEFAULT_LANG

        # Set VAD callback to record Audio_End
        self._vad.set_event_callback(self._on_vad_event)

    def set_transcript_callback(self, callback: TranscriptCallback) -> None:
        """Register callback for when a final transcript is ready."""
        self._transcript_callback = callback

    async def _on_vad_event(self, event: VADEvent, timestamp: float) -> None:
        """Handle VAD events — Audio_End drives transcription finalization."""
        if event == VADEvent.SPEECH_END:
            self._audio_end_time = timestamp
            logger.debug(f"[STT] Audio_End recorded at t={timestamp:.4f}s")

    # ------------------------------------------------------------------
    # WebSocket session context manager
    # ------------------------------------------------------------------

    def connect(
        self, lang_code: str | None = None
    ) -> "_SarvamSession":
        """
        Return an async context manager for a STT session.

        Args:
            lang_code: BCP-47 language code. None = use config default.
        """
        self._current_lang = lang_code or config.SARVAM_DEFAULT_LANG
        self._vad.reset()
        self._partial_transcript = ""
        self._final_transcript = ""
        self._text_ready_time = None
        self._audio_end_time = None
        return _SarvamSession(self, lang_code=self._current_lang)


class _SarvamSession:
    """
    Async context manager managing the lifecycle of a single STT WebSocket session.
    Handles connection, audio streaming, response parsing, and cleanup.
    """

    def __init__(self, client: SarvamSTTClient, lang_code: str):
        self._client = client
        self._lang_code = lang_code
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._final_event = asyncio.Event()

    async def __aenter__(self) -> "_SarvamSession":
        await self._open_with_retry()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._close()

    async def _open_with_retry(self) -> None:
        """Open WebSocket with exponential backoff on failure."""
        ws_url = _build_ws_url(self._lang_code)
        headers = {"Api-Subscription-Key": config.SARVAM_API_KEY}

        retryer = WebSocketRetryer(label="sarvam_stt")
        async for attempt in retryer:
            try:
                try:
                    self._ws = await websockets.connect(
                        ws_url,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=10,
                        open_timeout=5,
                    )
                except TypeError:
                    self._ws = await websockets.connect(
                        ws_url,
                        extra_headers=headers,
                        ping_interval=20,
                        ping_timeout=10,
                        open_timeout=5,
                    )
                logger.info(
                    f"[STT] Connected to Sarvam (attempt {attempt}) — "
                    f"lang={self._lang_code}, model={config.SARVAM_MODEL}"
                )
                # Start background receiver task
                self._recv_task = asyncio.create_task(self._recv_loop())
                break
            except Exception as exc:
                await retryer.record_failure(exc)

    async def _recv_loop(self) -> None:
        """
        Background task: receive messages from Sarvam WebSocket.
        Parses partial and final transcripts.
        """
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")
                transcript = msg.get("transcript", "")
                is_final = msg.get("is_final", False) or (event == "transcript.final")

                if event == "transcript.partial" or (transcript and not is_final):
                    self._client._partial_transcript = transcript
                    logger.debug(f"[STT] Partial: '{transcript[:60]}'")

                elif is_final or event == "transcript.final" or msg.get("type") == "data":
                    if not transcript and msg.get("data"):
                        transcript = msg["data"].get("transcript", "")

                    if transcript:
                        self._client._final_transcript = transcript
                        self._client._text_ready_time = time.perf_counter()
                        logger.info(f"[STT] Final transcript: '{transcript[:80]}'")

                        # Fire transcript callback
                        if self._client._transcript_callback:
                            await self._client._transcript_callback(
                                transcript,
                                self._lang_code,
                                self._client._text_ready_time,
                            )

                        # Signal waiting coroutines
                        self._final_event.set()

        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning(f"[STT] WebSocket closed: code={exc.code}, reason={exc.reason}")
        except Exception as exc:
            logger.error(f"[STT] Receiver error: {type(exc).__name__}: {exc}")

    async def send_audio(self, pcm_chunk: bytes) -> None:
        """
        Send a PCM audio chunk to Sarvam.

        Also feeds chunk to local VAD for Audio_End detection.

        Args:
            pcm_chunk: Raw PCM bytes (16kHz, 16-bit mono).
        """
        if self._ws is None:
            raise RuntimeError("Session not connected. Use async with client.connect().")

        # 1. Feed to local VAD
        await self._client._vad.process_chunk(pcm_chunk)

        # 2. Encode and send to Sarvam
        audio_b64 = base64.b64encode(pcm_chunk).decode("utf-8")
        message = json.dumps({"event": "audio_input", "audio": audio_b64})
        try:
            await self._ws.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("[STT] send_audio: connection closed mid-stream")
            raise

    async def send_audio_file(self, wav_path: str) -> None:
        """
        Stream a WAV file to Sarvam in chunks.

        Args:
            wav_path: Path to a 16kHz, 16-bit mono WAV file.
        """
        import soundfile as sf

        data, sr = sf.read(wav_path, dtype="int16")
        pcm_bytes = data.tobytes()

        # Send in 20ms chunks (640 bytes at 16kHz 16-bit mono)
        chunk_size = config.AUDIO_CHUNK_SAMPLES * 2  # 2 bytes per sample
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            await self.send_audio(chunk)
            await asyncio.sleep(config.AUDIO_CHUNK_MS / 1000.0)

    async def get_final_transcript(self, timeout: float = 10.0) -> tuple[str, float | None]:
        """
        Wait for and return the final transcript.

        Args:
            timeout: Max seconds to wait for is_final=true response.

        Returns:
            (transcript_text, text_ready_timestamp)
        """
        try:
            await asyncio.wait_for(self._final_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[STT] Timeout waiting for final transcript")

        return (
            self._client._final_transcript,
            self._client._text_ready_time,
        )

    async def _close(self) -> None:
        """Gracefully close the WebSocket and cancel receiver task."""
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.debug("[STT] Session closed")


# ---------------------------------------------------------------------------
# Convenience: transcribe a single WAV file (for eval harness & web upload)
# ---------------------------------------------------------------------------

_stt_http_client: httpx.AsyncClient | None = None

def _get_stt_client() -> httpx.AsyncClient:
    global _stt_http_client
    if _stt_http_client is None or _stt_http_client.is_closed:
        _stt_http_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
        )
    return _stt_http_client

async def transcribe_file(
    wav_path: str,
    lang_code: str | None = None,
) -> tuple[str, float | None, float | None]:
    """
    Transcribe a WAV file using Sarvam REST STT (saarika:v2.5) with Groq Whisper fallback.
    Automatically ensures 16kHz 16-bit mono format before sending.
    Reuses persistent HTTP keep-alive connection for minimal latency.
    Returns (transcript, audio_end_time, text_ready_time).
    """
    import soundfile as sf
    import numpy as np
    from pathlib import Path

    t0 = time.perf_counter()
    is_auto = not lang_code or lang_code.lower() in ("auto", "unknown", "detect")
    short_lang = None if is_auto else lang_code.split("-")[0].split("_")[0].lower()
    bcp47 = "unknown" if is_auto else config.LANG_TO_BCP47.get(short_lang, config.LANG_TO_BCP47.get(lang_code or "", config.SARVAM_DEFAULT_LANG))

    # Preprocess Audio: Ensure 16kHz mono 16-bit PCM WAV
    try:
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # convert stereo to mono
        if sr != config.AUDIO_SAMPLE_RATE:
            import scipy.signal
            target_samples = int(len(data) * config.AUDIO_SAMPLE_RATE / sr)
            data = scipy.signal.resample(data, target_samples)
            sr = config.AUDIO_SAMPLE_RATE
        # Re-write normalized 16-bit PCM WAV
        data_int16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(wav_path, data_int16, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
    except Exception as exc:
        logger.debug(f"[STT] Audio format normalization: {exc}")

    # If AUTO mode: Option 1 - Dual-Stream Speculative STT (concurrent parallel execution)
    if is_auto:
        try:
            from groq import AsyncGroq
            import asyncio
            groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
            with open(wav_path, "rb") as f:
                audio_data = f.read()

            file_name = Path(wav_path).name

            # Launch Tamil/Hindi-Anchored stream and General stream concurrently
            async def _stream_indic():
                return await groq_client.audio.transcriptions.create(
                    file=(file_name, audio_data),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    prompt="வணக்கம், தமிழ், தமிழ்நாடு, பாரதியார், காந்தி, சென்னை, இந்தியா, முதலமைச்சர், பிறந்த மாநிலம், மனைவி பெயர், हिन्दी, भारत.",
                )

            async def _stream_general():
                return await groq_client.audio.transcriptions.create(
                    file=(file_name, audio_data),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )

            res_indic, res_general = await asyncio.gather(
                _stream_indic(), _stream_general(), return_exceptions=True
            )

            text_indic = getattr(res_indic, "text", "") if not isinstance(res_indic, Exception) else ""
            text_general = getattr(res_general, "text", "") if not isinstance(res_general, Exception) else ""

            # Check character counts to pick winning transcription
            ta_chars_indic = sum(1 for c in text_indic if 0x0B80 <= ord(c) <= 0x0BFF)
            hi_chars_indic = sum(1 for c in text_indic if 0x0900 <= ord(c) <= 0x097F)
            te_chars_gen   = sum(1 for c in text_general if 0x0C00 <= ord(c) <= 0x0C7F)

            if ta_chars_indic > 0 or hi_chars_indic > 0:
                transcript = text_indic
            elif te_chars_gen > 0:
                transcript = text_general
            else:
                transcript = text_indic if len(text_indic) >= len(text_general) else text_general

            t1 = time.perf_counter()
            if transcript.strip():
                logger.info(f"[STT:Dual-Speculative] Selected: '{transcript}' in {(t1-t0)*1000:.1f}ms")
                return transcript.strip(), t0, t1
        except Exception as e:
            logger.warning(f"[STT] Dual-stream notice: {e}. Trying Sarvam...")


    # Primary for explicit language: Sarvam REST API (saarika:v2.5)
    try:
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": config.SARVAM_API_KEY}
        data = {"model": "saarika:v2.5"}
        if bcp47 and bcp47 != "unknown":
            data["language_code"] = bcp47
        else:
            data["language_code"] = config.SARVAM_DEFAULT_LANG

        with open(wav_path, "rb") as f:
            files = {"file": (Path(wav_path).name, f.read(), "audio/wav")}

        client = _get_stt_client()
        resp = await client.post(url, headers=headers, data=data, files=files)
        if resp.status_code == 200:
            result = resp.json()
            transcript = result.get("transcript", "").strip()
            if transcript:
                t1 = time.perf_counter()
                logger.info(f"[STT:Sarvam] Transcribed: '{transcript}' in {(t1-t0)*1000:.1f}ms")
                return transcript, t0, t1
    except Exception as e:
        logger.warning(f"[STT] Sarvam REST attempt notice: {e}. Trying Whisper fallback...")

    # Fallback: Groq Whisper Large v3 Turbo with language hint
    try:
        from groq import AsyncGroq
        groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
        whisper_kwargs = {
            "model": "whisper-large-v3-turbo",
            "response_format": "json",
            "prompt": "தமிழ், हिन्दी, తెలుగు, English. Transcribe in native script.",
        }
        if short_lang:
            whisper_kwargs["language"] = short_lang

        with open(wav_path, "rb") as f:
            transcription = await groq_client.audio.transcriptions.create(
                file=(Path(wav_path).name, f.read()),
                **whisper_kwargs,
            )
            transcript = getattr(transcription, "text", "") or ""
            t1 = time.perf_counter()
            if transcript.strip():
                logger.info(f"[STT:Whisper] Transcribed: '{transcript}' in {(t1-t0)*1000:.1f}ms")
                return transcript.strip(), t0, t1
    except Exception as e:
        logger.error(f"[STT] Groq Whisper failed: {e}")



    except Exception as e:
        logger.error(f"[STT] Groq Whisper failed: {e}")

    t1 = time.perf_counter()
    return "", t0, t1
