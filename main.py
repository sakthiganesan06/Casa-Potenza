"""
main.py — Entry point for the Multilingual Voice-Enabled RAG live demo.

Modes:
  1. Microphone mode (default): captures audio from system microphone with clear push-to-talk
  2. File mode: accepts a pre-recorded .wav file path as argument

Usage:
    python main.py --lang en               # microphone mode, English
    python main.py --lang hi               # microphone mode, Hindi
    python main.py --lang ta               # microphone mode, Tamil
    python main.py --lang te               # microphone mode, Telugu
    python main.py --file query.wav        # WAV file mode
    python main.py --strategy sliding      # use sliding window retrieval
"""
import argparse
import asyncio
import json
import sys
import time
import tempfile
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))
import config
from orchestrator.pipeline import initialize_pipeline, run_pipeline
from stt.client import transcribe_file

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console(force_terminal=True, legacy_windows=False)


# ---------------------------------------------------------------------------
# Microphone Mode
# ---------------------------------------------------------------------------

async def run_microphone_demo(lang_code: str = "en", strategy: str | None = None) -> None:
    """
    Interactive microphone session for Voice RAG.
    Records clean 16kHz audio on user prompt and executes the pipeline.
    """
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    console.print(
        Panel(
            f"[bold green]🎙️ Voice RAG — Microphone Interactive Mode[/bold green]\n"
            f"Language: [cyan]{lang_code.upper()} ({config.LANG_TO_BCP47.get(lang_code, 'en-IN')})[/cyan] | "
            f"Strategy: [cyan]{strategy or config.CHUNKING_STRATEGY}[/cyan]\n"
            f"[bold white]Press Enter to start recording, speak your query, then press Enter again to search.[/bold white]\n"
            f"[dim]Press Ctrl+C to exit.[/dim]",
            border_style="green",
        )
    )

    await initialize_pipeline()
    bcp47 = config.LANG_TO_BCP47.get(lang_code, config.SARVAM_DEFAULT_LANG)

    loop = asyncio.get_event_loop()

    while True:
        try:
            # Wait for user trigger to start recording
            await loop.run_in_executor(None, input, "\n🔴 Press [Enter] to START recording...")
            
            console.print("[bold red]🎙️ Recording... Speak now! Press [Enter] when done speaking.[/bold red]")
            
            audio_buffer = []
            stop_event = asyncio.Event()

            def callback(indata, frames, time_info, status):
                audio_buffer.append(indata.copy())

            stream = sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=1024,
                callback=callback,
            )

            with stream:
                # Wait for Enter to stop
                t0_audio_start = time.perf_counter()
                await loop.run_in_executor(None, input, "")
                t0_audio_end = time.perf_counter()

            if not audio_buffer:
                console.print("[yellow]No audio recorded. Try again.[/yellow]")
                continue

            recorded_pcm = np.concatenate(audio_buffer, axis=0)
            
            # Write temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, recorded_pcm, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")

            console.print("[dim]Transcribing speech with Sarvam AI & Groq Whisper...[/dim]")
            transcript, t0, t1 = await transcribe_file(tmp_path, lang_code=bcp47)
            
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

            if not transcript or not transcript.strip():
                console.print("[red]⚠️ No speech detected or transcription was empty. Please speak closer to microphone.[/red]")
                continue

            console.print(f"\n[bold yellow]📝 Transcribed Speech:[/bold yellow] [bold white]{transcript}[/bold white]\n")

            # Run RAG Pipeline
            result = await run_pipeline(
                transcript=transcript,
                lang_code=lang_code,
                t0_audio_end=t0_audio_end,
                t1_text_ready=t1 or time.perf_counter(),
                chunking_strategy=strategy,
            )

            _print_result(result)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting Voice RAG demo.[/yellow]")
            break


# ---------------------------------------------------------------------------
# File Mode
# ---------------------------------------------------------------------------

async def run_file_demo(
    wav_path: str,
    lang_code: str = "en",
    strategy: str | None = None,
) -> dict:
    """Process a pre-recorded .wav file through the full pipeline."""
    console.print(
        Panel(
            f"[bold blue]📁 Voice RAG — File Mode[/bold blue]\n"
            f"File: [cyan]{wav_path}[/cyan]\n"
            f"Language: [cyan]{lang_code}[/cyan] | "
            f"Strategy: [cyan]{strategy or config.CHUNKING_STRATEGY}[/cyan]",
            border_style="blue",
        )
    )

    if not Path(wav_path).exists():
        logger.error(f"File not found: {wav_path}")
        sys.exit(1)

    await initialize_pipeline()
    bcp47 = config.LANG_TO_BCP47.get(lang_code, config.SARVAM_DEFAULT_LANG)

    console.print("[dim]Transcribing audio...[/dim]")
    transcript, t0, t1 = await transcribe_file(wav_path, lang_code=bcp47)

    if not transcript:
        console.print("[red]No transcript returned. Please check audio file format.[/red]")
        sys.exit(1)

    console.print(f"\n[bold yellow]📝 Transcript:[/bold yellow] [bold white]{transcript}[/bold white]\n")

    result = await run_pipeline(
        transcript=transcript,
        lang_code=lang_code,
        t0_audio_end=t0 or time.perf_counter(),
        t1_text_ready=t1 or time.perf_counter(),
        chunking_strategy=strategy,
    )

    _print_result(result)
    return result


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    """Pretty-print the RAG result with Rich."""
    latency = result.get("latency", {})
    total   = latency.get("total_ms", 0)
    budget_color = "green" if latency.get("within_budget", False) else "yellow"

    if result.get("refused"):
        console.print(Panel(
            f"[bold red]❌ REFUSED[/bold red]\n"
            f"Reason: {result.get('refusal_reason', 'unknown')}\n"
            f"[dim]The model could not find sufficient context in the 24,693 vectors to answer reliably.[/dim]",
            border_style="red",
        ))
    else:
        answer  = result.get("answer", "—")
        sources = ", ".join(result.get("sources", []))
        confidence = result.get("confidence", 0.0)
        console.print(Panel(
            f"[bold green]✅ Grounded Answer:[/bold green]\n{answer}\n\n"
            f"[dim]Sources: {sources}\n"
            f"Confidence: {confidence:.2f}[/dim]",
            border_style="green",
        ))

    console.print(
        f"[{budget_color}]⚡ Latency Breakdown:[/{budget_color}] "
        f"STT={latency.get('stt_ms', 0):.0f}ms | "
        f"Retrieval={latency.get('retrieval_ms', 0):.1f}ms | "
        f"LLM_TTFT={latency.get('llm_ttft_ms', 0):.0f}ms | "
        f"[bold {budget_color}]Total={total:.0f}ms[/bold {budget_color}]"
    )
    console.print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def _main():
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG System — Live Demo")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .wav audio file (16kHz, 16-bit mono). Default: microphone.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=list(config.LANG_TO_BCP47.keys()),
        help="Language code: en (English), hi (Hindi), ta (Tamil), te (Telugu)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["semantic", "sliding"],
        default=None,
        help="Chunking strategy (default: semantic)",
    )
    args = parser.parse_args()

    if args.file:
        await run_file_demo(args.file, lang_code=args.lang, strategy=args.strategy)
    else:
        await run_microphone_demo(lang_code=args.lang, strategy=args.strategy)


if __name__ == "__main__":
    asyncio.run(_main())
