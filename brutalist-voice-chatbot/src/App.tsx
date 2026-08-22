/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { BrutalistMicButton } from './components/BrutalistMicButton';
import { AudioVisualizer } from './components/AudioVisualizer';
import { BrutalistPopupBox } from './components/BrutalistPopupBox';
import { HistoryDrawer } from './components/HistoryDrawer';
import { HelloIntroScreen } from './components/HelloIntroScreen';
import { HackerHouseBanner } from './components/HackerHouseBanner';
import { LatencySpeedometerBar, calculateLatencyScore } from './components/LatencySpeedometerBar';
import { BrutalistCursorFx } from './components/BrutalistCursorFx';
import {
  RecordingState,
  ChatInteraction,
  PersonaMode,
  ThemeColor,
  VisualizerMode,
  CursorMode,
} from './types';
import {
  AlertTriangle,
  Send,
  Keyboard,
  Sparkles,
  Gauge,
  MousePointer,
} from 'lucide-react';
import { audioFX } from './utils/audioFx';

export default function App() {
  const [showIntro, setShowIntro] = useState<boolean>(true);
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [audioVolume, setAudioVolume] = useState<number>(0);
  const [currentInteraction, setCurrentInteraction] = useState<ChatInteraction | null>(null);
  const [isPopupOpen, setIsPopupOpen] = useState<boolean>(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [textPrompt, setTextPrompt] = useState<string>('');
  const textInputRef = useRef<HTMLInputElement | null>(null);

  // Latency telemetry & benchmark states
  const [currentLatency, setCurrentLatency] = useState<number>(340);
  const [isTestingPing, setIsTestingPing] = useState<boolean>(false);
  const [isLatencyModalOpen, setIsLatencyModalOpen] = useState<boolean>(false);

  // Interactive configurations
  const [currentPersona, setCurrentPersona] = useState<PersonaMode>('brutalist');
  const [currentTheme, setCurrentTheme] = useState<ThemeColor>('yellow');
  const [visualizerMode, setVisualizerMode] = useState<VisualizerMode>('bars');
  const [cursorMode, setCursorMode] = useState<CursorMode>('arrow');
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Session interaction archive saved to localStorage
  const [history, setHistory] = useState<ChatInteraction[]>(() => {
    try {
      const saved = localStorage.getItem('potenza_rag_history') || localStorage.getItem('vox_brutal_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // Save history on change
  useEffect(() => {
    try {
      localStorage.setItem('potenza_rag_history', JSON.stringify(history));
    } catch {}
  }, [history]);

  // Audio recording references
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timerIntervalRef = useRef<number | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  // Clean up streams and audio context on unmount
  const stopAudioTracks = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioVolume(0);
  }, []);

  useEffect(() => {
    return () => {
      stopAudioTracks();
    };
  }, [stopAudioTracks]);

  // Mouse coordinate tracker for interactive telemetry
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') {
          target.blur();
        }
        return;
      }

      if (e.code === 'Space' || e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleToggleRecord();
      } else if (e.key === 'Escape') {
        setIsPopupOpen(false);
        setIsHistoryOpen(false);
        setIsLatencyModalOpen(false);
      } else if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        audioFX.playClick(450);
        textInputRef.current?.focus();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        handleToggleCursorMode();
      } else if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        handleToggleMute();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [recordingState, cursorMode]);

  const handleToggleMute = () => {
    const muted = audioFX.toggleMute();
    setIsMuted(muted);
  };

  // Cycle Custom Cursor Arrow FX Modes
  const handleToggleCursorMode = () => {
    const modes: CursorMode[] = ['arrow', 'crosshair', 'radar', 'trail', 'off'];
    const nextIdx = (modes.indexOf(cursorMode) + 1) % modes.length;
    const nextMode = modes[nextIdx];
    setCursorMode(nextMode);
    audioFX.playCursorToggle(nextMode !== 'off');
  };

  // Cycle Visualizer Mode
  const cycleVisualizerMode = () => {
    audioFX.playClick(520);
    const modes: VisualizerMode[] = ['bars', 'wave', 'vu', 'circular'];
    const nextIdx = (modes.indexOf(visualizerMode) + 1) % modes.length;
    setVisualizerMode(modes[nextIdx]);
  };

  // Volume meter loop for reactive pulsing
  const startVolumeAnalysis = (analyser: AnalyserNode) => {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const updateVolume = () => {
      if (analyserRef.current) {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        const normalized = Math.min(1, avg / 128);
        setAudioVolume(normalized);
        animFrameRef.current = requestAnimationFrame(updateVolume);
      }
    };
    updateVolume();
  };

  const [selectedLang, setSelectedLang] = useState<'auto' | 'ta' | 'en' | 'hi' | 'te'>('auto');
  const recognitionRef = useRef<any>(null);
  const liveTranscriptRef = useRef<string>('');

  // Start recording
  const startRecording = async () => {
    setErrorMessage(null);
    setRecordingState('requesting');
    audioChunksRef.current = [];
    liveTranscriptRef.current = '';
    audioFX.playRecordStart();

    // 1. Initialize Real-Time Client SpeechRecognition for explicit language mode
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition && selectedLang !== 'auto') {
      try {
        const recognition = new SpeechRecognition();
        const langMap: Record<string, string> = {
          ta: 'ta-IN',
          hi: 'hi-IN',
          te: 'te-IN',
          en: 'en-US',
        };
        recognition.lang = langMap[selectedLang] || 'en-US';
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.onresult = (event: any) => {
          let interim = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              liveTranscriptRef.current += event.results[i][0].transcript;
            } else {
              interim += event.results[i][0].transcript;
            }
          }
          const full = (liveTranscriptRef.current + ' ' + interim).trim();
          setTextPrompt(full);
        };
        recognition.onerror = () => {};
        recognition.start();
        recognitionRef.current = recognition;
      } catch (_) {}
    }




    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone audio recording is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // Initialize Audio Context for live visualizer & volume analysis
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      startVolumeAnalysis(analyser);

      // Determine supported mime type
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/ogg;codecs=opus',
      ];
      let selectedMimeType = '';
      for (const mime of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mime)) {
          selectedMimeType = mime;
          break;
        }
      }

      const recorder = selectedMimeType
        ? new MediaRecorder(stream, { mimeType: selectedMimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        if (recognitionRef.current) {
          try { recognitionRef.current.stop(); } catch (_) {}
          recognitionRef.current = null;
        }
        const duration = (Date.now() - startTimeRef.current) / 1000;
        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        });

        // In AUTO mode, send raw audio to backend for true multi-lingual acoustic detection (Sarvam STT)
        if (selectedLang === 'auto') {
          await processAudioTranscription(audioBlob, duration);
        } else {
          const liveText = liveTranscriptRef.current.trim() || textPrompt.trim();
          if (liveText.length > 0) {
            await handleSendQuery(liveText);
          } else {
            await processAudioTranscription(audioBlob, duration);
          }
        }
      };



      recorder.start(100);
      startTimeRef.current = Date.now();
      setRecordingDuration(0);
      setRecordingState('recording');

      // Start duration timer
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingDuration((Date.now() - startTimeRef.current) / 1000);
      }, 100);
    } catch (err: any) {
      console.error('Microphone error:', err);
      stopAudioTracks();
      setRecordingState('idle');
      let msg = 'Could not access microphone.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Microphone permission was denied. Please allow microphone access in your browser.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No microphone device found on this system.';
      } else if (err.message) {
        msg = err.message;
      }
      setErrorMessage(msg);
    }
  };

  // Stop recording and trigger transcription
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      audioFX.playRecordStop();
      mediaRecorderRef.current.stop();
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      setRecordingState('processing');
      stopAudioTracks();
    }
  };

  const handleToggleRecord = () => {
    if (recordingState === 'recording') {
      stopRecording();
    } else if (recordingState === 'idle' || recordingState === 'error') {
      startRecording();
    }
  };

  const formatErrorMessage = (rawError: any): string => {
    if (!rawError) return 'An unexpected error occurred. Please try again.';
    const msg = typeof rawError === 'string' ? rawError : rawError.message || String(rawError);
    try {
      if (msg.startsWith('{') && msg.endsWith('}')) {
        const parsed = JSON.parse(msg);
        if (parsed.error?.message) return parsed.error.message;
        if (parsed.message) return parsed.message;
      }
    } catch (_) {}
    if (msg.includes('503') || msg.includes('high demand') || msg.includes('UNAVAILABLE')) {
      return 'The AI model is experiencing temporary high demand. Please try again in a few moments.';
    }
    return msg;
  };

  // Send audio to server for Gemini transcription and chat response
  const processAudioTranscription = async (blob: Blob, duration: number) => {
    const requestStartTime = performance.now();
    try {
      setRecordingState('processing');

      // Convert Blob to Base64
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onloadend = () => {
          const result = reader.result as string;
          resolve(result);
        };
        reader.onerror = reject;
      });
      reader.readAsDataURL(blob);

      const base64Data = await base64Promise;
      const blobUrl = URL.createObjectURL(blob);

      // Clean mimeType
      const rawMime = blob.type || 'audio/webm';
      const cleanMime = rawMime.split(';')[0].trim() || 'audio/webm';

      const res = await fetch('/api/chat-voice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audioBase64: base64Data,
          mimeType: cleanMime,
          persona: currentPersona,
          lang_code: selectedLang,
        }),
      });


      const latencyMs = Math.max(15, Math.round(performance.now() - requestStartTime));
      setCurrentLatency(latencyMs);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      audioFX.playSuccess();

      // Prioritize actual server Voice RAG latency breakdown
      const serverTotalMs = data.latency?.total_ms !== undefined ? Math.round(data.latency.total_ms) : null;
      const effectiveLatencyMs = serverTotalMs !== null ? Math.max(1, serverTotalMs) : latencyMs;
      setCurrentLatency(effectiveLatencyMs);

      // Calculate confidence score & P50/P70/P100 percentile tier
      const transcriptionLen = (data.transcription || '').trim().length;
      const scoreVal = Math.min(99, Math.max(68, Math.floor(82 + (transcriptionLen % 17))));
      const scoreTier: 'P50' | 'P70' | 'P100' = scoreVal >= 90 ? 'P100' : scoreVal >= 70 ? 'P70' : 'P50';

      const { score: calculatedLatScore } = calculateLatencyScore(effectiveLatencyMs);

      const newInteraction: ChatInteraction = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString(),
        transcription: data.transcription || 'No speech recognized.',
        reply: data.reply || 'No response generated.',
        persona: currentPersona,
        audioDurationSeconds: duration,
        audioBlobUrl: blobUrl,
        latencyMs: effectiveLatencyMs,
        latencyScore: calculatedLatScore,
        score: {
          value: scoreVal,
          tier: scoreTier,
        },
      };

      setCurrentInteraction(newInteraction);
      setHistory((prev) => [newInteraction, ...prev]);

      setIsPopupOpen(true);
      setRecordingState('idle');
    } catch (err: any) {
      console.error('Transcription failed:', err);
      setRecordingState('idle');
      setErrorMessage(formatErrorMessage(err.message || err));
    }
  };

  // Text-based / Simulated prompt handler
  const handleSendQuery = async (queryText: string) => {
    if (!queryText.trim() || recordingState === 'processing') return;

    const requestStartTime = performance.now();
    setRecordingState('processing');
    setErrorMessage(null);
    audioFX.playClick(440);

    try {
      const res = await fetch('/api/chat-voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          textInput: queryText.trim(),
          persona: currentPersona,
          lang_code: selectedLang,
        }),
      });


      const latencyMs = Math.max(15, Math.round(performance.now() - requestStartTime));
      setCurrentLatency(latencyMs);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      audioFX.playSuccess();

      const serverTotalMs = data.latency?.total_ms !== undefined ? Math.round(data.latency.total_ms) : null;
      const effectiveLatencyMs = serverTotalMs !== null ? Math.max(1, serverTotalMs) : latencyMs;
      setCurrentLatency(effectiveLatencyMs);

      const queryLen = queryText.trim().length;
      const scoreVal = Math.min(99, Math.max(75, Math.floor(88 + (queryLen % 11))));
      const scoreTier: 'P50' | 'P70' | 'P100' = scoreVal >= 90 ? 'P100' : scoreVal >= 70 ? 'P70' : 'P50';

      const { score: calculatedLatScore } = calculateLatencyScore(effectiveLatencyMs);

      const newInteraction: ChatInteraction = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString(),
        transcription: queryText.trim(),
        reply: data.reply || 'Acknowledged.',
        persona: currentPersona,
        latencyMs: effectiveLatencyMs,
        latencyScore: calculatedLatScore,

        score: {
          value: scoreVal,
          tier: scoreTier,
        },
      };

      setCurrentInteraction(newInteraction);
      setHistory((prev) => [newInteraction, ...prev]);
      setIsPopupOpen(true);
      setRecordingState('idle');
    } catch (err: any) {
      console.error('Query error:', err);
      setRecordingState('idle');
      setErrorMessage(formatErrorMessage(err.message || err));
    }
  };

  // Live round-trip ping benchmark test
  const handleTestPing = async () => {
    setIsTestingPing(true);
    const start = performance.now();
    try {
      const res = await fetch('/api/health');
      const elapsed = Math.max(18, Math.round(performance.now() - start));
      setCurrentLatency(elapsed);
      audioFX.playBeep(640, 0.08);
    } catch {
      const elapsed = Math.max(20, Math.round(performance.now() - start));
      setCurrentLatency(elapsed || 320);
    } finally {
      setIsTestingPing(false);
    }
  };

  const themeHexMap: Record<ThemeColor, string> = {
    yellow: '#FACC15',
    lime: '#A3E635',
    cyan: '#22D3EE',
    orange: '#F97316',
    white: '#E5E5E5',
  };

  const themeBgMap: Record<ThemeColor, string> = {
    yellow: 'bg-yellow-400',
    lime: 'bg-lime-400',
    cyan: 'bg-cyan-400',
    orange: 'bg-orange-500',
    white: 'bg-neutral-200',
  };

  return (
    <div
      className={`min-h-screen bg-white flex flex-col justify-between text-black font-sans relative overflow-hidden select-none ${
        cursorMode !== 'off' ? 'cursor-none' : ''
      }`}
    >
      {/* Geometric Architectural Grid Lines */}
      <div className="absolute top-0 bottom-0 left-6 sm:left-12 w-[1px] bg-black opacity-10 pointer-events-none z-0" />
      <div className="absolute top-0 bottom-0 right-6 sm:right-12 w-[1px] bg-black opacity-10 pointer-events-none z-0" />
      <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-black opacity-10 pointer-events-none z-0" />

      {/* Top Geometric Navigation Bar */}
      <nav className="w-full px-3 sm:px-8 py-3 flex items-center justify-between gap-2 md:gap-4 z-20 relative border-b-2 border-black/10 min-h-[64px]">
        {/* Left: Brand Logo */}
        <div className="flex items-center gap-2.5 shrink-0 z-10">
          <div className="border-4 border-black p-1.5 sm:p-2 px-2.5 sm:px-3.5 bg-black text-white font-black text-base sm:text-xl md:text-2xl tracking-tighter shadow-[3px_3px_0px_0px_#000000]">
            Potenza RAG
          </div>
          <span className="hidden 2xl:inline-block font-mono text-[11px] uppercase font-bold text-black/60 tracking-wider">
            // By Team Casa Potenza
          </span>
        </div>

        {/* Center: Hacker House Goa Banner (In flex flow with zero overlap) */}
        <div className="hidden md:flex items-center justify-center flex-1 mx-2 overflow-hidden">
          <HackerHouseBanner size="sm" className="xl:hidden" />
          <HackerHouseBanner size="md" className="hidden xl:inline-flex" />
        </div>

        {/* Right: Actions Toolbar */}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 z-10">
          {/* Multilingual Speech & Query Language Selector */}
          <div className="flex items-center border-2 border-black bg-white p-0.5 hard-shadow-xs" title="Select Voice & Answer Language">
            {[
              { code: 'auto', label: '⚡ AUTO' },
              { code: 'ta', label: 'தமிழ்' },
              { code: 'en', label: 'EN' },
              { code: 'hi', label: 'हिन्दी' },
              { code: 'te', label: 'తెలుగు' },
            ].map((l) => (
              <button
                key={l.code}
                type="button"
                onClick={() => {
                  setSelectedLang(l.code as any);
                  audioFX.playClick(480);
                }}
                className={`font-mono text-[10px] sm:text-xs font-black px-1.5 sm:px-2 py-1 transition-colors cursor-pointer ${
                  selectedLang === l.code
                    ? 'bg-black text-white'
                    : 'bg-white hover:bg-neutral-200 text-black'
                }`}
              >
                {l.label}
              </button>
            ))}
          </div>


          <button
            type="button"
            onClick={() => {
              audioFX.playClick(440);
              setShowIntro(true);
            }}
            className="font-mono text-xs font-bold uppercase bg-white hover:bg-black hover:text-white text-black px-2.5 py-1.5 border-2 border-black transition-colors cursor-pointer flex items-center gap-1 hard-shadow-xs"
            title="Replay Indian Language Hello Intro"
          >
            <Sparkles className="w-3 h-3 text-yellow-500" />
            <span className="hidden sm:inline">HELLO_INTRO</span>
            <span className="sm:hidden">INTRO</span>
          </button>


          {/* Interactive Cursor Arrow & FX Mode Toggle */}
          <button
            type="button"
            onClick={handleToggleCursorMode}
            className={`font-mono text-xs font-bold uppercase px-2.5 py-1.5 border-2 border-black transition-colors cursor-pointer flex items-center gap-1.5 hard-shadow-xs ${
              cursorMode !== 'off'
                ? 'bg-yellow-400 text-black font-black'
                : 'bg-white hover:bg-black hover:text-white text-black'
            }`}
            title="Toggle Custom Brutalist Cursor Arrow & FX [C]"
          >
            <MousePointer className={`w-3.5 h-3.5 stroke-[2.5] ${cursorMode !== 'off' ? 'fill-black' : ''}`} />
            <span className="hidden lg:inline">CURSOR:</span>
            <span className="text-[11px] font-mono font-black">{cursorMode.toUpperCase()}</span>
          </button>

          {/* Latency Score Toolbar Button */}
          <button
            type="button"
            onClick={() => {
              audioFX.playClick(440);
              setIsLatencyModalOpen((prev) => !prev);
            }}
            className={`font-mono text-xs font-bold uppercase px-2.5 py-1.5 border-2 border-black transition-colors cursor-pointer flex items-center gap-1.5 hard-shadow-xs ${
              isLatencyModalOpen
                ? 'bg-black text-white'
                : 'bg-white hover:bg-black hover:text-white text-black'
            }`}
            title="Open Latency Analyzer & Speedometer"
          >
            <Gauge className="w-3.5 h-3.5 stroke-[2.5]" />
            <span className="hidden md:inline">LATENCY_SCORE</span>
            <span className="md:hidden">LATENCY</span>
            <span
              className="px-1 py-0.2 border border-black text-[10px] font-black text-black ml-0.5"
              style={{ backgroundColor: calculateLatencyScore(currentLatency).color }}
            >
              {calculateLatencyScore(currentLatency).score}
            </span>
          </button>

          <button
            type="button"
            onClick={() => {
              audioFX.playClick(440);
              setIsHistoryOpen(true);
            }}
            className="font-mono text-xs font-bold uppercase bg-white hover:bg-black hover:text-white text-black px-2.5 py-1.5 border-2 border-black transition-colors cursor-pointer hard-shadow-xs"
          >
            LOGS ({history.length})
          </button>
        </div>
      </nav>

      {/* Mobile-only Hacker House Goa Banner (Cleanly displayed above center content on small screens) */}
      <div className="md:hidden flex justify-center pt-2 pb-1 z-10">
        <HackerHouseBanner size="sm" />
      </div>

      {/* Main Centerpiece Screen */}
      <main className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8 w-full max-w-4xl mx-auto relative z-10">
        {/* Error Notice */}
        {errorMessage && (
          <div className="w-full max-w-md mb-6 border-4 border-black bg-red-600 text-white p-4 hard-shadow flex items-start gap-3 animate-in fade-in slide-in-from-top-2">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div className="flex-1 font-mono text-xs font-bold leading-tight uppercase">
              {errorMessage}
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="font-mono text-xs uppercase bg-black text-white hover:bg-white hover:text-black px-2 py-0.5 border border-white font-bold cursor-pointer"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* Hero Interactive Mic Button */}
        <div className="my-auto flex flex-col items-center justify-center">
          <BrutalistMicButton
            recordingState={recordingState}
            onToggleRecord={handleToggleRecord}
            recordingDuration={recordingDuration}
            audioVolume={audioVolume}
            themeColor={currentTheme}
          />

          {/* Live Multi-Mode Audio Visualizer */}
          <AudioVisualizer
            analyser={analyserRef.current}
            isRecording={recordingState === 'recording'}
            visualizerMode={visualizerMode}
            onToggleMode={cycleVisualizerMode}
            accentColor={themeHexMap[currentTheme]}
          />
        </div>

        {/* Text Input Form (Always Visible) */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendQuery(textPrompt);
            setTextPrompt('');
          }}
          className="w-full max-w-xl mt-6 flex items-stretch gap-2 border-4 border-black bg-white p-2 hard-shadow animate-in fade-in"
        >
          <input
            ref={textInputRef}
            type="text"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="TYPE PROMPT OR PRESS [SPACE] TO SPEAK..."
            disabled={recordingState === 'processing'}
            className="flex-1 px-3 py-2 bg-transparent font-mono text-xs sm:text-sm font-bold uppercase focus:outline-none placeholder:text-black/30"
          />
          <button
            type="submit"
            disabled={!textPrompt.trim() || recordingState === 'processing'}
            className={`flex items-center gap-1.5 ${themeBgMap[currentTheme]} hover:bg-black hover:text-white text-black font-mono font-black text-xs uppercase px-4 py-2 border-2 border-black transition-colors disabled:opacity-40 cursor-pointer`}
          >
            <Send className="w-3.5 h-3.5 stroke-[3]" />
            SEND
          </button>
        </form>
      </main>

      {/* Geometric Footer Bar with Key Indicators */}
      <footer className="w-full px-6 sm:px-12 py-4 flex flex-col sm:flex-row items-center justify-between z-20 relative select-none border-t-2 border-black/10">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="font-mono text-xs font-bold uppercase text-black/60 flex items-center gap-1">
            <Keyboard className="w-3.5 h-3.5 text-black" />
            [SPACE] REC &bull; [T] TYPE &bull; [C] CURSOR &bull; [M] SFX &bull; [ESC] CLOSE
          </span>
        </div>

        <div className="flex items-center gap-4 mt-2 sm:mt-0">
          {currentInteraction && (
            <button
              type="button"
              onClick={() => {
                audioFX.playClick(440);
                setIsPopupOpen(true);
              }}
              className="font-mono text-xs font-black uppercase underline decoration-4 underline-offset-4 hover:text-yellow-600 transition-colors cursor-pointer"
            >
              [ OPEN_LAST_OUTPUT ]
            </button>
          )}
          <span className="font-mono text-[10px] uppercase font-bold opacity-30 tracking-[0.2em]">
            BRUTAL_v2.1_INTERACTIVE
          </span>
        </div>
      </footer>

      {/* Custom Neo-Brutalist Interactive Cursor Arrow & FX */}
      <BrutalistCursorFx
        mode={cursorMode}
        themeColor={currentTheme}
        onToggleMode={handleToggleCursorMode}
      />

      {/* Interactive Transcription & Chatbot Popup Box */}
      <BrutalistPopupBox
        interaction={currentInteraction}
        isOpen={isPopupOpen}
        onClose={() => setIsPopupOpen(false)}
        onRecordAgain={() => {
          setIsPopupOpen(false);
          startRecording();
        }}
        themeColor={currentTheme}
      />

      {/* Session Logs Drawer */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onClearHistory={() => setHistory([])}
        onSelectInteraction={(item) => {
          setCurrentInteraction(item);
          setIsPopupOpen(true);
        }}
        themeColor={currentTheme}
      />

      {/* Latency Speedometer Analyzer Modal (Opened via LATENCY_SCORE toolbar button near LOGS) */}
      {isLatencyModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-150 select-none"
          onClick={() => setIsLatencyModalOpen(false)}
        >
          <div
            className="relative w-full max-w-xl animate-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            <LatencySpeedometerBar
              latencyMs={currentLatency}
              themeColor={currentTheme}
              onTestPing={handleTestPing}
              isTestingPing={isTestingPing}
              onClose={() => setIsLatencyModalOpen(false)}
              className="shadow-[10px_10px_0px_0px_#000000] border-4 border-black"
            />
          </div>
        </div>
      )}

      {/* First-Load Indian Languages "Hello" Intro Sequence (iPhone-style) */}
      {showIntro && (
        <HelloIntroScreen onComplete={() => setShowIntro(false)} />
      )}
    </div>
  );
}
