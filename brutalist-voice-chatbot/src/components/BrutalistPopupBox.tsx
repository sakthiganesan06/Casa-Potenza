import React, { useState } from 'react';
import { X, Copy, Check, Volume2, VolumeX, Mic, ArrowRight, Sparkles, MessageSquare, Terminal } from 'lucide-react';
import { ChatInteraction, ThemeColor } from '../types';
import { audioFX } from '../utils/audioFx';
import { ScoreSmallBar } from './ScoreSmallBar';
import { LatencySpeedometerBar } from './LatencySpeedometerBar';

interface BrutalistPopupBoxProps {
  interaction: ChatInteraction | null;
  isOpen: boolean;
  onClose: () => void;
  onRecordAgain: () => void;
  themeColor?: ThemeColor;
}

export const BrutalistPopupBox: React.FC<BrutalistPopupBoxProps> = ({
  interaction,
  isOpen,
  onClose,
  onRecordAgain,
  themeColor = 'yellow',
}) => {
  const [copiedSection, setCopiedSection] = useState<'transcription' | 'reply' | 'all' | null>(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [activeTTSSection, setActiveTTSSection] = useState<'transcription' | 'reply' | null>(null);

  if (!isOpen || !interaction) return null;

  const copyToClipboard = (text: string, section: 'transcription' | 'reply' | 'all') => {
    navigator.clipboard.writeText(text);
    setCopiedSection(section);
    audioFX.playClick(650);
    setTimeout(() => {
      setCopiedSection(null);
    }, 2000);
  };

  const handlePlayTTS = (text: string, section: 'transcription' | 'reply') => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      if (activeTTSSection === section) {
        setActiveTTSSection(null);
        return;
      }

      audioFX.playClick(440);
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.onend = () => setActiveTTSSection(null);
      utterance.onerror = () => setActiveTTSSection(null);
      setActiveTTSSection(section);
      window.speechSynthesis.speak(utterance);
    }
  };

  const handlePlayRecordedAudio = () => {
    if (!interaction.audioBlobUrl) return;
    const audio = new Audio(interaction.audioBlobUrl);
    setIsPlayingAudio(true);
    audio.onended = () => setIsPlayingAudio(false);
    audio.onerror = () => setIsPlayingAudio(false);
    audio.play();
  };

  const themeBgMap: Record<ThemeColor, string> = {
    yellow: 'bg-yellow-400',
    lime: 'bg-lime-400',
    cyan: 'bg-cyan-400',
    orange: 'bg-orange-500',
    white: 'bg-neutral-200',
  };

  const currentThemeBg = themeBgMap[themeColor];

  return (
    <div
      id="brutalist-popup-overlay"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/65 backdrop-blur-[2px] transition-all"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          if ('speechSynthesis' in window) window.speechSynthesis.cancel();
          audioFX.playClick(300);
          onClose();
        }
      }}
    >
      <div
        id="brutalist-popup-content"
        role="dialog"
        aria-modal="true"
        className="relative w-full max-w-2xl bg-white border-[6px] border-black p-6 sm:p-8 hard-shadow z-30 flex flex-col max-h-[90vh] overflow-hidden animate-in fade-in zoom-in-95 duration-150"
      >
        {/* Geometric Header Bar */}
        <div className="flex justify-between items-center mb-5 border-b-4 border-black pb-3 select-none">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-black" />
            <span className="font-mono font-black uppercase text-xs sm:text-sm tracking-tight">
              Live_Transcription_Stream
            </span>
          </div>

          <div className="flex items-center gap-3 sm:gap-4">
            <span className={`font-mono text-xs font-black uppercase ${currentThemeBg} text-black px-2 py-0.5 border border-black`}>
              PERSONA: {interaction.persona}
            </span>
            <span className="font-mono text-xs font-bold opacity-40">
              {interaction.timestamp}
            </span>
            <button
              id="close-popup-button"
              type="button"
              onClick={() => {
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                audioFX.playClick(300);
                onClose();
              }}
              aria-label="Close dialog"
              className="bg-black text-white hover:bg-red-600 p-1 border-2 border-black transition-colors cursor-pointer"
            >
              <X className="w-4 h-4 stroke-[3]" />
            </button>
          </div>
        </div>

        {/* Scrollable Content Body */}
        <div className="space-y-5 overflow-y-auto pr-1">
          {/* 1. Raw Transcription Input Headline */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className={`font-mono text-[11px] font-bold uppercase tracking-widest ${currentThemeBg} border border-black px-2 py-0.5`}>
                VERBATIM_INPUT {interaction.audioDurationSeconds ? `(${interaction.audioDurationSeconds.toFixed(1)}s)` : ''}
              </span>
              <div className="flex items-center gap-2">
                {interaction.audioBlobUrl && (
                  <button
                    type="button"
                    onClick={handlePlayRecordedAudio}
                    title="Play original audio recording"
                    className="font-mono text-xs font-bold uppercase hover:bg-black hover:text-white px-2 py-0.5 border border-black transition-colors cursor-pointer flex items-center gap-1"
                  >
                    <Volume2 className="w-3 h-3" />
                    {isPlayingAudio ? 'PLAYING...' : 'REPLAY_AUDIO'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => handlePlayTTS(interaction.transcription, 'transcription')}
                  title="Read aloud verbatim user prompt"
                  className={`font-mono text-xs font-bold uppercase px-2 py-0.5 border border-black transition-colors cursor-pointer flex items-center gap-1 ${
                    activeTTSSection === 'transcription'
                      ? 'bg-black text-white'
                      : 'bg-white hover:bg-black hover:text-white text-black'
                  }`}
                >
                  <Volume2 className={`w-3 h-3 ${activeTTSSection === 'transcription' ? 'animate-pulse text-yellow-400' : ''}`} />
                  {activeTTSSection === 'transcription' ? 'STOP' : 'READ_ALOUD'}
                </button>
                <button
                  type="button"
                  onClick={() => copyToClipboard(interaction.transcription, 'transcription')}
                  title="Copy transcription"
                  className="font-mono text-xs font-bold uppercase hover:bg-black hover:text-white px-2 py-0.5 border border-black transition-colors flex items-center gap-1 cursor-pointer"
                >
                  {copiedSection === 'transcription' ? <Check className="w-3 h-3 text-green-600" /> : <Copy className="w-3 h-3" />}
                  {copiedSection === 'transcription' ? 'COPIED' : 'COPY'}
                </button>
              </div>
            </div>

            <p className="text-xl sm:text-2xl font-black uppercase leading-tight tracking-tighter font-sans select-text border-l-4 border-black pl-3 py-1">
              "{interaction.transcription}"
            </p>
          </div>

          {/* 2. Latency Analysis Speedometer Bar & Score Metrics */}
          <LatencySpeedometerBar
            latencyMs={interaction.latencyMs || 340}
            latencyScore={interaction.latencyScore}
            themeColor={themeColor}
          />

          {/* 3. Score Smallbar (P50 / P70 / P100 Score Metric) */}
          <ScoreSmallBar
            score={interaction.score || { value: 92, tier: 'P100' }}
            themeColor={themeColor}
          />

          {/* 4. AI Transcribed Answer Output Card */}
          <div className="bg-neutral-50 border-4 border-black p-5 hard-shadow-xs relative">
            <div className="flex items-center justify-between border-b-2 border-black pb-2 mb-3">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 ${currentThemeBg} border border-black`} />
                <span className="font-mono text-xs font-black uppercase tracking-wider">
                  AI_TRANSCRIBED_ANSWER
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handlePlayTTS(interaction.reply, 'reply')}
                  className={`font-mono text-xs font-bold uppercase px-2.5 py-1 border-2 border-black transition-colors cursor-pointer flex items-center gap-1.5 ${
                    activeTTSSection === 'reply'
                      ? 'bg-black text-white'
                      : `${currentThemeBg} hover:bg-black hover:text-white text-black font-black`
                  }`}
                  title="Read aloud the transcribed answer"
                >
                  <Volume2 className={`w-3.5 h-3.5 ${activeTTSSection === 'reply' ? 'text-yellow-400 animate-pulse' : ''}`} />
                  {activeTTSSection === 'reply' ? 'STOP_READING' : 'READ_ALOUD'}
                </button>
                <button
                  type="button"
                  onClick={() => copyToClipboard(interaction.reply, 'reply')}
                  className="font-mono text-xs font-bold uppercase bg-white hover:bg-black hover:text-white text-black px-2 py-1 border-2 border-black transition-colors flex items-center gap-1 cursor-pointer"
                  title="Copy answer"
                >
                  {copiedSection === 'reply' ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedSection === 'reply' ? 'COPIED' : 'COPY'}
                </button>
              </div>
            </div>

            <p className="font-mono text-xs sm:text-sm leading-relaxed whitespace-pre-wrap select-text font-semibold text-black">
              {interaction.reply}
            </p>
          </div>
        </div>

        {/* Geometric Bottom Bar with Indicator Blocks */}
        <div className="mt-5 pt-4 border-t-4 border-black flex flex-col sm:flex-row justify-between items-center gap-4 select-none">
          {/* Compact score indicator in footer */}
          <div className="flex items-center gap-2">
            <ScoreSmallBar
              score={interaction.score || { value: 92, tier: 'P100' }}
              themeColor={themeColor}
              compact={true}
            />
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
            <button
              type="button"
              onClick={() => {
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                audioFX.playClick(300);
                onClose();
              }}
              className="font-mono text-xs font-black uppercase underline decoration-4 underline-offset-4 hover:text-yellow-600 transition-colors px-2 py-1 cursor-pointer"
            >
              Close_Stream
            </button>
            <button
              type="button"
              onClick={() => {
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                audioFX.playClick(500);
                onRecordAgain();
              }}
              className={`font-mono text-xs font-black uppercase ${currentThemeBg} hover:bg-black hover:text-white text-black px-4 py-2 border-3 border-black hard-shadow-xs active:translate-x-[2px] active:translate-y-[2px] transition-all cursor-pointer flex items-center gap-1.5`}
            >
              <Mic className="w-3.5 h-3.5 stroke-[3]" />
              Record_Again
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
