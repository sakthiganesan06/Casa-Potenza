import React, { useState } from 'react';
import { X, Copy, Check, Volume2, Mic } from 'lucide-react';
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

export function cleanAiReplyText(text: string): string {
  if (!text) return '';
  if (text.includes("thinking process") || text.includes("**Analyze") || text.includes("<think>")) {
    const withoutThink = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
    const formulateMatch = withoutThink.match(/\*\*Formulate Answer:?\*\*\s*[\r\n]+([\s\S]+)$/i);
    if (formulateMatch && formulateMatch[1]) {
      return formulateMatch[1].replace(/^[-\s*•"':]+/, '').replace(/["'`]+$/, '').trim();
    }
    const lines = withoutThink.split('\n').map(l => l.trim()).filter(Boolean);
    const nonThinking = lines.filter(l => 
      !l.startsWith('1.') && !l.startsWith('2.') && !l.startsWith('3.') && !l.startsWith('4.') && !l.startsWith('5.') && 
      !l.startsWith('-') && !l.startsWith('*') && !l.startsWith('#') &&
      !l.toLowerCase().includes('thinking process') && 
      !l.toLowerCase().includes('analyze') && 
      !l.toLowerCase().includes('determine') &&
      !l.toLowerCase().includes('formulate')
    );
    if (nonThinking.length > 0) {
      return nonThinking[nonThinking.length - 1].replace(/^[-\s*•"':]+/, '').replace(/["'`]+$/, '').trim();
    }
    if (lines.length > 0) {
      return lines[lines.length - 1].replace(/^[-\s*•"':]+/, '').replace(/["'`]+$/, '').trim();
    }
  }
  return text.replace(/^"|"$/g, '').trim();
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

  const displayReply = cleanAiReplyText(interaction.reply);

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
        className="relative w-full max-w-2xl bg-white border-4 sm:border-[6px] border-black p-4 sm:p-8 hard-shadow z-30 flex flex-col max-h-[90vh] overflow-hidden animate-in fade-in zoom-in-95 duration-150"
      >
        {/* Geometric Header Bar */}
        <div className="flex flex-wrap justify-between items-center gap-2 mb-4 border-b-3 sm:border-b-4 border-black pb-3 select-none">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-3.5 h-3.5 bg-black" />
            <span className="font-mono font-black uppercase text-xs sm:text-sm tracking-tight">
              Live_Transcription_Stream
            </span>
          </div>

          <div className="flex items-center gap-2 sm:gap-4 ml-auto">
            <span className={`font-mono text-[10px] sm:text-xs font-black uppercase ${currentThemeBg} text-black px-1.5 sm:px-2 py-0.5 border border-black`}>
              PERSONA: {interaction.persona}
            </span>
            <span className="font-mono text-[10px] sm:text-xs font-bold opacity-40">
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
        <div className="space-y-4 sm:space-y-5 overflow-y-auto pr-1">
          {/* 1. Raw Transcription Input Headline */}
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className={`font-mono text-[10px] sm:text-[11px] font-bold uppercase tracking-widest ${currentThemeBg} border border-black px-1.5 sm:px-2 py-0.5`}>
                VERBATIM_INPUT {interaction.audioDurationSeconds ? `(${interaction.audioDurationSeconds.toFixed(1)}s)` : ''}
              </span>
              <div className="flex items-center gap-1.5 sm:gap-2">
                {interaction.audioBlobUrl && (
                  <button
                    type="button"
                    onClick={handlePlayRecordedAudio}
                    title="Play original audio recording"
                    className="font-mono text-[11px] sm:text-xs font-bold uppercase hover:bg-black hover:text-white px-2 py-0.5 border border-black transition-colors cursor-pointer flex items-center gap-1"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    {isPlayingAudio ? 'PLAYING...' : 'AUDIO'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => handlePlayTTS(interaction.transcription, 'transcription')}
                  className={`font-mono text-[11px] sm:text-xs font-bold uppercase px-2 py-0.5 border border-black transition-colors cursor-pointer flex items-center gap-1 ${
                    activeTTSSection === 'transcription'
                      ? 'bg-black text-white'
                      : 'bg-white hover:bg-black hover:text-white text-black'
                  }`}
                  title="Read aloud the transcribed input"
                >
                  <Volume2 className={`w-3.5 h-3.5 ${activeTTSSection === 'transcription' ? 'text-yellow-400 animate-pulse' : ''}`} />
                  {activeTTSSection === 'transcription' ? 'STOP' : 'TTS'}
                </button>
                <button
                  type="button"
                  onClick={() => copyToClipboard(interaction.transcription, 'transcription')}
                  className="font-mono text-[11px] sm:text-xs font-bold uppercase bg-white hover:bg-black hover:text-white text-black px-2 py-0.5 border border-black transition-colors flex items-center gap-1 cursor-pointer"
                  title="Copy transcription"
                >
                  {copiedSection === 'transcription' ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedSection === 'transcription' ? 'COPIED' : 'COPY'}
                </button>
              </div>
            </div>

            <div className="bg-white border-3 sm:border-4 border-black p-3 sm:p-4 hard-shadow-xs">
              <p className="font-mono text-xs sm:text-sm md:text-base font-bold uppercase leading-relaxed text-black select-text">
                &ldquo;{interaction.transcription}&rdquo;
              </p>
            </div>
          </div>

          {/* 2. Latency Speedometer & Score Breakdown in Popup */}
          <LatencySpeedometerBar
            latencyMs={interaction.latencyMs || 280}
            latencyScore={interaction.latencyScore}
            themeColor={themeColor}
          />

          {/* 3. Score Smallbar (P50 / P70 / P100 Score Metric) */}
          <ScoreSmallBar
            score={interaction.score || { value: 92, tier: 'P100' }}
            themeColor={themeColor}
          />

          {/* 4. AI Transcribed Answer Output Card */}
          <div className="bg-neutral-50 border-3 sm:border-4 border-black p-3 sm:p-5 hard-shadow-xs relative">
            <div className="flex flex-wrap items-center justify-between border-b-2 border-black pb-2 mb-3 gap-2">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 ${currentThemeBg} border border-black`} />
                <span className="font-mono text-xs font-black uppercase tracking-wider">
                  AI_TRANSCRIBED_ANSWER
                </span>
              </div>
              <div className="flex items-center gap-1.5 sm:gap-2">
                <button
                  type="button"
                  onClick={() => handlePlayTTS(displayReply, 'reply')}
                  className={`font-mono text-[11px] sm:text-xs font-bold uppercase px-2 sm:px-2.5 py-1 border-2 border-black transition-colors cursor-pointer flex items-center gap-1 sm:gap-1.5 ${
                    activeTTSSection === 'reply'
                      ? 'bg-black text-white'
                      : `${currentThemeBg} hover:bg-black hover:text-white text-black font-black`
                  }`}
                  title="Read aloud the transcribed answer"
                >
                  <Volume2 className={`w-3.5 h-3.5 ${activeTTSSection === 'reply' ? 'text-yellow-400 animate-pulse' : ''}`} />
                  {activeTTSSection === 'reply' ? 'STOP' : 'READ_ALOUD'}
                </button>
                <button
                  type="button"
                  onClick={() => copyToClipboard(displayReply, 'reply')}
                  className="font-mono text-[11px] sm:text-xs font-bold uppercase bg-white hover:bg-black hover:text-white text-black px-2 sm:px-2.5 py-1 border-2 border-black transition-colors flex items-center gap-1 cursor-pointer"
                  title="Copy answer"
                >
                  {copiedSection === 'reply' ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedSection === 'reply' ? 'COPIED' : 'COPY'}
                </button>
              </div>
            </div>

            <p className="font-mono text-xs sm:text-sm leading-relaxed whitespace-pre-wrap select-text font-semibold text-black">
              {displayReply}
            </p>
          </div>
        </div>

        {/* Geometric Bottom Bar with Indicator Blocks */}
        <div className="mt-4 pt-3 border-t-3 sm:border-t-4 border-black flex flex-col sm:flex-row justify-between items-center gap-3 select-none">
          {/* Compact score indicator in footer */}
          <div className="flex items-center gap-2 w-full sm:w-auto justify-start">
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
              className={`font-mono text-xs font-black uppercase ${currentThemeBg} hover:bg-black hover:text-white text-black px-3 sm:px-4 py-2 border-2 sm:border-3 border-black hard-shadow-xs active:translate-x-[2px] active:translate-y-[2px] transition-all cursor-pointer flex items-center gap-1.5`}
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
