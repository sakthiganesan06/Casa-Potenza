import React, { useState } from 'react';
import { motion } from 'motion/react';
import { ChatInteraction, ThemeColor } from '../types';
import { X, Trash2, Volume2, Copy, Check, Mic, ArrowUpRight } from 'lucide-react';
import { audioFX } from '../utils/audioFx';
import { ScoreSmallBar } from './ScoreSmallBar';
import { LatencySpeedometerBar } from './LatencySpeedometerBar';

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  history: ChatInteraction[];
  onClearHistory: () => void;
  onSelectInteraction: (item: ChatInteraction) => void;
  themeColor: ThemeColor;
}

const listContainerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.07,
      delayChildren: 0.08,
    },
  },
};

const listItemVariants = {
  hidden: { opacity: 0, y: 14 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.3,
      ease: [0.215, 0.61, 0.355, 1],
    },
  },
};

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  history,
  onClearHistory,
  onSelectInteraction,
  themeColor,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleCopy = (item: ChatInteraction) => {
    const text = `[INPUT (${item.persona})]: ${item.transcription}\n\n[REPLY]: ${item.reply}`;
    navigator.clipboard.writeText(text);
    setCopiedId(item.id);
    audioFX.playClick(600);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleTTS = (text: string, id: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      if (playingId === id) {
        setPlayingId(null);
        return;
      }
      const u = new SpeechSynthesisUtterance(text);
      u.onend = () => setPlayingId(null);
      u.onerror = () => setPlayingId(null);
      setPlayingId(id);
      window.speechSynthesis.speak(u);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/60 backdrop-blur-[2px] transition-all"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative w-full max-w-3xl bg-white border-[6px] border-black p-6 hard-shadow z-30 flex flex-col max-h-[85vh] overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between border-b-4 border-black pb-4 select-none">
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-black" />
            <h2 className="font-mono text-base sm:text-lg font-black uppercase tracking-tight">
              SESSION_LOGS // ARCHIVE ({history.length})
            </h2>
          </div>
          <div className="flex items-center gap-3">
            {history.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  audioFX.playClick(300);
                  onClearHistory();
                }}
                className="flex items-center gap-1 font-mono text-xs font-bold uppercase bg-red-100 text-red-700 hover:bg-red-600 hover:text-white px-2.5 py-1 border-2 border-black transition-colors cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                CLEAR_ALL
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="bg-black text-white hover:bg-red-600 p-1 border-2 border-black transition-colors cursor-pointer"
            >
              <X className="w-4 h-4 stroke-[3]" />
            </button>
          </div>
        </div>

        {/* List of Interactions with Staggered Fade-in Animation */}
        <div className="flex-1 overflow-y-auto py-4 pr-1">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <Mic className="w-12 h-12 text-black/30 stroke-[2] mb-3" />
              <p className="font-mono text-sm font-bold uppercase text-black/50">
                NO_PAST_RECORDINGS_FOUND
              </p>
              <p className="font-mono text-xs text-black/40 mt-1">
                Record with the mic to populate the session stream
              </p>
            </div>
          ) : (
            <motion.div
              variants={listContainerVariants}
              initial="hidden"
              animate="visible"
              className="space-y-4"
            >
              {history.map((item) => (
                <motion.div
                  key={item.id}
                  variants={listItemVariants}
                  className="border-3 border-black p-4 bg-neutral-50 hover:bg-white transition-colors hard-shadow-xs relative group"
                >
                  <div className="flex items-center justify-between border-b-2 border-black/20 pb-2 mb-2 font-mono text-xs font-bold">
                    <div className="flex items-center gap-2">
                      <span className="bg-yellow-400 text-black px-1.5 py-0.5 uppercase text-[10px] font-black border border-black">
                        {item.persona}
                      </span>
                      <span className="opacity-40">{item.timestamp}</span>
                      {item.audioDurationSeconds && (
                        <span className="opacity-50">({item.audioDurationSeconds.toFixed(1)}s)</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleTTS(item.reply, item.id)}
                        className={`font-mono text-[10px] uppercase font-bold px-2 py-0.5 border border-black transition-colors flex items-center gap-1 cursor-pointer ${
                          playingId === item.id ? 'bg-black text-white' : 'bg-white hover:bg-yellow-400 text-black'
                        }`}
                        title="Read Aloud Transcribed Answer"
                      >
                        <Volume2 className={`w-3 h-3 ${playingId === item.id ? 'text-yellow-400 animate-pulse' : ''}`} />
                        {playingId === item.id ? 'STOP' : 'READ_ALOUD'}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleCopy(item)}
                        className="hover:text-yellow-600 p-1 cursor-pointer bg-white border border-black"
                        title="Copy Full Interaction"
                      >
                        {copiedId === item.id ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          onSelectInteraction(item);
                          onClose();
                        }}
                        className="font-mono text-[10px] uppercase font-bold bg-black text-white hover:bg-yellow-400 hover:text-black px-2 py-0.5 border border-black transition-colors flex items-center gap-1 cursor-pointer"
                      >
                        VIEW_POPUP <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="font-mono text-xs sm:text-sm font-black text-black">
                      &gt; "{item.transcription}"
                    </div>
                    <div className="font-mono text-xs sm:text-sm text-black/80 font-medium whitespace-pre-wrap pl-3 border-l-2 border-yellow-400">
                      {item.reply}
                    </div>
                    <div className="pt-2 border-t border-black/10 flex flex-wrap items-center justify-between gap-2">
                      <LatencySpeedometerBar
                        latencyMs={item.latencyMs || 340}
                        latencyScore={item.latencyScore}
                        themeColor={themeColor}
                        compact={true}
                      />
                      <ScoreSmallBar
                        score={item.score || { value: 90, tier: 'P100' }}
                        themeColor={themeColor}
                        compact={true}
                      />
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t-4 border-black pt-3 flex justify-between items-center select-none">
          <span className="font-mono text-[10px] uppercase font-bold text-black/40">
            RAW_STORAGE // PERSISTENT_SESSION
          </span>
          <button
            type="button"
            onClick={onClose}
            className="font-mono text-xs font-black uppercase bg-black text-white hover:bg-yellow-400 hover:text-black px-4 py-1.5 border-2 border-black transition-colors cursor-pointer"
          >
            CLOSE_LOGS
          </button>
        </div>
      </div>
    </div>
  );
};
