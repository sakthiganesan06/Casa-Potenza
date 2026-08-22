import React from 'react';
import { Square, Loader2 } from 'lucide-react';
import { RecordingState, ThemeColor } from '../types';

interface BrutalistMicButtonProps {
  recordingState: RecordingState;
  onToggleRecord: () => void;
  recordingDuration: number;
  audioVolume: number; // 0 to 1
  themeColor?: ThemeColor;
}

export const BrutalistMicButton: React.FC<BrutalistMicButtonProps> = ({
  recordingState,
  onToggleRecord,
  recordingDuration,
  audioVolume,
  themeColor = 'yellow',
}) => {
  const isRecording = recordingState === 'recording';
  const isProcessing = recordingState === 'processing';
  const isRequesting = recordingState === 'requesting';

  // Format recording seconds (e.g. 00:04.2)
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const dec = Math.floor((seconds % 1) * 10);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${dec}`;
  };

  // Convert volume to dB approximation for telemetry
  const approxDb = Math.round(audioVolume * 60 - 60);

  // Dynamic scale from volume
  const dynamicVolumeScale = isRecording ? 1 + audioVolume * 0.28 : 1;

  // Theme color maps for primary accent
  const themeBgMap: Record<ThemeColor, string> = {
    yellow: 'bg-yellow-400',
    lime: 'bg-lime-400',
    cyan: 'bg-cyan-400',
    orange: 'bg-orange-500',
    white: 'bg-neutral-100',
  };

  const themeTextMap: Record<ThemeColor, string> = {
    yellow: 'text-yellow-400',
    lime: 'text-lime-400',
    cyan: 'text-cyan-400',
    orange: 'text-orange-500',
    white: 'text-white',
  };

  const currentThemeBg = themeBgMap[themeColor];
  const currentThemeText = themeTextMap[themeColor];

  return (
    <div className="relative flex flex-col items-center justify-center select-none my-auto">
      {/* Geometric Concentric Pulse Rings */}
      <div className="relative flex items-center justify-center">
        {/* Ring 1 - Immediate Audio Reactive Ring */}
        <div
          className={`pulse-ring ring-1 transition-transform duration-75 ${
            isRecording ? `animate-ring-1 border-black ${currentThemeBg}/30` : ''
          }`}
          style={{
            transform: isRecording ? `scale(${1 + audioVolume * 0.45})` : 'scale(1)',
          }}
        />
        {/* Ring 2 - Mid Shockwave */}
        <div
          className={`pulse-ring ring-2 transition-transform duration-100 ${
            isRecording ? 'animate-ring-2 border-black bg-red-500/15' : ''
          }`}
          style={{
            transform: isRecording ? `scale(${1 + audioVolume * 0.75})` : 'scale(1)',
          }}
        />
        {/* Ring 3 - Outer Shockwave */}
        <div
          className={`pulse-ring ring-3 transition-transform duration-150 ${
            isRecording ? 'animate-ring-3 border-black' : ''
          }`}
          style={{
            transform: isRecording ? `scale(${1 + audioVolume * 1.05})` : 'scale(1)',
          }}
        />

        {/* Hero Geometric Balance Mic Button */}
        <button
          id="brutalist-mic-button"
          type="button"
          onClick={onToggleRecord}
          disabled={isProcessing || isRequesting}
          aria-label={isRecording ? 'Stop Recording' : 'Start Voice Recording'}
          style={{ transform: `scale(${dynamicVolumeScale})` }}
          className={`group relative z-10 w-44 h-44 sm:w-52 sm:h-52 border-[6px] border-black flex items-center justify-center hard-shadow active:translate-x-[4px] active:translate-y-[4px] active:shadow-none transition-all cursor-pointer ${
            isRecording
              ? 'bg-red-600 text-white'
              : isProcessing
              ? `${currentThemeBg} text-black cursor-wait`
              : `${currentThemeBg} text-black hover:-translate-x-[3px] hover:-translate-y-[3px] hover:shadow-[16px_16px_0px_0px_#000000]`
          }`}
        >
          {isProcessing ? (
            <div className="flex flex-col items-center justify-center">
              <Loader2 className="w-16 h-16 animate-spin text-black stroke-[3]" />
              <span className="font-mono text-[11px] font-black tracking-widest uppercase bg-black text-white px-2 py-0.5 mt-2 border border-white">
                SYNTHESIZING
              </span>
            </div>
          ) : isRecording ? (
            <div className="flex flex-col items-center justify-center">
              <Square className="w-14 h-14 sm:w-16 sm:h-16 fill-current text-white mb-2 stroke-[3]" />
              <span className="font-mono text-xs font-black tracking-widest uppercase bg-black text-white px-3 py-1 border-2 border-white">
                STOP_REC
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center">
              {/* Geometric Sharp Microphone SVG */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="square"
                strokeLinejoin="miter"
                className="w-16 h-16 sm:w-20 sm:h-20 text-black group-hover:scale-110 group-hover:-rotate-2 transition-all"
              >
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
              <span className={`font-mono text-xs font-black tracking-widest uppercase bg-black ${currentThemeText} px-2.5 py-0.5 mt-2 border border-black`}>
                CLICK_TO_TALK
              </span>
            </div>
          )}
        </button>
      </div>

      {/* Geometric Status Bar & Live Telemetry */}
      {(isRecording || isProcessing) && (
        <div className="mt-8 flex flex-col items-center gap-2 relative z-10 animate-in fade-in duration-150">
          {isRecording ? (
            <div className="flex items-center gap-3 bg-black text-white px-4 py-2 border-4 border-black hard-shadow-sm font-mono text-xs sm:text-sm font-black uppercase tracking-wider">
              <span className="w-2.5 h-2.5 bg-red-600 rounded-full animate-ping" />
              <span>RECORDING // {formatTime(recordingDuration)}</span>
              <span className="text-[10px] text-yellow-400 border-l border-white/40 pl-2">
                {approxDb > -50 ? `${approxDb} dB` : 'AUDIO_IN'}
              </span>
            </div>
          ) : (
            <div className={`flex items-center gap-3 ${currentThemeBg} text-black px-4 py-2 border-4 border-black hard-shadow-sm font-mono text-xs sm:text-sm font-black uppercase tracking-wider`}>
              <Loader2 className="w-4 h-4 animate-spin stroke-[3]" />
              <span>TRANSCRIBING_STREAM...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
