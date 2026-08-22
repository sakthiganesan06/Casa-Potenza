import React from 'react';
import { ThemeColor } from '../types';

interface ScoreSmallBarProps {
  score?: {
    value: number; // 0 - 100
    tier: 'P50' | 'P70' | 'P100';
  };
  themeColor?: ThemeColor;
  compact?: boolean;
}

export const ScoreSmallBar: React.FC<ScoreSmallBarProps> = ({
  score = { value: 85, tier: 'P70' },
  themeColor = 'yellow',
  compact = false,
}) => {
  const themeBgMap: Record<ThemeColor, string> = {
    yellow: 'bg-yellow-400',
    lime: 'bg-lime-400',
    cyan: 'bg-cyan-400',
    orange: 'bg-orange-500',
    white: 'bg-neutral-300',
  };

  const currentThemeBg = themeBgMap[themeColor] || 'bg-yellow-400';
  const val = Math.min(100, Math.max(0, score.value));
  const activeTier = score.tier || (val >= 90 ? 'P100' : val >= 70 ? 'P70' : 'P50');

  if (compact) {
    return (
      <div className="flex items-center gap-2 font-mono text-[10px] select-none">
        <div className="flex items-center border border-black bg-white">
          <span
            className={`px-1 py-0.5 font-black border-r border-black ${
              activeTier === 'P50' ? currentThemeBg : 'text-black/40'
            }`}
          >
            P50
          </span>
          <span
            className={`px-1 py-0.5 font-black border-r border-black ${
              activeTier === 'P70' ? currentThemeBg : 'text-black/40'
            }`}
          >
            P70
          </span>
          <span
            className={`px-1 py-0.5 font-black ${
              activeTier === 'P100' ? currentThemeBg : 'text-black/40'
            }`}
          >
            P100
          </span>
        </div>
        <div className="w-16 h-2 bg-neutral-200 border border-black relative overflow-hidden">
          <div
            className={`h-full ${currentThemeBg} transition-all duration-300`}
            style={{ width: `${val}%` }}
          />
        </div>
        <span className="font-bold text-black">{val}%</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 p-2.5 bg-white border-2 border-black hard-shadow-xs font-mono select-none">
      <div className="flex items-center justify-between text-[11px] font-black uppercase">
        <div className="flex items-center gap-2">
          <span className="text-black/70">CONFIDENCE_SCORE:</span>
          <span className={`px-1.5 py-0.2 border border-black ${currentThemeBg} text-black font-black`}>
            {val}% [{activeTier}]
          </span>
        </div>
        <div className="flex items-center gap-1 text-[10px]">
          <span
            className={`px-1.5 py-0.5 border border-black font-black transition-colors ${
              activeTier === 'P50' ? `${currentThemeBg} text-black` : 'bg-neutral-100 text-black/40'
            }`}
          >
            P50
          </span>
          <span
            className={`px-1.5 py-0.5 border border-black font-black transition-colors ${
              activeTier === 'P70' ? `${currentThemeBg} text-black` : 'bg-neutral-100 text-black/40'
            }`}
          >
            P70
          </span>
          <span
            className={`px-1.5 py-0.5 border border-black font-black transition-colors ${
              activeTier === 'P100' ? `${currentThemeBg} text-black` : 'bg-neutral-100 text-black/40'
            }`}
          >
            P100
          </span>
        </div>
      </div>

      {/* Small Brutalist Metric Progress Bar with P50 / P70 / P100 Target Ticks */}
      <div className="relative w-full h-3.5 bg-neutral-100 border-2 border-black">
        {/* Filled Progress Bar */}
        <div
          className={`h-full ${currentThemeBg} border-r-2 border-black transition-all duration-500 ease-out`}
          style={{ width: `${val}%` }}
        />

        {/* Ticks for P50, P70, P100 */}
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-black/40 pointer-events-none"
          style={{ left: '50%' }}
          title="P50 Benchmark"
        />
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-black/60 pointer-events-none"
          style={{ left: '70%' }}
          title="P70 Benchmark"
        />
        <div
          className="absolute top-0 bottom-0 w-[2px] bg-black pointer-events-none"
          style={{ left: '100%', transform: 'translateX(-2px)' }}
          title="P100 Benchmark"
        />
      </div>

      {/* Benchmark Tick Labels Underneath */}
      <div className="flex justify-between text-[9px] font-bold text-black/50 tracking-wider">
        <span>0</span>
        <span className="pl-4">50 (P50)</span>
        <span className="pl-2">70 (P70)</span>
        <span>100 (P100)</span>
      </div>
    </div>
  );
};
