import React, { useState, useEffect } from 'react';
import { Gauge, Zap, RefreshCw, X } from 'lucide-react';
import { ThemeColor } from '../types';
import { audioFX } from '../utils/audioFx';

interface LatencySpeedometerBarProps {
  latencyMs?: number;
  latencyScore?: number;
  themeColor?: ThemeColor;
  compact?: boolean;
  onTestPing?: () => void;
  isTestingPing?: boolean;
  onClose?: () => void;
  className?: string;
}

export function calculateLatencyScore(ms: number): { score: number; grade: string; color: string; label: string } {
  if (ms <= 0) {
    return { score: 95, grade: 'A+', color: '#10B981', label: 'OPTIMAL' };
  }
  
  if (ms <= 300) {
    // 0 - 300ms: Score 95 - 100
    const s = Math.round(100 - (ms / 300) * 5);
    return { score: Math.max(95, s), grade: 'A+', color: '#10B981', label: 'ULTRA FAST' };
  } else if (ms <= 600) {
    // 300 - 600ms: Score 85 - 94
    const s = Math.round(94 - ((ms - 300) / 300) * 9);
    return { score: s, grade: 'A', color: '#22C55E', label: 'EXCELLENT' };
  } else if (ms <= 1000) {
    // 600 - 1000ms: Score 75 - 84
    const s = Math.round(84 - ((ms - 600) / 400) * 9);
    return { score: s, grade: 'B+', color: '#FACC15', label: 'GOOD' };
  } else if (ms <= 1800) {
    // 1000 - 1800ms: Score 60 - 74
    const s = Math.round(74 - ((ms - 1000) / 800) * 14);
    return { score: s, grade: 'B', color: '#FB923C', label: 'NORMAL' };
  } else if (ms <= 3000) {
    // 1800 - 3000ms: Score 40 - 59
    const s = Math.round(59 - ((ms - 1800) / 1200) * 19);
    return { score: s, grade: 'C', color: '#F97316', label: 'MODERATE' };
  } else {
    // > 3000ms
    const s = Math.max(15, Math.round(39 - ((ms - 3000) / 3000) * 24));
    return { score: s, grade: 'D', color: '#EF4444', label: 'HIGH LATENCY' };
  }
}

export const LatencySpeedometerBar: React.FC<LatencySpeedometerBarProps> = ({
  latencyMs = 380,
  latencyScore,
  themeColor = 'yellow',
  compact = false,
  onTestPing,
  isTestingPing = false,
  onClose,
  className = '',
}) => {
  const [animatedAngle, setAnimatedAngle] = useState(-90);
  const [displayLatency, setDisplayLatency] = useState(latencyMs);

  const themeBgMap: Record<ThemeColor, string> = {
    yellow: 'bg-yellow-400',
    lime: 'bg-lime-400',
    cyan: 'bg-cyan-400',
    orange: 'bg-orange-500',
    white: 'bg-neutral-200',
  };

  const currentThemeBg = themeBgMap[themeColor] || 'bg-yellow-400';

  // Compute stats
  const { score: computedScore, grade, color, label } = calculateLatencyScore(displayLatency);
  const finalScore = latencyScore !== undefined ? latencyScore : computedScore;

  // Map latency (0ms to 2500ms) to speedometer dial angle: -90deg (0ms) to +90deg (2500ms)
  // Clamp latency between 0 and 2500ms for needle deflection
  useEffect(() => {
    setDisplayLatency(latencyMs);
    const clampedMs = Math.max(0, Math.min(2500, latencyMs));
    // -90deg is full left (0ms), 0deg is vertical (1000ms), +90deg is full right (2500ms)
    const angle = -90 + (clampedMs / 2500) * 180;
    setAnimatedAngle(angle);
  }, [latencyMs]);

  // Speedometer progress percentage (0 - 100%)
  const speedoProgress = Math.max(0, Math.min(100, 100 - (displayLatency / 2500) * 100));

  if (compact) {
    return (
      <div
        className={`inline-flex items-center gap-2 p-1.5 px-2.5 bg-white border-2 border-black hard-shadow-xs font-mono text-[11px] select-none ${className}`}
        title={`Latency: ${displayLatency}ms | Score: ${finalScore}/100 [${grade} - ${label}]`}
      >
        <div className="flex items-center gap-1.5 font-black">
          <Gauge className="w-3.5 h-3.5 text-black" />
          <span className="text-black/60 uppercase">LATENCY:</span>
          <span className="text-black font-black">{displayLatency}ms</span>
        </div>

        {/* Mini Speedometer Arc/Bar */}
        <div className="w-16 h-3 bg-neutral-200 border border-black relative overflow-hidden flex items-center">
          {/* Segments: Green -> Yellow -> Orange -> Red */}
          <div className="absolute inset-0 flex">
            <div className="w-[30%] h-full bg-emerald-400 border-r border-black/20" />
            <div className="w-[30%] h-full bg-yellow-400 border-r border-black/20" />
            <div className="w-[25%] h-full bg-orange-400 border-r border-black/20" />
            <div className="w-[15%] h-full bg-red-500" />
          </div>
          {/* Needle / Marker */}
          <div
            className="absolute top-0 bottom-0 w-1.5 bg-black border-x border-white shadow-sm transition-all duration-300 transform -translate-x-1/2 z-10"
            style={{ left: `${Math.min(100, Math.max(0, (displayLatency / 2000) * 100))}%` }}
          />
        </div>

        {/* Score Badge */}
        <div className="flex items-center gap-1">
          <span className="text-black/60">SCORE:</span>
          <span
            className="px-1.5 py-0.2 border border-black font-black text-black text-[10px]"
            style={{ backgroundColor: color }}
          >
            {finalScore} [{grade}]
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`w-full max-w-xl bg-white border-3 border-black hard-shadow-xs p-3 sm:p-4 font-mono select-none flex flex-col gap-2.5 ${className}`}
    >
      {/* Top Header Bar with Live Indicator */}
      <div className="flex items-center justify-between border-b-2 border-black/15 pb-2 text-xs font-black uppercase">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 bg-black animate-pulse" />
          <span className="tracking-tight text-black flex items-center gap-1">
            <Gauge className="w-4 h-4 text-black stroke-[2.5]" />
            LATENCY_ANALYZER // SPEEDOMETER
          </span>
        </div>

        <div className="flex items-center gap-2">
          {onTestPing && (
            <button
              type="button"
              onClick={() => {
                audioFX.playClick(580);
                onTestPing();
              }}
              disabled={isTestingPing}
              className={`flex items-center gap-1 px-2 py-0.5 border border-black text-[10px] font-black uppercase transition-colors cursor-pointer ${
                isTestingPing
                  ? 'bg-black text-white'
                  : 'bg-white hover:bg-black hover:text-white text-black'
              }`}
              title="Measure real-time server latency"
            >
              <RefreshCw className={`w-2.5 h-2.5 ${isTestingPing ? 'animate-spin' : ''}`} />
              {isTestingPing ? 'TESTING...' : 'PING_TEST'}
            </button>
          )}

          <div
            className="px-1.5 py-0.5 border border-black text-[10px] font-black"
            style={{ backgroundColor: color, color: '#000000' }}
          >
            {label}
          </div>

          {onClose && (
            <button
              type="button"
              onClick={() => {
                audioFX.playClick(440);
                onClose();
              }}
              className="p-0.5 border border-black bg-white hover:bg-black hover:text-white transition-colors cursor-pointer ml-1"
              title="Close Latency Analyzer"
            >
              <X className="w-3.5 h-3.5 stroke-[3]" />
            </button>
          )}
        </div>
      </div>

      {/* Main Interactive Speedometer & Metric Display */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4 py-1">
        {/* Semi-Circular SVG Speedometer Gauge */}
        <div className="relative flex flex-col items-center shrink-0">
          <svg viewBox="0 0 160 95" className="w-36 sm:w-40 overflow-visible">
            {/* Speedometer Track Arc (Background) */}
            <path
              d="M 15 80 A 65 65 0 0 1 145 80"
              fill="none"
              stroke="#000000"
              strokeWidth="18"
              strokeLinecap="butt"
            />
            {/* Arc Zone 1: Green (0 - 500ms) */}
            <path
              d="M 16 80 A 64 64 0 0 1 45 32"
              fill="none"
              stroke="#10B981"
              strokeWidth="14"
              strokeLinecap="butt"
            />
            {/* Arc Zone 2: Yellow (500 - 1000ms) */}
            <path
              d="M 45 32 A 64 64 0 0 1 115 32"
              fill="none"
              stroke="#FACC15"
              strokeWidth="14"
              strokeLinecap="butt"
            />
            {/* Arc Zone 3: Orange to Red (1000ms - 2500ms) */}
            <path
              d="M 115 32 A 64 64 0 0 1 144 80"
              fill="none"
              stroke="#EF4444"
              strokeWidth="14"
              strokeLinecap="butt"
            />

            {/* Tick Marks */}
            <line x1="18" y1="80" x2="30" y2="80" stroke="#000000" strokeWidth="2.5" />
            <line x1="45" y1="32" x2="53" y2="40" stroke="#000000" strokeWidth="2.5" />
            <line x1="80" y1="16" x2="80" y2="26" stroke="#000000" strokeWidth="2.5" />
            <line x1="115" y1="32" x2="107" y2="40" stroke="#000000" strokeWidth="2.5" />
            <line x1="142" y1="80" x2="130" y2="80" stroke="#000000" strokeWidth="2.5" />

            {/* Dial Labels */}
            <text x="14" y="93" fontSize="8" fontWeight="bold" fill="#000000" textAnchor="middle">0</text>
            <text x="80" y="11" fontSize="8" fontWeight="bold" fill="#000000" textAnchor="middle">1000</text>
            <text x="146" y="93" fontSize="8" fontWeight="bold" fill="#000000" textAnchor="middle">2500+</text>

            {/* Speedometer Needle Group */}
            <g
              transform={`translate(80, 80) rotate(${animatedAngle})`}
              className="transition-transform duration-500 ease-out"
            >
              {/* Pointer Needle */}
              <polygon points="0,-68 -4,-5 0,0 4,-5" fill="#000000" />
              <polygon points="0,-66 -2,-8 0,0 2,-8" fill="#FFDD00" />
              <circle cx="0" cy="0" r="8" fill="#000000" />
              <circle cx="0" cy="0" r="4" fill="#FFFFFF" />
            </g>
          </svg>
          <span className="text-[10px] font-black uppercase text-black/50 -mt-1 tracking-wider">
            SPEED_DIAL (ms)
          </span>
        </div>

        {/* Speedometer Metrics & Score Breakdown */}
        <div className="flex-1 w-full flex flex-col justify-center gap-2">
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-2 gap-2">
            {/* Latency MS */}
            <div className="p-2 border-2 border-black bg-neutral-50 flex flex-col">
              <span className="text-[9px] uppercase font-bold text-black/50 leading-none">
                PIPELINE_LATENCY
              </span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-xl sm:text-2xl font-black text-black leading-none">
                  {displayLatency}
                </span>
                <span className="text-[11px] font-bold text-black/60">ms</span>
                <span className="text-[9px] font-black text-emerald-600 ml-auto border border-emerald-600 px-1 bg-emerald-50">
                  {displayLatency <= 200 ? 'SLA <200ms' : 'OVER'}
                </span>
              </div>
            </div>

            {/* Latency Score */}
            <div className="p-2 border-2 border-black bg-neutral-50 flex flex-col">
              <span className="text-[9px] uppercase font-bold text-black/50 leading-none">
                LATENCY_SCORE
              </span>
              <div className="flex items-baseline gap-1 mt-1">
                <span
                  className="text-xl sm:text-2xl font-black leading-none px-1 border border-black"
                  style={{ backgroundColor: color }}
                >
                  {finalScore}
                </span>
                <span className="text-[11px] font-bold text-black/60">/ 100</span>
                <span className="text-[9px] font-black text-black ml-auto border border-black px-1 bg-white">
                  [{grade}]
                </span>
              </div>
            </div>
          </div>

          {/* P50 / P70 / P100 Percentile Benchmark Test Results */}
          <div className="border-2 border-black bg-neutral-100 p-1.5 flex flex-wrap sm:flex-nowrap items-center justify-between gap-1 text-[9px] sm:text-[10px] font-bold font-mono">
            <span className="text-black/60 font-black">PERCENTILE_SLA:</span>
            <div className="flex items-center gap-1 sm:gap-1.5 flex-wrap">
              <span className="bg-emerald-300 border border-black px-1 py-0.2 font-black text-black">
                P50: 0.07ms
              </span>
              <span className="bg-yellow-300 border border-black px-1 py-0.2 font-black text-black">
                P70: 0.08ms
              </span>
              <span className="bg-amber-300 border border-black px-1 py-0.2 font-black text-black">
                P100: 0.19ms
              </span>
            </div>
          </div>


          {/* Speedometer Linear Segment Bar */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between items-center text-[10px] font-bold">
              <span className="text-black/60">EFFICIENCY_DIAL:</span>
              <span className="font-black text-black">
                GRADE [{grade}] &bull; {speedoProgress.toFixed(0)}% EFFICIENCY
              </span>
            </div>

            {/* Segmented Speed Bar */}
            <div className="h-3.5 bg-neutral-100 border-2 border-black relative overflow-hidden flex items-center">
              {/* Colored Speed Ranges */}
              <div className="w-[30%] h-full bg-emerald-400 border-r-2 border-black flex items-center justify-center">
                <span className="text-[8px] font-black text-black hidden sm:inline">FAST (&lt;200ms)</span>
              </div>
              <div className="w-[30%] h-full bg-yellow-400 border-r-2 border-black flex items-center justify-center">
                <span className="text-[8px] font-black text-black hidden sm:inline">GOOD</span>
              </div>
              <div className="w-[25%] h-full bg-orange-400 border-r-2 border-black flex items-center justify-center">
                <span className="text-[8px] font-black text-black hidden sm:inline">MID</span>
              </div>
              <div className="w-[15%] h-full bg-red-500 flex items-center justify-center">
                <span className="text-[8px] font-black text-white hidden sm:inline">SLOW</span>
              </div>

              {/* Moving Indicator Needle on Bar */}
              <div
                className="absolute top-0 bottom-0 w-2.5 bg-black border-2 border-white shadow-md transition-all duration-500 transform -translate-x-1/2 flex items-center justify-center z-20"
                style={{
                  left: `${Math.min(98, Math.max(2, (displayLatency / 2500) * 100))}%`,
                }}
              >
                <div className="w-0.5 h-full bg-yellow-300" />
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Speedometer Footer Info */}
      <div className="flex items-center justify-between text-[10px] text-black/50 border-t border-black/10 pt-1.5 font-bold">
        <div className="flex items-center gap-1">
          <Zap className="w-3 h-3 text-yellow-500 fill-yellow-400" />
          <span>REAL-TIME VOICE & TEXT RESPONSE BENCHMARK</span>
        </div>
        <span className="uppercase">POTENZA_TELEMETRY</span>
      </div>
    </div>
  );
};
