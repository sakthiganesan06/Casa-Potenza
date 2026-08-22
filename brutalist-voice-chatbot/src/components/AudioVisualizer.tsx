import React, { useEffect, useRef } from 'react';
import { AudioVisualizerProps, VisualizerMode } from '../types';
import { RefreshCw, Activity, BarChart2, Radio, Gauge } from 'lucide-react';

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  analyser,
  isRecording,
  visualizerMode,
  onToggleMode,
  accentColor = '#FACC15',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!analyser || !isRecording || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    const bufferLength = analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);
    const timeData = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(timeData);

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, width, height);

      // Subtle background grid
      ctx.strokeStyle = '#EEEEEE';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < width; x += 20) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y < height; y += 15) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      if (visualizerMode === 'bars') {
        // Mode 1: Heavy Brutalist Bars
        const barCount = 32;
        const totalSpacing = 2;
        const barWidth = (width - (barCount - 1) * totalSpacing) / barCount;
        const step = Math.floor(bufferLength / barCount);

        for (let i = 0; i < barCount; i++) {
          let value = 0;
          for (let j = 0; j < step; j++) {
            value += freqData[i * step + j] || 0;
          }
          value = value / step;

          const normalizedHeight = Math.max(3, (value / 255) * (height - 6));
          const x = i * (barWidth + totalSpacing);
          const y = height - normalizedHeight;

          ctx.fillStyle = '#000000';
          ctx.fillRect(Math.round(x), Math.round(y), Math.round(barWidth), Math.round(normalizedHeight));

          if (value > 30) {
            ctx.fillStyle = accentColor;
            ctx.fillRect(Math.round(x), Math.round(y), Math.round(barWidth), 3);
          }
        }
      } else if (visualizerMode === 'wave') {
        // Mode 2: Realtime Oscilloscope Line
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#000000';
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = timeData[i] / 128.0;
          const y = (v * height) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Center zero line
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (visualizerMode === 'vu') {
        // Mode 3: Dual Digital VU Meter
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += freqData[i];
        }
        const avg = sum / bufferLength;
        const percent = Math.min(1, avg / 128);

        const meterHeight = (height - 16) / 2;
        const totalSegments = 24;
        const activeSegments = Math.round(percent * totalSegments);
        const segWidth = (width - 40) / totalSegments;

        ['CH_L', 'CH_R'].forEach((ch, idx) => {
          const y = 8 + idx * (meterHeight + 4);
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 9px monospace';
          ctx.fillText(ch, 6, y + meterHeight - 2);

          for (let s = 0; s < totalSegments; s++) {
            const segX = 36 + s * segWidth;
            if (s < activeSegments) {
              if (s > totalSegments * 0.8) {
                ctx.fillStyle = '#EF4444'; // Red clip zone
              } else if (s > totalSegments * 0.5) {
                ctx.fillStyle = accentColor; // Accent warning zone
              } else {
                ctx.fillStyle = '#000000'; // Normal zone
              }
            } else {
              ctx.fillStyle = '#E5E5E5';
            }
            ctx.fillRect(segX, y, segWidth - 2, meterHeight - 2);
          }
        });
      } else {
        // Mode 4: Circular Radar Waves
        const centerX = width / 2;
        const centerY = height / 2;
        const maxRadius = Math.min(width, height) / 2 - 4;

        let avg = 0;
        for (let i = 0; i < 30; i++) {
          avg += freqData[i];
        }
        avg = avg / 30;
        const pulse = (avg / 255) * maxRadius;

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, Math.max(6, pulse), 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = accentColor;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Crosshairs
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(centerX - 30, centerY);
        ctx.lineTo(centerX + 30, centerY);
        ctx.moveTo(centerX, centerY - 20);
        ctx.lineTo(centerX, centerY + 20);
        ctx.stroke();
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [analyser, isRecording, visualizerMode, accentColor]);

  if (!isRecording) return null;

  const modeIcons: Record<VisualizerMode, React.ReactNode> = {
    bars: <BarChart2 className="w-3 h-3" />,
    wave: <Activity className="w-3 h-3" />,
    vu: <Gauge className="w-3 h-3" />,
    circular: <Radio className="w-3 h-3" />,
  };

  return (
    <div className="w-full max-w-xs sm:max-w-sm mt-6 border-4 border-black bg-white p-1 hard-shadow-xs animate-in fade-in zoom-in-95 duration-100">
      <div className="flex items-center justify-between px-2 py-1 bg-black text-white font-mono text-[10px] uppercase font-black tracking-widest select-none">
        <div className="flex items-center gap-1.5">
          <span>{modeIcons[visualizerMode]}</span>
          <span>MODE: {visualizerMode}</span>
        </div>
        <div className="flex items-center gap-2">
          {onToggleMode && (
            <button
              type="button"
              onClick={onToggleMode}
              title="Switch Visualizer View"
              className="flex items-center gap-1 bg-white text-black hover:bg-yellow-400 px-1.5 py-0.5 text-[9px] font-bold uppercase transition-colors cursor-pointer"
            >
              <RefreshCw className="w-2.5 h-2.5" />
              CYCLE
            </button>
          )}
          <span className="flex items-center gap-1 text-red-500 font-bold">
            <span className="w-2 h-2 rounded-full bg-red-600 animate-ping inline-block" />
            LIVE
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={320}
        height={56}
        className="w-full h-14 bg-white block border-2 border-black mt-1"
      />
    </div>
  );
};
