import React, { useEffect, useState, useRef } from 'react';
import { CursorMode, ThemeColor } from '../types';
import { audioFX } from '../utils/audioFx';

interface BrutalistCursorFxProps {
  mode: CursorMode;
  themeColor: ThemeColor;
  onToggleMode?: () => void;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  opacity: number;
  rotation: number;
}

interface ClickRipple {
  id: number;
  x: number;
  y: number;
  size: number;
  opacity: number;
  color: string;
}

export const BrutalistCursorFx: React.FC<BrutalistCursorFxProps> = ({
  mode,
  themeColor,
}) => {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [velocity, setVelocity] = useState({ vx: 0, vy: 0, speed: 0 });
  const [angle, setAngle] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [ripples, setRipples] = useState<ClickRipple[]>([]);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [trailPositions, setTrailPositions] = useState<{ x: number; y: number; time: number }[]>([]);
  const [toggleAlert, setToggleAlert] = useState<string | null>(null);

  const lastPosRef = useRef({ x: 0, y: 0, time: performance.now() });
  const animRef = useRef<number | null>(null);

  const themeColors: Record<ThemeColor, string> = {
    yellow: '#FACC15',
    lime: '#A3E635',
    cyan: '#22D3EE',
    orange: '#F97316',
    white: '#FFFFFF',
  };

  const activeColor = themeColors[themeColor] || '#FACC15';

  // Trigger brief HUD notification when cursor mode changes
  useEffect(() => {
    if (mode === 'off') {
      setToggleAlert('CURSOR: DEFAULT SYSTEM');
    } else if (mode === 'arrow') {
      setToggleAlert('CURSOR: BRUTAL ARROW + TELEMETRY');
    } else if (mode === 'crosshair') {
      setToggleAlert('CURSOR: TACTICAL CROSSHAIR');
    } else if (mode === 'radar') {
      setToggleAlert('CURSOR: SONAR RADAR SWEEP');
    } else if (mode === 'trail') {
      setToggleAlert('CURSOR: GEOMETRIC TRAIL');
    }

    const timer = setTimeout(() => setToggleAlert(null), 1800);
    return () => clearTimeout(timer);
  }, [mode]);

  // Track mouse movements & hover targets
  useEffect(() => {
    if (mode === 'off') return;

    const handleMouseMove = (e: MouseEvent) => {
      const now = performance.now();
      const dt = Math.max(1, now - lastPosRef.current.time);
      const dx = e.clientX - lastPosRef.current.x;
      const dy = e.clientY - lastPosRef.current.y;
      const speed = Math.sqrt(dx * dx + dy * dy) / dt;

      // Calculate movement angle
      let currentAngle = angle;
      if (Math.hypot(dx, dy) > 2) {
        currentAngle = (Math.atan2(dy, dx) * 180) / Math.PI + 90;
      }

      setPos({ x: e.clientX, y: e.clientY });
      setVelocity({ vx: dx, vy: dy, speed });
      setAngle(currentAngle);

      lastPosRef.current = { x: e.clientX, y: e.clientY, time: now };

      // Add to trail
      setTrailPositions((prev) => [
        { x: e.clientX, y: e.clientY, time: now },
        ...prev.slice(0, 10),
      ]);

      // Check hover on interactive elements
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'BUTTON' ||
          target.tagName === 'A' ||
          target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.closest('button') ||
          target.closest('a') ||
          target.getAttribute('role') === 'button' ||
          target.classList.contains('cursor-pointer'))
      ) {
        setIsHovered(true);
      } else {
        setIsHovered(false);
      }

      // If trail mode is active, spawn particles on move
      if (mode === 'trail' && speed > 0.3 && Math.random() > 0.4) {
        const newParticle: Particle = {
          id: Math.random(),
          x: e.clientX + (Math.random() - 0.5) * 8,
          y: e.clientY + (Math.random() - 0.5) * 8,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          size: Math.random() * 8 + 4,
          color: activeColor,
          opacity: 1,
          rotation: Math.random() * 360,
        };
        setParticles((prev) => [...prev.slice(-20), newParticle]);
      }
    };

    const handleMouseDown = (e: MouseEvent) => {
      setIsMouseDown(true);
      audioFX.playCursorClick();

      // Spawn click shockwave ripples
      const newRipple: ClickRipple = {
        id: Date.now() + Math.random(),
        x: e.clientX,
        y: e.clientY,
        size: 10,
        opacity: 1,
        color: activeColor,
      };
      setRipples((prev) => [...prev.slice(-6), newRipple]);

      // Spawn click burst particles
      const burstCount = 8;
      const burst: Particle[] = [];
      for (let i = 0; i < burstCount; i++) {
        const theta = (i / burstCount) * 2 * Math.PI;
        const vel = Math.random() * 4 + 2;
        burst.push({
          id: Math.random() + Date.now(),
          x: e.clientX,
          y: e.clientY,
          vx: Math.cos(theta) * vel,
          vy: Math.sin(theta) * vel,
          size: Math.random() * 8 + 4,
          color: i % 2 === 0 ? activeColor : '#000000',
          opacity: 1,
          rotation: Math.random() * 360,
        });
      }
      setParticles((prev) => [...prev.slice(-25), ...burst]);
    };

    const handleMouseUp = () => {
      setIsMouseDown(false);
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [mode, activeColor, angle]);

  // Particle & Ripple Animation loop
  useEffect(() => {
    if (mode === 'off') return;

    const updateFX = () => {
      // Update Ripples
      setRipples((prev) =>
        prev
          .map((r) => ({
            ...r,
            size: r.size + 4,
            opacity: r.opacity - 0.04,
          }))
          .filter((r) => r.opacity > 0)
      );

      // Update Particles
      setParticles((prev) =>
        prev
          .map((p) => ({
            ...p,
            x: p.x + p.vx,
            y: p.y + p.vy,
            opacity: p.opacity - 0.03,
            rotation: p.rotation + 4,
          }))
          .filter((p) => p.opacity > 0)
      );

      animRef.current = requestAnimationFrame(updateFX);
    };

    animRef.current = requestAnimationFrame(updateFX);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [mode]);

  if (mode === 'off') {
    return (
      <>
        {toggleAlert && (
          <div className="fixed top-20 right-6 z-50 pointer-events-none animate-in slide-in-from-top-2 fade-in duration-200">
            <div className="bg-black text-white font-mono text-xs font-black uppercase px-3 py-1.5 border-2 border-white shadow-[4px_4px_0px_0px_#000000] flex items-center gap-2">
              <div className="w-2 h-2 bg-neutral-400" />
              {toggleAlert}
            </div>
          </div>
        )}
      </>
    );
  }

  return (
    <div className="fixed inset-0 pointer-events-none z-[9999] overflow-hidden select-none">
      {/* Dynamic HUD Alert on Toggle */}
      {toggleAlert && (
        <div className="fixed top-20 right-6 z-50 animate-in slide-in-from-top-2 fade-in duration-200">
          <div
            className="bg-black text-white font-mono text-xs font-black uppercase px-3 py-1.5 border-2 border-white shadow-[4px_4px_0px_0px_#000000] flex items-center gap-2"
          >
            <div
              className="w-2.5 h-2.5 animate-ping"
              style={{ backgroundColor: activeColor }}
            />
            {toggleAlert}
          </div>
        </div>
      )}

      {/* Screen Laser Guide Lines (Crosshair & Radar Mode) */}
      {(mode === 'crosshair' || mode === 'radar') && (
        <>
          <div
            className="absolute left-0 right-0 h-[1px] bg-black/20 pointer-events-none"
            style={{ top: `${pos.y}px` }}
          />
          <div
            className="absolute top-0 bottom-0 w-[1px] bg-black/20 pointer-events-none"
            style={{ left: `${pos.x}px` }}
          />
        </>
      )}

      {/* Click Expanding Shockwaves */}
      {ripples.map((ripple) => (
        <div
          key={ripple.id}
          className="absolute transform -translate-x-1/2 -translate-y-1/2 border-2 border-black"
          style={{
            left: `${ripple.x}px`,
            top: `${ripple.y}px`,
            width: `${ripple.size}px`,
            height: `${ripple.size}px`,
            opacity: ripple.opacity,
            backgroundColor: ripple.color,
            boxShadow: '2px 2px 0px 0px #000000',
          }}
        />
      ))}

      {/* Flying Particle Spark Trail */}
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute border border-black transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${particle.x}px`,
            top: `${particle.y}px`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            backgroundColor: particle.color,
            opacity: particle.opacity,
            transform: `translate(-50%, -50%) rotate(${particle.rotation}deg)`,
            boxShadow: '1px 1px 0px 0px #000000',
          }}
        />
      ))}

      {/* Geometric Ghost Trails */}
      {trailPositions.slice(1, 5).map((t, idx) => (
        <div
          key={idx}
          className="absolute border border-black/40 bg-black/10 transform -translate-x-1/2 -translate-y-1/2 transition-all duration-75"
          style={{
            left: `${t.x}px`,
            top: `${t.y}px`,
            width: `${12 - idx * 2}px`,
            height: `${12 - idx * 2}px`,
            opacity: 0.4 - idx * 0.08,
          }}
        />
      ))}

      {/* 1. BRUTAL ARROW MODE */}
      {mode === 'arrow' && (
        <div
          className="absolute transition-transform duration-75 ease-out"
          style={{
            left: `${pos.x}px`,
            top: `${pos.y}px`,
            transform: `translate(-4px, -4px) scale(${isMouseDown ? 0.85 : isHovered ? 1.25 : 1})`,
          }}
        >
          {/* Custom Neo-Brutalist Sharp Arrow SVG */}
          <svg
            width="32"
            height="32"
            viewBox="0 0 32 32"
            className="drop-shadow-[3px_3px_0px_#000000] overflow-visible"
          >
            {/* Arrow Polygon */}
            <polygon
              points="0,0 28,14 16,18 24,30 18,32 10,20 0,28"
              fill={activeColor}
              stroke="#000000"
              strokeWidth="2.5"
              strokeLinejoin="miter"
            />
            {/* Inner Accent Line */}
            <line x1="4" y1="6" x2="16" y2="16" stroke="#000000" strokeWidth="2" />
          </svg>

          {/* Telemetry Coordinate Box */}
          <div
            className={`absolute left-7 top-4 bg-black text-white font-mono text-[9px] font-bold px-1 py-0.2 border border-white whitespace-nowrap shadow-[2px_2px_0px_0px_#000000] transition-opacity duration-150 ${
              velocity.speed > 0.5 || isHovered ? 'opacity-100' : 'opacity-0'
            }`}
          >
            {isHovered ? '[TARGET_LOCKED]' : `X:${Math.round(pos.x)} Y:${Math.round(pos.y)}`}
          </div>

          {/* Hover Bracket Lock Box */}
          {isHovered && (
            <div
              className="absolute -inset-2 border-2 border-dashed border-black animate-spin"
              style={{ animationDuration: '6s' }}
            />
          )}
        </div>
      )}

      {/* 2. TACTICAL CROSSHAIR MODE */}
      {mode === 'crosshair' && (
        <div
          className="absolute transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${pos.x}px`,
            top: `${pos.y}px`,
            transform: `translate(-50%, -50%) scale(${isMouseDown ? 0.8 : isHovered ? 1.3 : 1})`,
          }}
        >
          {/* Tactical Target SVG */}
          <svg width="48" height="48" viewBox="0 0 48 48" className="overflow-visible">
            {/* Outer Box */}
            <rect
              x="8"
              y="8"
              width="32"
              height="32"
              fill="none"
              stroke="#000000"
              strokeWidth="2"
              strokeDasharray="4 2"
            />
            {/* Center Circle */}
            <circle cx="24" cy="24" r="6" fill={activeColor} stroke="#000000" strokeWidth="2" />
            {/* Cross Lines */}
            <line x1="2" y1="24" x2="16" y2="24" stroke="#000000" strokeWidth="2.5" />
            <line x1="32" y1="24" x2="46" y2="24" stroke="#000000" strokeWidth="2.5" />
            <line x1="24" y1="2" x2="24" y2="16" stroke="#000000" strokeWidth="2.5" />
            <line x1="24" y1="32" x2="24" y2="46" stroke="#000000" strokeWidth="2.5" />
          </svg>

          {/* Tactical Readout */}
          <div className="absolute left-10 top-0 bg-black text-white font-mono text-[9px] font-black px-1.5 py-0.5 border border-white whitespace-nowrap shadow-[2px_2px_0px_0px_#000000]">
            RAD_{isHovered ? 'ENGAGED' : 'ACTIVE'} &bull; {pos.x},{pos.y}
          </div>
        </div>
      )}

      {/* 3. SONAR RADAR SWEEP MODE */}
      {mode === 'radar' && (
        <div
          className="absolute transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${pos.x}px`,
            top: `${pos.y}px`,
          }}
        >
          {/* Radar Ring Concentrics */}
          <div className="relative w-20 h-20 flex items-center justify-center">
            {/* Outer Ring */}
            <div className="absolute inset-0 rounded-full border-2 border-black bg-white/20 shadow-[2px_2px_0px_0px_#000000]" />
            <div className="absolute inset-2 rounded-full border border-black/40" />
            <div className="absolute inset-5 rounded-full border border-black/60" />

            {/* Rotating Sonar Beam */}
            <div
              className="absolute inset-0 rounded-full animate-spin"
              style={{
                background: `conic-gradient(from 0deg, transparent 0deg, ${activeColor} 30deg, transparent 60deg)`,
                animationDuration: '2s',
              }}
            />

            {/* Center Pointer */}
            <div
              className="w-3 h-3 border-2 border-black z-10"
              style={{ backgroundColor: activeColor }}
            />
          </div>

          <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 bg-black text-white font-mono text-[8px] font-black px-1 border border-white whitespace-nowrap">
            SONAR_SCAN // {pos.x}x{pos.y}
          </div>
        </div>
      )}

      {/* 4. GEOMETRIC TRAIL MODE */}
      {mode === 'trail' && (
        <div
          className="absolute transform -translate-x-1/2 -translate-y-1/2"
          style={{
            left: `${pos.x}px`,
            top: `${pos.y}px`,
            transform: `translate(-50%, -50%) scale(${isMouseDown ? 0.75 : isHovered ? 1.4 : 1})`,
          }}
        >
          {/* Geometric Diamond Pointer */}
          <div
            className="w-5 h-5 border-2 border-black transform rotate-45 shadow-[2px_2px_0px_0px_#000000] flex items-center justify-center"
            style={{ backgroundColor: activeColor }}
          >
            <div className="w-1.5 h-1.5 bg-black" />
          </div>

          {isHovered && (
            <div className="absolute left-6 top-0 bg-black text-white font-mono text-[9px] font-black px-1.5 border border-white whitespace-nowrap">
              SPARKLE_LOCK
            </div>
          )}
        </div>
      )}
    </div>
  );
};
