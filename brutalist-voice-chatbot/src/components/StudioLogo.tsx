import React from 'react';

interface StudioLogoProps {
  className?: string;
  height?: number | string;
}

export const StudioLogo: React.FC<StudioLogoProps> = ({
  className = '',
  height = 46,
}) => {
  return (
    <div
      className={`inline-flex items-center gap-2 group select-none ${className}`}
      title="2:47 PM STUDIO"
    >
      <div className="relative flex flex-col items-center justify-center p-1.5 px-3 bg-[#FFDD00] border-2 border-black hard-shadow-xs transform -rotate-1 hover:rotate-0 transition-transform">
        <div className="text-[#000000] font-black leading-none tracking-tight flex items-baseline gap-0.5" style={{ fontFamily: '"Comic Sans MS", "Chalkboard SE", "Arial Black", sans-serif' }}>
          <span className="text-xl sm:text-2xl font-black">2:47</span>
          <span className="text-sm sm:text-base font-black uppercase ml-0.5">PM</span>
        </div>
        <div
          className="text-[#000000] text-[10px] sm:text-xs font-black tracking-[0.2em] uppercase leading-none mt-0.5"
          style={{ fontFamily: '"Comic Sans MS", "Chalkboard SE", "Arial Black", sans-serif' }}
        >
          STUDIO
        </div>
      </div>
    </div>
  );
};

