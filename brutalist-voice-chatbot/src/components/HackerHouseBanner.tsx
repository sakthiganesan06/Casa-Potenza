import React from 'react';

interface HackerHouseBannerProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const HackerHouseBanner: React.FC<HackerHouseBannerProps> = ({
  className = '',
  size = 'md',
}) => {
  const containerSizes = {
    sm: 'py-1 px-3 text-sm sm:text-base',
    md: 'py-1.5 px-4 sm:px-6 text-base sm:text-xl md:text-2xl',
    lg: 'py-2.5 px-6 sm:px-8 text-xl sm:text-2xl md:text-3xl',
  };

  const devanagariSizes = {
    sm: 'text-[11px] sm:text-xs px-1 py-0.5 -mx-0.5',
    md: 'text-xs sm:text-sm px-1.5 py-0.5 -mx-1',
    lg: 'text-sm sm:text-base px-2 py-0.5 -mx-1.5',
  };

  return (
    <div
      className={`inline-flex items-center justify-center select-none ${className}`}
      title="HACKER गोवा HOUSE"
    >
      <div
        className={`relative flex items-center justify-center bg-[#054320] border-2 sm:border-[3px] border-black shadow-[4px_4px_0px_0px_#000000] overflow-visible ${containerSizes[size]}`}
      >
        <div className="flex items-center justify-center tracking-tight leading-none gap-1 sm:gap-1.5">
          {/* HACKER */}
          <span
            className="font-serif font-black uppercase text-[#FFE100] tracking-tight"
            style={{
              fontFamily: '"Times New Roman", Times, "Playfair Display", "Bodoni MT", Didot, serif',
            }}
          >
            HACKER
          </span>

          {/* गोवा */}
          <span
            className={`relative z-10 font-black bg-[#E6007A] text-white border-2 border-black rounded-[2px] transform -rotate-2 shadow-[2px_2px_0px_0px_#000000] font-sans ${devanagariSizes[size]}`}
            style={{
              letterSpacing: '0.02em',
              textShadow: '0 1px 1px rgba(0,0,0,0.5)',
            }}
          >
            गोवा
          </span>

          {/* HOUSE */}
          <span
            className="font-serif font-black uppercase text-[#FFE100] tracking-tight"
            style={{
              fontFamily: '"Times New Roman", Times, "Playfair Display", "Bodoni MT", Didot, serif',
            }}
          >
            HOUSE
          </span>
        </div>
      </div>
    </div>
  );
};

