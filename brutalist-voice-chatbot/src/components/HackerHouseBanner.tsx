import React from 'react';

interface HackerHouseBannerProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'responsive';
}

export const HackerHouseBanner: React.FC<HackerHouseBannerProps> = ({
  className = '',
  size = 'responsive',
}) => {
  const containerSizes = {
    sm: 'py-0.5 sm:py-1 px-2 sm:px-3 text-xs sm:text-sm',
    md: 'py-1 sm:py-1.5 px-3 sm:px-5 text-sm sm:text-lg md:text-xl',
    lg: 'py-2 sm:py-2.5 px-5 sm:px-8 text-lg sm:text-2xl md:text-3xl',
    responsive: 'py-0.5 sm:py-1.5 px-2.5 sm:px-4 md:px-5 text-xs sm:text-base md:text-lg',
  };

  const devanagariSizes = {
    sm: 'text-[9px] sm:text-[11px] px-1 py-0.2 -mx-0.5',
    md: 'text-xs sm:text-sm px-1.5 py-0.5 -mx-1',
    lg: 'text-sm sm:text-base px-2 py-0.5 -mx-1.5',
    responsive: 'text-[10px] sm:text-xs md:text-sm px-1 sm:px-1.5 py-0.5 -mx-0.5 sm:-mx-1',
  };

  return (
    <a
      href="https://hhgoa.com"
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center justify-center select-none cursor-pointer group transition-transform active:translate-x-[2px] active:translate-y-[2px] ${className}`}
      title="Visit Official Hacker House Goa (hhgoa.com)"
      aria-label="Visit Official Hacker House Goa website"
    >
      <div
        className={`relative flex items-center justify-center bg-[#054320] group-hover:bg-[#075c2c] border-2 sm:border-[3px] border-black shadow-[3px_3px_0px_0px_#000000] sm:shadow-[4px_4px_0px_0px_#000000] group-hover:shadow-[6px_6px_0px_0px_#000000] group-active:shadow-[2px_2px_0px_0px_#000000] transition-all overflow-visible ${containerSizes[size]}`}
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
            className={`relative z-10 font-black bg-[#E6007A] group-hover:bg-[#ff0084] text-white border-2 border-black rounded-[2px] transform -rotate-2 group-hover:rotate-0 shadow-[2px_2px_0px_0px_#000000] transition-all font-sans ${devanagariSizes[size]}`}
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
    </a>
  );
};



