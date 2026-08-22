import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowRight, Sparkles } from 'lucide-react';
import { audioFX } from '../utils/audioFx';
import { StudioLogo } from './StudioLogo';
import { HackerHouseBanner } from './HackerHouseBanner';

interface GreetingItem {
  greeting: string;
  language: string;
  transliteration: string;
}

const INDIAN_GREETINGS: GreetingItem[] = [
  { greeting: 'Hello', language: 'English', transliteration: 'Hello' },
  { greeting: 'வணக்கம்', language: 'Tamil', transliteration: 'Vanakkam' },
  { greeting: 'नमस्ते', language: 'Hindi', transliteration: 'Namaste' },
  { greeting: 'నమస్కారం', language: 'Telugu', transliteration: 'Namaskaram' },
  { greeting: 'ನಮಸ್ಕಾರ', language: 'Kannada', transliteration: 'Namaskara' },
  { greeting: 'നമസ്കാരം', language: 'Malayalam', transliteration: 'Namaskaram' },
  { greeting: 'নমস্কার', language: 'Bengali', transliteration: 'Nomoshkar' },
  { greeting: 'नमस्कार', language: 'Marathi', transliteration: 'Namaskar' },
  { greeting: 'નમસ્તે', language: 'Gujarati', transliteration: 'Namaste' },
  { greeting: 'ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ', language: 'Punjabi', transliteration: 'Sat Sri Akal' },
  { greeting: 'ନମସ୍କାର', language: 'Odia', transliteration: 'Nomoskar' },
  { greeting: 'নমস্কাৰ', language: 'Assamese', transliteration: 'Nomoskar' },
  { greeting: 'سلام', language: 'Urdu', transliteration: 'Salaam' },
  { greeting: 'प्रणाम', language: 'Sanskrit', transliteration: 'Pranam' },
];

interface HelloIntroScreenProps {
  onComplete: () => void;
}

export const HelloIntroScreen: React.FC<HelloIntroScreenProps> = ({ onComplete }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isExiting, setIsExiting] = useState(false);

  // Auto cycle through greetings
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= INDIAN_GREETINGS.length - 1) {
          // Completed all languages, transition to home page
          clearInterval(timer);
          setTimeout(() => {
            handleEnterHome();
          }, 1200);
          return prev;
        }
        return prev + 1;
      });
    }, 1300);

    return () => clearInterval(timer);
  }, []);

  const handleEnterHome = () => {
    if (isExiting) return;
    audioFX.playClick(520);
    setIsExiting(true);
    setTimeout(() => {
      onComplete();
    }, 450);
  };

  // Keyboard shortcut (Enter, Space, Escape to skip to home)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') {
        e.preventDefault();
        handleEnterHome();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isExiting]);

  const currentItem = INDIAN_GREETINGS[currentIndex];

  return (
    <motion.div
      initial={{ opacity: 1 }}
      animate={{ opacity: isExiting ? 0 : 1 }}
      transition={{ duration: 0.4 }}
      onClick={handleEnterHome}
      className="fixed inset-0 z-50 flex flex-col justify-between items-center bg-[#faf9f5] text-black cursor-pointer select-none p-6 sm:p-12 overflow-hidden"
    >
      {/* Subtle Brutalist grid background */}
      <div className="absolute inset-0 bg-grid opacity-25 pointer-events-none" />

      {/* Top Bar: Studio Logo, Centered Hacker House Goa Banner & Enter Home Action */}
      <div className="w-full max-w-5xl flex items-center justify-between gap-2 sm:gap-4 z-20 relative min-h-[50px]">
        <div className="z-10">
          <StudioLogo />
        </div>

        {/* Precise Screen Center Banner */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-auto z-10 flex justify-center">
          <HackerHouseBanner size="md" />
        </div>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            handleEnterHome();
          }}
          className="font-mono text-xs font-black uppercase px-3 py-1.5 border-2 border-black bg-white hover:bg-black hover:text-white transition-colors cursor-pointer flex items-center gap-1.5 hard-shadow-xs shrink-0 z-10"
        >
          <span className="hidden sm:inline">ENTER HOME</span>
          <span className="sm:hidden">ENTER</span>
          <ArrowRight className="w-3.5 h-3.5 stroke-[3]" />
        </button>
      </div>

      {/* Center Display: Animated Indian Language Greetings */}
      <div className="flex-1 flex flex-col items-center justify-center text-center z-10 max-w-3xl px-4 py-8">
        <div className="relative min-h-[190px] sm:min-h-[240px] flex flex-col items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentIndex}
              initial={{ opacity: 0, y: 22, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -22, scale: 1.03 }}
              transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
              className="flex flex-col items-center gap-3"
            >
              {/* Big Display "Hello" Typography */}
              <h1 className="text-5xl sm:text-7xl md:text-8xl font-black tracking-tight font-sans text-black select-none drop-shadow-sm">
                {currentItem.greeting}
              </h1>

              {/* Language Name Tag & Transliteration */}
              <div className="flex items-center gap-2 mt-2">
                <span className="font-mono text-xs sm:text-sm font-black uppercase px-2.5 py-0.5 bg-yellow-400 border-2 border-black">
                  {currentItem.language}
                </span>
                {currentItem.transliteration !== currentItem.greeting && (
                  <span className="font-mono text-xs sm:text-sm text-black/60 font-bold uppercase tracking-wider">
                    ({currentItem.transliteration})
                  </span>
                )}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom Bar: Multi-language Progress Dots & Tap prompt */}
      <div className="w-full max-w-md flex flex-col items-center gap-4 z-10">
        {/* Discrete segment dots */}
        <div className="flex items-center justify-center gap-1.5 flex-wrap max-w-xs">
          {INDIAN_GREETINGS.map((item, idx) => (
            <div
              key={item.language}
              className={`h-1.5 transition-all duration-300 ${
                idx === currentIndex
                  ? 'w-6 bg-black'
                  : idx < currentIndex
                  ? 'w-2 bg-black/60'
                  : 'w-2 bg-black/20'
              }`}
            />
          ))}
        </div>

        {/* Prompt action */}
        <div className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-widest text-black/60 animate-pulse flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5" />
          <span>CLICK ANYWHERE OR PRESS [SPACE] TO START</span>
        </div>
      </div>
    </motion.div>
  );
};
