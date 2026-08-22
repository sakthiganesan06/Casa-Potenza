export type RecordingState = 
  | 'idle'
  | 'requesting'
  | 'recording'
  | 'processing'
  | 'result'
  | 'error';

export type PersonaMode = 'brutalist' | 'tech' | 'philosophy' | 'haiku';

export type VisualizerMode = 'bars' | 'wave' | 'vu' | 'circular';

export type ThemeColor = 'yellow' | 'lime' | 'cyan' | 'orange' | 'white';

export type CursorMode = 'arrow' | 'crosshair' | 'radar' | 'trail' | 'off';

export interface ChatInteraction {
  id: string;
  timestamp: string;
  transcription: string;
  reply: string;
  persona: PersonaMode;
  audioDurationSeconds?: number;
  audioBlobUrl?: string;
  latencyMs?: number;
  latencyScore?: number;
  score?: {
    value: number; // 0 to 100
    tier: 'P50' | 'P70' | 'P100';
  };
}

export interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  isRecording: boolean;
  visualizerMode: VisualizerMode;
  onToggleMode?: () => void;
  accentColor?: string;
}

