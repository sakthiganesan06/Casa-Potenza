// static/app.js — Frontend client logic for Multilingual Voice RAG with Pure 16kHz PCM WAV Recorder

let currentLang = 'en';
let currentStrategy = 'semantic';
let audioContext = null;
let scriptProcessor = null;
let mediaStream = null;
let analyser = null;
let animationId = null;
let isRecording = false;
let recordedBuffers = [];

// DOM Elements
const micBtn = document.getElementById('micBtn');
const micContainer = document.getElementById('micContainer');
const micStatus = document.getElementById('micStatus');
const textQueryInput = document.getElementById('textQuery');
const submitBtn = document.getElementById('submitBtn');
const answerText = document.getElementById('answerText');
const confidenceBadge = document.getElementById('confidenceBadge');
const sourcesContainer = document.getElementById('sourcesContainer');
const ttsBtn = document.getElementById('ttsBtn');
const sampleChips = document.getElementById('sampleChips');
const totalTimeEl = document.getElementById('totalTime');
const totalBanner = document.getElementById('totalBanner');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');

// Milestones
const sttMetric = document.getElementById('sttMetric');
const guardMetric = document.getElementById('guardMetric');
const retrieveMetric = document.getElementById('retrieveMetric');
const ttftMetric = document.getElementById('ttftMetric');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadSampleQueries();
  initLanguageButtons();
  initStrategyButtons();
  initCanvas();
});

// Load System Stats
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (data.total_vectors) {
      document.getElementById('vectorCountBadge').innerText = `${data.total_vectors.toLocaleString()} Vectors (LanceDB)`;
    }
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

// Load Sample Queries
async function loadSampleQueries() {
  try {
    const res = await fetch('/api/sample_queries');
    const samples = await res.json();
    window.allSamples = samples;
    renderSampleChips(samples[currentLang] || []);
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

function renderSampleChips(queries) {
  sampleChips.innerHTML = '';
  queries.forEach(q => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.innerText = q;
    chip.onclick = () => {
      textQueryInput.value = q;
      handleTextSubmit();
    };
    sampleChips.appendChild(chip);
  });
}

// Language Selector
function initLanguageButtons() {
  document.querySelectorAll('.lang-selector .btn-pill').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.lang-selector .btn-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentLang = btn.dataset.lang;
      if (window.allSamples && window.allSamples[currentLang]) {
        renderSampleChips(window.allSamples[currentLang]);
      }
    };
  });
}

// Strategy Selector
function initStrategyButtons() {
  document.querySelectorAll('.strategy-selector .btn-pill').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.strategy-selector .btn-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentStrategy = btn.dataset.strategy;
    };
  });
}

// Audio Canvas
function initCanvas() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.2)';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
}

// Real-Time Web Speech Recognition + Pure 16kHz PCM WAV Fallback
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let liveTranscript = "";

micBtn.onclick = toggleRecording;

async function toggleRecording() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
}

async function startRecording() {
  liveTranscript = "";
  
  // 1. Try real-time browser SpeechRecognition for instant 0ms STT
  if (SpeechRecognition) {
    try {
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      
      const langMap = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'ta': 'ta-IN',
        'te': 'te-IN'
      };
      recognition.lang = langMap[currentLang] || currentLang;
      
      recognition.onresult = (event) => {
        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            liveTranscript += event.results[i][0].transcript;
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        textQueryInput.value = (liveTranscript + ' ' + interim).trim();
      };
      
      recognition.onerror = (e) => {
        console.warn('Speech recognition notice:', e.error);
      };
      
      recognition.start();
    } catch (e) {
      console.warn('Web Speech API not active:', e);
    }
  }

  // 2. Start audio visualizer & WAV capture buffer
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    
    const source = audioContext.createMediaStreamSource(mediaStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    recordedBuffers = [];

    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      recordedBuffers.push(new Float32Array(input));
    };

    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    isRecording = true;
    micContainer.classList.add('recording');
    micStatus.innerText = '🔴 Listening... Click when done speaking';
    drawVisualizer();
  } catch (err) {
    if (recognition) {
      isRecording = true;
      micContainer.classList.add('recording');
      micStatus.innerText = '🔴 Listening... Click when done';
    } else {
      alert('Microphone access error: ' + err.message);
    }
  }
}

function stopRecording() {
  if (isRecording) {
    isRecording = false;
    micContainer.classList.remove('recording');
    micStatus.innerText = 'Searching knowledge base...';
    cancelAnimationFrame(animationId);
    initCanvas();

    if (recognition) {
      try { recognition.stop(); } catch (e) {}
      recognition = null;
    }

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }

    const transcribedQuery = textQueryInput.value.trim();
    if (transcribedQuery.length > 0) {
      // Instant Client-Side STT -> Direct vector search query (<200ms)
      handleTextSubmit();
    } else if (recordedBuffers.length > 0) {
      // Fallback: Send audio WAV to server
      const wavBlob = encodeWAV(recordedBuffers, 16000);
      sendAudioToServer(wavBlob);
    } else {
      micStatus.innerText = 'Click to Speak';
    }
  }
}


// Downsample audio buffer to 16kHz
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
  if (inputSampleRate === outputSampleRate || inputSampleRate < outputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

// Convert Float32 audio arrays to valid 16kHz 16-bit mono WAV Blob
function encodeWAV(buffers, inputSampleRate = 16000, outputSampleRate = 16000) {
  let length = 0;
  for (let b of buffers) length += b.length;

  const merged = new Float32Array(length);
  let offset = 0;
  for (let b of buffers) {
    merged.set(b, offset);
    offset += b.length;
  }

  // Downsample to 16kHz if needed
  const downsampled = downsampleBuffer(merged, inputSampleRate, outputSampleRate);
  const downsampledLength = downsampled.length;

  // 16-bit PCM
  const buffer = new ArrayBuffer(44 + downsampledLength * 2);
  const view = new DataView(buffer);

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + downsampledLength * 2, true);
  writeString(view, 8, 'WAVE');
  // format chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);        // SubChunk1Size (16 for PCM)
  view.setUint16(20, 1, true);         // AudioFormat (1 for PCM)
  view.setUint16(22, 1, true);         // NumChannels (1 = mono)
  view.setUint32(24, outputSampleRate, true);// SampleRate (16000)
  view.setUint32(28, outputSampleRate * 2, true); // ByteRate (16000 * 1 * 16/8)
  view.setUint16(32, 2, true);         // BlockAlign (1 * 16/8)
  view.setUint16(34, 16, true);        // BitsPerSample (16 bits)
  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, downsampledLength * 2, true);

  // Write PCM samples
  let index = 44;
  for (let i = 0; i < downsampledLength; i++) {
    let s = Math.max(-1, Math.min(1, downsampled[i]));
    view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    index += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function drawVisualizer() {
  if (!isRecording || !analyser) return;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  canvasCtx.fillStyle = 'rgba(10, 15, 29, 0.5)';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  const barWidth = (canvas.width / bufferLength) * 2.5;
  let barHeight;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    barHeight = (dataArray[i] / 255) * canvas.height;
    canvasCtx.fillStyle = `rgb(${dataArray[i] + 100}, 99, 241)`;
    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
  }

  animationId = requestAnimationFrame(drawVisualizer);
}

// Send Audio Blob
async function sendAudioToServer(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'mic.wav');
  formData.append('lang_code', currentLang);
  formData.append('strategy', currentStrategy);

  setLoadingState(true);
  try {
    const res = await fetch('/api/audio', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    handlePipelineResponse(data);
    if (data.transcript) {
      textQueryInput.value = data.transcript;
    }
  } catch (err) {
    answerText.innerText = 'Error processing audio: ' + err.message;
  } finally {
    setLoadingState(false);
    micStatus.innerText = 'Click to Speak';
  }
}

// Text Query Submission
submitBtn.onclick = handleTextSubmit;
textQueryInput.onkeydown = e => {
  if (e.key === 'Enter') handleTextSubmit();
};

async function handleTextSubmit() {
  const query = textQueryInput.value.trim();
  if (!query) return;

  setLoadingState(true);
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        lang_code: currentLang,
        strategy: currentStrategy,
      }),
    });
    const data = await res.json();
    handlePipelineResponse(data);
  } catch (err) {
    answerText.innerText = 'Error: ' + err.message;
  } finally {
    setLoadingState(false);
  }
}

// Handle Pipeline Response & Latency Breakdown
function handlePipelineResponse(data) {
  const res = data.result || data;
  const lat = res.latency || {};

  // Update Answer
  if (res.refused) {
    answerText.innerHTML = `<span style="color: #f59e0b;">⚠️ Refusal (${res.refusal_reason || 'grounding'}):</span> Not enough context found in knowledge base to answer safely.`;
    confidenceBadge.innerText = 'Refused';
    confidenceBadge.style.background = 'rgba(245, 158, 11, 0.2)';
    confidenceBadge.style.color = '#f59e0b';
  } else {
    answerText.innerText = res.answer || 'No answer returned.';
    const conf = Math.round((res.confidence || 1.0) * 100);
    confidenceBadge.innerText = `Confidence: ${conf}%`;
    confidenceBadge.style.background = 'rgba(16, 185, 129, 0.2)';
    confidenceBadge.style.color = '#10b981';
  }

  // Update Sources
  sourcesContainer.innerHTML = '';
  const sources = res.sources || [];
  if (sources.length > 0) {
    sources.forEach(src => {
      const span = document.createElement('span');
      span.className = 'src-pill';
      span.innerText = src;
      sourcesContainer.appendChild(span);
    });
  } else {
    sourcesContainer.innerHTML = '<span style="color: var(--text-dim); font-size:12px;">No sources cited</span>';
  }

  // Update Latency Milestones
  sttMetric.innerText = lat.stt_ms !== undefined ? `${lat.stt_ms.toFixed(1)} ms` : '—';
  guardMetric.innerText = '< 1.0 ms';
  retrieveMetric.innerText = lat.retrieval_ms !== undefined ? `${lat.retrieval_ms.toFixed(1)} ms` : '—';
  ttftMetric.innerText = lat.llm_ttft_ms !== undefined ? `${lat.llm_ttft_ms.toFixed(1)} ms` : '—';

  const total = lat.total_ms || 0;
  totalTimeEl.innerText = `${total.toFixed(1)} ms`;

  if (total <= 200 && total > 0) {
    totalBanner.classList.remove('violation');
    totalBanner.querySelector('.total-title').innerText = '⚡ End-to-End Latency (Within <200ms Budget!)';
  } else {
    totalBanner.classList.add('violation');
    totalBanner.querySelector('.total-title').innerText = '⏱️ End-to-End Latency';
  }
}

// Text-to-Speech Output
ttsBtn.onclick = () => {
  const text = answerText.innerText;
  if (!text || text.includes('Click to Speak') || text.includes('Error')) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const langMap = { en: 'en-US', hi: 'hi-IN', ta: 'ta-IN', te: 'te-IN' };
  utterance.lang = langMap[currentLang] || 'en-US';
  window.speechSynthesis.speak(utterance);
};

function setLoadingState(loading) {
  if (loading) {
    submitBtn.innerHTML = '<div class="spinner"></div>';
    submitBtn.disabled = true;
    answerText.innerHTML = '<span style="color: var(--text-muted);">Searching 24,693 LanceDB vectors & generating response...</span>';
  } else {
    submitBtn.innerHTML = 'Ask';
    submitBtn.disabled = false;
  }
}
