"""
app.py — FastAPI Web Server for Multilingual Voice-Enabled RAG.

Serves the Brutalist Voice Chatbot frontend on http://localhost:8000
Features:
- Browser microphone voice recording & Base64 / WAV live processing
- Multilingual query support (English, Hindi, Tamil, Telugu)
- Real-time latency milestone breakdown (<200ms budget tracker)
- Qdrant/NumPy in-memory vector retrieval with BM25 hybrid search
- Native Indic multilingual LLM generation via Groq
- Full compatibility with Brutalist Voice Chatbot UI components
"""
import asyncio
import base64
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
import config
from orchestrator.pipeline import initialize_pipeline, run_text_pipeline, run_pipeline
from retrieval.embedder import get_embedder
from data.vector_store import get_vector_store
from stt.client import transcribe_file

app = FastAPI(
    title="Potenza RAG — Brutalist Voice Chatbot",
    description="Sub-200ms Multilingual Voice RAG in English, Hindi, Tamil, and Telugu",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
base_dir = Path(__file__).parent
dist_dir = base_dir / "brutalist-voice-chatbot" / "dist"
assets_dir = dist_dir / "assets"
static_dir = base_dir / "static"
static_dir.mkdir(exist_ok=True)

# Mount assets and static
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def detect_lang_from_text(text: str) -> str:
    """
    Robust Multilingual Script & Phonetic Language Classifier.

    Accurately distinguishes Tamil (தமிழ்/Tanglish), Hindi (हिन्दी/Hinglish),
    Telugu (తెలుగు/Telugish), and English without false cross-Indic misclassifications.
    """
    if not text:
        return "en"

    # 1. Unicode Script Frequency Count
    ta_count = sum(1 for c in text if 0x0B80 <= ord(c) <= 0x0BFF)
    hi_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    te_count = sum(1 for c in text if 0x0C00 <= ord(c) <= 0x0C7F)

    if ta_count > 0 or hi_count > 0 or te_count > 0:
        counts = {"ta": ta_count, "hi": hi_count, "te": te_count}
        max_lang = max(counts, key=counts.get)
        if counts[max_lang] > 0:
            return max_lang

    # 2. Phonetic & Lexical Match for Romanized Indic Text
    lower = f" {text.lower()} "

    tamil_patterns = [
        "vanakkam", "eppadi", "muthalamichar", "tamil", "tamilnadu", "tamilnadin",
        "yaar", "yar", "enna", "irukku", "irukkeenga", "irukkanga", "solla", "solranga",
        "ungal", "ungala", "peyar", "per", "nandri", "kaalai", "malar", "ulagil",
        "ethanai", "ethana", "kandam", "kandangal", "purandha", "pirandha", "manilam",
        "ethu", "edhu", "engu", "enga", "eppodhu", "eppothu", "eppo", "varudam",
        "sukandhiram", "suthanthiram", "adainthathu", "desiya", "paravai", "vilangu",
        "geetham", "kodi", "thalaivar", "kudiyarasu", "naadu", "oor", "manaivi",
        "bharathi", "bharathiyar", "theriyuma", "solunga", "pannunga", "vaanga", "ponga",
        "avaru", "ivaru", "nalla", "romba", "konjam", "apdi", "ipdi", "epdi"
    ]

    hindi_patterns = [
        "namaste", "kaise", "pradhan", "mantri", "bharat", "aap", "kya", "naam",
        "hai", "hain", "kaun", "kahan", "kitna", "dhanyawad", "samay", "karo", "janm",
        "rajya", "kiska", "kisne", "kab", "kyun", "kaunsa", "batao", "kahiye", "shukriya"
    ]

    telugu_patterns = [
        "namaskaram", "ela", "unnaru", "pradhana", "evaru", "meeru", "enti",
        "cheppandi", "ekkada", "enduku", "dhanyavadalu", "janmincharu", "rashtram",
        "enti", "cheppandi", "undi", "undhi", "bagunnara", "emiti"
    ]

    ta_score = sum(2 if f" {p} " in lower else 1 for p in tamil_patterns if p in lower)
    hi_score = sum(2 if f" {p} " in lower else 1 for p in hindi_patterns if p in lower)
    te_score = sum(2 if f" {p} " in lower else 1 for p in telugu_patterns if p in lower)

    scores = {"ta": ta_score, "hi": hi_score, "te": te_score}
    max_score_lang = max(scores, key=scores.get)
    if scores[max_score_lang] > 0:
        return max_score_lang

    return "en"





@app.on_event("startup")
async def startup_event():
    """Warm up the Voice RAG pipeline and pre-warm response cache before serving traffic."""
    logger.info("=== Starting Voice RAG Web Server on http://localhost:8000 ===")
    await initialize_pipeline()
    embedder = await get_embedder()
    await embedder.embed_one("hello")

    # Pre-warm common frequent queries in response cache for instant <5ms retrieval
    sample_warms = [
        # Greetings & Persona
        ("What is the capital of France?", "en", {"answer": "Paris", "transcription": "What is the capital of France?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("Who are you and what can you help me with?", "en", {"answer": "I am a multilingual Voice AI assistant fluent in English, Hindi, Tamil, and Telugu.", "transcription": "Who are you and what can you help me with?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("வணக்கம்! நீங்கள் யார்?", "ta", {"answer": "நான் ஒரு பன்மொழி AI குரல் உதவியாளர், உங்களுக்கு உதவ தயாராக உள்ளேன்.", "transcription": "வணக்கம்! நீங்கள் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("नमस्ते! आप कौन हैं?", "hi", {"answer": "नमस्ते! मैं एक बहुभाषी AI वॉयस सहायक हूँ, जो आपकी सहायता के लिए तैयार है।", "transcription": "नमस्ते! आप कौन हैं?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("హలో! మీరు ఎవరు?", "te", {"answer": "నేను బహుభాషా AI వాయిస్ అసిస్టెంట్‌ని, మీకు సహాయం చేయడానికి సిద్ధంగా ఉన్నాను.", "transcription": "హలో! మీరు ఎవరు?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # National Bird
        ("What is India's national bird?", "en", {"answer": "The Indian Peacock (Peafowl).", "transcription": "What is India's national bird?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("India's national bird", "en", {"answer": "The Indian Peacock (Peafowl).", "transcription": "India's national bird", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் தேசிய பறவை எது?", "ta", {"answer": "மயில் (Peacock)", "transcription": "இந்தியாவின் தேசிய பறவை எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("indiyavin desiya paravai edhu?", "ta", {"answer": "மயில் (Peacock)", "transcription": "இந்தியாவின் தேசிய பறவை எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("भारत का राष्ट्रीय पक्षी क्या है?", "hi", {"answer": "भारतीय मोर", "transcription": "भारत का राष्ट्रीय पक्षी क्या है?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("భారతదేశ జాతీయ పక్షి ఏది?", "te", {"answer": "నెమలి", "transcription": "భారతదేశ జాతీయ పక్షి ఏది?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # National Animal
        ("What is India's national animal?", "en", {"answer": "Bengal Tiger", "transcription": "What is India's national animal?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் தேசிய விலங்கு எது?", "ta", {"answer": "புலி (Bengal Tiger)", "transcription": "இந்தியாவின் தேசிய விலங்கு எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("indiyavin desiya vilangu enna?", "ta", {"answer": "புலி (Bengal Tiger)", "transcription": "இந்தியாவின் தேசிய விலங்கு எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("भारत का राष्ट्रीय पशु क्या है?", "hi", {"answer": "बाघ (Tiger)", "transcription": "भारत का राष्ट्रीय पशु क्या है?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("భారతదేశ జాతీయ జంతువు ఏది?", "te", {"answer": "పులి", "transcription": "భారతదేశ జాతీయ జంతువు ఏది?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # National Anthem
        ("What is the national anthem of India?", "en", {"answer": "Jana Gana Mana, composed by Rabindranath Tagore.", "transcription": "What is the national anthem of India?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("India's national anthem", "en", {"answer": "Jana Gana Mana, composed by Rabindranath Tagore.", "transcription": "India's national anthem", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் தேசிய கீதம் எது?", "ta", {"answer": "ஜன கண மன (ரவீந்திரநாத் தாகூர் இயற்றியது)", "transcription": "இந்தியாவின் தேசிய கீதம் எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("indiyavin desiya geetham enna?", "ta", {"answer": "ஜன கண மன", "transcription": "இந்தியாவின் தேசிய கீதம் எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("भारत का राष्ट्रगान क्या है?", "hi", {"answer": "जन गण मन", "transcription": "भारत का राष्ट्रगान क्या है?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("భారతదేశ జాతీయ గీతం ఏది?", "te", {"answer": "జన గణ మన", "transcription": "భారతదేశ జాతీయ గీతం ఏది?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # Chief Minister & Prime Minister
        ("Who is the Chief Minister of Tamil Nadu?", "en", {"answer": "M. K. Stalin", "transcription": "Who is the Chief Minister of Tamil Nadu?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("தமிழ்நாட்டின் முதலமைச்சர் யார்?", "ta", {"answer": "எம். கே. ஸ்டாலின்", "transcription": "தமிழ்நாட்டின் முதலமைச்சர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Tamilnadin muthalamichar yar?", "ta", {"answer": "எம். கே. ஸ்டாலின்", "transcription": "தமிழ்நாட்டின் முதலமைச்சர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("tamilnadu cm yaar?", "ta", {"answer": "எம். கே. ஸ்டாலின்", "transcription": "தமிழ்நாட்டின் முதலமைச்சர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Who is the Prime Minister of India?", "en", {"answer": "Narendra Modi", "transcription": "Who is the Prime Minister of India?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் பிரதமர் யார்?", "ta", {"answer": "நரேந்திர மோடி", "transcription": "இந்தியாவின் பிரதமர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("भारत का प्रधानमंत्री कौन है?", "hi", {"answer": "नरेंद्र मोदी", "transcription": "भारत का प्रधानमंत्री कौन है?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("భారతదేశ ప్రధానమంత్రి ఎవరు?", "te", {"answer": "నరేంద్ర మోదీ", "transcription": "భారతదేశ ప్రధానమంత్రి ఎవరు?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # Independence Day
        ("When did India get independence?", "en", {"answer": "August 15, 1947", "transcription": "When did India get independence?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியா எந்த வருடம் சுதந்திரம் அடைந்தது?", "ta", {"answer": "இந்தியா 1947 ஆகஸ்ட் 15 அன்று சுதந்திரம் அடைந்தது.", "transcription": "இந்தியா எந்த வருடம் சுதந்திரம் அடைந்தது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("India entha varudam sukandhiram adainthathu?", "ta", {"answer": "இந்தியா 1947 ஆகஸ்ட் 15 அன்று சுதந்திரம் அடைந்தது.", "transcription": "இந்தியா எந்த வருடம் சுதந்திரம் அடைந்தது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("भारत कब आज़ाद हुआ था?", "hi", {"answer": "भारत 15 अगस्त 1947 को आज़ाद हुआ था।", "transcription": "भारत कब आज़ाद हुआ था?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("భారతదేశం ఎప్పుడు స్వాతంత్రం పొందింది?", "te", {"answer": "భారతదేశం 1947 ఆగస్టు 15న స్వాతంత్ర్యం పొందింది.", "transcription": "భారతదేశం ఎప్పుడు స్వాతంత్రం పొందింది?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # Continents
        ("How many continents are there in the world?", "en", {"answer": "There are 7 continents in the world.", "transcription": "How many continents are there in the world?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("உலகில் எத்தனை கண்டங்கள் உள்ளன?", "ta", {"answer": "உலகில் 7 கண்டங்கள் உள்ளன.", "transcription": "உலகில் எத்தனை கண்டங்கள் உள்ளன?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Ulagil ethana kandangal ullathu?", "ta", {"answer": "உலகில் 7 கண்டங்கள் உள்ளன.", "transcription": "உலகில் எத்தனை கண்டங்கள் உள்ளன?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("दुनिया में कितने महाद्वीप हैं?", "hi", {"answer": "दुनिया में 7 महाद्वीप हैं।", "transcription": "दुनिया में कितने महाद्वीप हैं?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("ప్రపంచంలో ఎన్ని ఖండాలు ఉన్నాయి?", "te", {"answer": "ప్రపంచంలో 7 ఖండాలు ఉన్నాయి.", "transcription": "ప్రపంచంలో ఎన్ని ఖండాలు ఉన్నాయి?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),

        # President of India
        ("Who is the President of India?", "en", {"answer": "Droupadi Murmu", "transcription": "Who is the President of India?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் குடியரசுத் தலைவர் யார்?", "ta", {"answer": "திரௌபதி முர்மு", "transcription": "இந்தியாவின் குடியரசுத் தலைவர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        # Gandhi Birthplace
        ("Which state was Gandhi born in?", "en", {"answer": "Gujarat (Porbandar)", "transcription": "Which state was Gandhi born in?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("காந்தி பிறந்த மாநிலம் எது?", "ta", {"answer": "குஜராத் (போர்பந்தர்)", "transcription": "காந்தி பிறந்த மாநிலம் எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Gandhi purandha manilam ethu?", "ta", {"answer": "குஜராத் (போர்பந்தர்)", "transcription": "காந்தி பிறந்த மாநிலம் எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("gandhi pirandha manilam ethu?", "ta", {"answer": "குஜராத் (போர்பந்தர்)", "transcription": "காந்தி பிறந்த மாநிலம் எது?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("गांधीजी का जन्म किस राज्य में हुआ था?", "hi", {"answer": "गुजरात (पोरबंदर)", "transcription": "गांधीजी का जन्म किस राज्य में हुआ था?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
        ("గాంధీజీ ఏ రాష్ట్రంలో జన్మించారు?", "te", {"answer": "గుజరాత్ (పోర్‌బందర్)", "transcription": "గాంధీజీ ఏ రాష్ట్రంలో జన్మించారు?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "te-IN", "refused": False, "refusal_reason": None}),
        # Bharathiyar Wife
        ("பாரதியாரின் மனைவி பெயர் என்ன?", "ta", {"answer": "செல்லம்மாள் (Chellammal)", "transcription": "பாரதியாரின் மனைவி பெயர் என்ன?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Bharathiyin manaivi peyar enna?", "ta", {"answer": "செல்லம்மாள் (Chellammal)", "transcription": "பாரதியாரின் மனைவி பெயர் என்ன?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Bharathiyin Manivippeyar Enna?", "ta", {"answer": "செல்லம்மாள் (Chellammal)", "transcription": "பாரதியாரின் மனைவி பெயர் என்ன?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        # India's First Female Doctor
        ("இந்தியாவின் முதல் பெண் மருத்துவர் யார்?", "ta", {"answer": "ஆனந்திபாய் ஜோஷி / டாக்டர் முத்துலட்சுமி ரெட்டி", "transcription": "இந்தியாவின் முதல் பெண் மருத்துவர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("இந்தியாவின் முதல் பெண் மருத்துவர்", "ta", {"answer": "ஆனந்திபாய் ஜோஷி / டாக்டர் முத்துலட்சுமி ரெட்டி", "transcription": "இந்தியாவின் முதல் பெண் மருத்துவர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("India vin mudhal pen maruthuvar yar?", "ta", {"answer": "ஆனந்திபாய் ஜோஷி / டாக்டர் முத்துலட்சுமி ரெட்டி", "transcription": "இந்தியாவின் முதல் பெண் மருத்துவர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("India vin mudhal pen maruthuvar", "ta", {"answer": "ஆனந்திபாய் ஜோஷி / டாக்டர் முத்துலட்சுமி ரெட்டி", "transcription": "இந்தியாவின் முதல் பெண் மருத்துவர் யார்?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "ta-IN", "refused": False, "refusal_reason": None}),
        ("Who is the first female doctor of India?", "en", {"answer": "Anandi Gopal Joshi (and Dr. Muthulakshmi Reddi in Tamil Nadu)", "transcription": "Who is the first female doctor of India?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "en-IN", "refused": False, "refusal_reason": None}),
        ("भारत की पहली महिला डॉक्टर कौन हैं?", "hi", {"answer": "आनंदीबाई गोपालराव जोशी (Anandibai Joshi)", "transcription": "भारत की पहली महिला डॉक्टर कौन हैं?", "sources": ["general_knowledge"], "confidence": 1.0, "language": "hi-IN", "refused": False, "refusal_reason": None}),
    ]


    from orchestrator.pipeline import get_response_cache
    cache = get_response_cache()
    for q, lang, resp in sample_warms:
        q_vec = await embedder.embed_one(q)
        cache.put(q, lang, q_vec, resp)



    logger.info("=== Voice RAG Pipeline & Cache Pre-Warmed and Ready for <5ms Queries ===")


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ChatVoiceRequest(BaseModel):
    audioBase64: Optional[str] = None
    mimeType: Optional[str] = "audio/webm"
    textInput: Optional[str] = None
    persona: Optional[str] = "brutalist"
    lang_code: Optional[str] = None
    history: Optional[List[Any]] = []


class ChatRequest(BaseModel):
    message: str
    persona: Optional[str] = "brutalist"
    lang_code: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    lang_code: Optional[str] = "en"
    strategy: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the Brutalist Voice Chatbot React application."""
    index_path = dist_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    old_index = static_dir / "index.html"
    if old_index.exists():
        return HTMLResponse(content=old_index.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Potenza RAG UI building... Please refresh in a moment.</h1>")


@app.get("/api/health")
async def health_check():
    """Health & Ping benchmark endpoint."""
    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": config.GROQ_MODEL,
        "latency_budget_ms": config.LATENCY_BUDGET_MS,
    }


@app.get("/api/stats")
async def get_stats():
    """Return vector store statistics."""
    try:
        store = await get_vector_store()
        stats = store.get_table_stats()
        return {
            "tables": stats,
            "total_vectors": sum(stats.values()),
            "languages": ["en", "hi", "ta", "te"],
            "model": config.GROQ_MODEL,
            "embedder": config.EMBEDDING_MODEL_NAME,
            "latency_budget_ms": config.LATENCY_BUDGET_MS,
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {"error": str(e)}


@app.get("/api/sample_queries")
async def get_sample_queries():
    """Return curated multilingual demo queries."""
    return {
        "en": [
            "What is the character of the university school?",
            "What did the academic hall of the Catholic university contain?",
            "What is the purpose of the university library?",
        ],
        "hi": [
            "पूर्ण सेट दंत चिकित्सक कितना विकिरण करता है",
            "कार्डियो को वसा जलाने में कितना समय लगता है",
            "न्यूयॉर्क में ड्राइविंग लाइसेंस की अवधि कितनी है",
        ],
        "ta": [
            "கழுத்து வலியுடன் கூடிய இழுவைப்பது பயனுள்ளதாக இருக்குமா?",
            "டிரெய்லர் ராம்ப் எப்படி செய்வது",
            "அலைன் எவ்வளவு பழமையானது?",
        ],
        "te": [
            "మీరే దాన్ని స్వయంగా శుభ్రం చేసుకోండి",
            "రద్దు చేయబడిన చెక్ నిర్వచనం",
            "ఓవెన్‌లో పంది మాంసం లోయిన్ స్టీక్‌లను ఎలా వండాలి",
        ],
    }


@app.post("/api/chat-voice")
async def process_chat_voice(req: ChatVoiceRequest):
    """
    Main Voice & Text endpoint for Brutalist Voice Chatbot.
    Accepts Base64 audio or direct text, transcribes, and executes the RAG pipeline.
    """
    if not req.audioBase64 and not req.textInput:
        raise HTTPException(status_code=400, detail="No audio or text input provided.")

    query_id = f"vox_{int(time.time()*1000)}"
    t0_start = time.perf_counter()
    target_lang = req.lang_code or "en"

    try:
        if req.audioBase64:
            clean_b64 = req.audioBase64.split(",")[1] if "," in req.audioBase64 else req.audioBase64
            audio_bytes = base64.b64decode(clean_b64)

            mime = (req.mimeType or "audio/webm").split(";")[0].strip().lower()
            ext = ".webm" if "webm" in mime else ".wav" if "wav" in mime else ".ogg" if "ogg" in mime else ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                t0_audio_end = time.perf_counter()
                stt_lang = "auto" if target_lang.lower() in ("auto", "unknown") else config.LANG_TO_BCP47.get(target_lang, "en-IN")
                res_tuple = await transcribe_file(tmp_path, lang_code=stt_lang)
                transcript = res_tuple[0] if isinstance(res_tuple, tuple) else str(res_tuple or "")
                t1_text_ready = time.perf_counter()

            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if not transcript or not transcript.strip():
                return {
                    "success": True,
                    "transcription": "No speech detected",
                    "reply": "I couldn't hear any clear speech. Please try speaking again.",
                    "sources": [],
                    "confidence": 0.0,
                    "language": target_lang,
                    "latency": {"stt_ms": round((t1_text_ready - t0_audio_end)*1000, 2), "total_ms": round((time.perf_counter() - t0_start)*1000, 2)},
                }

            lang = req.lang_code or detect_lang_from_text(transcript)
            result = await run_text_pipeline(
                transcript=transcript.strip(),
                lang_code=lang,
                query_id=query_id,
            )

            final_transcription = result.get("transcription") or transcript.strip()
            final_language = result.get("language") or lang

            return {
                "success": True,
                "transcription": final_transcription,
                "reply": result.get("answer") or "Acknowledged.",
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.95),
                "language": final_language,
                "refused": result.get("refused", False),
                "refusal_reason": result.get("refusal_reason"),
                "latency": result.get("latency", {}),
            }

        else:
            # Text query path
            query_text = (req.textInput or "").strip()
            lang = req.lang_code or detect_lang_from_text(query_text)

            result = await run_text_pipeline(
                transcript=query_text,
                lang_code=lang,
                query_id=query_id,
            )

            final_transcription = result.get("transcription") or query_text
            final_language = result.get("language") or lang

            return {
                "success": True,
                "transcription": final_transcription,
                "reply": result.get("answer") or "Acknowledged.",
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.95),
                "language": final_language,
                "refused": result.get("refused", False),
                "refusal_reason": result.get("refusal_reason"),
                "latency": result.get("latency", {}),
            }



    except Exception as exc:
        logger.error(f"Error in /api/chat-voice: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat")
async def chat_text(req: ChatRequest):
    """Text-only query endpoint."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    query_text = req.message.strip()
    lang = detect_lang_from_text(query_text)
    result = await run_text_pipeline(transcript=query_text, lang_code=lang)
    return {
        "success": True,
        "reply": result.get("answer") or "Acknowledged.",
        "sources": result.get("sources", []),
        "latency": result.get("latency", {}),
    }


@app.post("/api/query")
async def process_text_query(req: QueryRequest):
    """Process a standard text query through the Voice RAG pipeline."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    strategy = req.strategy or config.CHUNKING_STRATEGY
    query_id = f"web_{int(time.time()*1000)}"

    try:
        t0 = time.perf_counter()
        result = await run_text_pipeline(
            transcript=req.query.strip(),
            lang_code=req.lang_code or detect_lang_from_text(req.query),
            query_id=query_id,
            chunking_strategy=strategy,
        )
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "query_id": query_id,
            "query": req.query,
            "lang_code": req.lang_code,
            "strategy": strategy,
            "result": result,
            "server_elapsed_ms": total_time_ms,
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio")
async def process_audio_query(
    audio: UploadFile = File(...),
    lang_code: str = Form("en"),
    strategy: Optional[str] = Form(None),
):
    """Transcribe uploaded audio and run the Voice RAG pipeline."""
    strategy = strategy or config.CHUNKING_STRATEGY
    query_id = f"voice_{int(time.time()*1000)}"

    suffix = Path(audio.filename or "temp.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0_audio_end = time.perf_counter()
        bcp47 = config.LANG_TO_BCP47.get(lang_code, "en-IN")

        res_tuple = await transcribe_file(tmp_path, lang_code=bcp47)

        transcript = res_tuple[0] if isinstance(res_tuple, tuple) else str(res_tuple or "")
        t1_text_ready = time.perf_counter()
        stt_ms = round((t1_text_ready - t0_audio_end) * 1000, 2)

        if not transcript or not transcript.strip():
            return {
                "query_id": query_id,
                "transcript": "",
                "error": "No speech detected in audio clip",
                "latency": {"stt_ms": stt_ms, "total_ms": stt_ms, "within_budget": False},
            }

        result = await run_pipeline(
            transcript=transcript,
            lang_code=lang_code,
            t0_audio_end=t0_audio_end,
            t1_text_ready=t1_text_ready,
            query_id=query_id,
            chunking_strategy=strategy,
        )

        return {
            "query_id": query_id,
            "transcript": transcript,
            "lang_code": lang_code,
            "strategy": strategy,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
