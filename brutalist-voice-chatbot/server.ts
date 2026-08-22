import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

let aiClient: GoogleGenAI | null = null;

function getGenAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY || "";
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // JSON payload parser with generous limit for audio blobs
  app.use(express.json({ limit: "50mb" }));

  // Health check
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // Voice transcription & Chat endpoint
  app.post("/api/chat-voice", async (req, res) => {
    try {
      const { audioBase64, mimeType = "audio/webm", textInput, persona = "brutalist", history = [] } = req.body;

      if (!audioBase64 && !textInput) {
        return res.status(400).json({ error: "No audio or text input provided." });
      }

      const ai = getGenAI();

      let promptParts: any[] = [];

      let personaPrompt = "Honest, bold, concise, articulate, and direct with zero fluff or generic filler phrases.";
      if (persona === "tech") {
        personaPrompt = "Technical, architectural, ultra-precise, system-oriented with code/data breakdowns where relevant.";
      } else if (persona === "philosophy") {
        personaPrompt = "Deep, socratic, intellectually rigorous, questioning assumptions directly.";
      } else if (persona === "haiku") {
        personaPrompt = "Poetic, philosophical, crafted in traditional 5-7-5 syllable haiku format.";
      }

      const systemInstruction = `You are an interactive brutalist AI voice chatbot.
Persona Tone: ${personaPrompt}
Instructions:
1. If audio is provided, accurately transcribe everything the user spoke. If text was sent, keep that text as transcription.
2. Provide an intelligent, insightful, and direct answer to their query in character.
3. Keep the reply punchy and impactful (1-4 clear paragraphs or bullet points).
You must output valid JSON with exactly two fields:
{
  "transcription": string, // verbatim speech transcribed from the input
  "reply": string // your direct chatbot answer
}`;

      if (audioBase64) {
        // Strip data URL prefix robustly (everything up to and including the first comma)
        const cleanBase64 = audioBase64.includes(",")
          ? audioBase64.split(",")[1].trim()
          : audioBase64.trim();

        // Normalize mime type to standard base audio MIME type (e.g. 'audio/webm;codecs=opus' -> 'audio/webm')
        let cleanMimeType = (mimeType || "audio/webm").split(";")[0].trim().toLowerCase();
        if (!cleanMimeType.startsWith("audio/")) {
          cleanMimeType = "audio/webm";
        }

        promptParts = [
          {
            inlineData: {
              mimeType: cleanMimeType,
              data: cleanBase64,
            },
          },
          {
            text: "Transcribe the audio above verbatim and reply directly to the question or statement asked in the audio. Return your output strictly as a JSON object with 'transcription' and 'reply' keys.",
          },
        ];
      } else {
        promptParts = [
          {
            text: `User query: "${textInput}". Return JSON object with 'transcription': "${textInput}" and 'reply': "<your direct answer>".`,
          },
        ];
      }

      // Try primary model (gemini-3.7-flash) with fallback to gemini-flash-latest and gemini-3.1-flash-lite on transient 503/429
      const candidateModels = ["gemini-3.7-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"];
      let lastError: any = null;
      let response: any = null;

      for (const model of candidateModels) {
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            response = await ai.models.generateContent({
              model,
              contents: promptParts,
              config: {
                systemInstruction,
                responseMimeType: "application/json",
                temperature: 0.7,
              },
            });
            if (response && response.text) {
              break;
            }
          } catch (err: any) {
            lastError = err;
            const errMsg = err?.message || String(err);
            const isTransient =
              errMsg.includes("503") ||
              errMsg.includes("UNAVAILABLE") ||
              errMsg.includes("high demand") ||
              errMsg.includes("429") ||
              errMsg.includes("RESOURCE_EXHAUSTED") ||
              errMsg.includes("FetchError");

            if (isTransient && attempt === 0) {
              // Quick backoff before retrying same model
              await new Promise((r) => setTimeout(r, 600));
              continue;
            }
            // If still failing, break inner loop to try next model in fallback chain
            break;
          }
        }
        if (response && response.text) {
          break;
        }
      }

      if (!response || !response.text) {
        throw lastError || new Error("All model endpoints are temporarily unavailable. Please try again.");
      }

      const responseText = response.text || "{}";
      let parsedData: { transcription?: string; reply?: string } = {};

      try {
        parsedData = JSON.parse(responseText);
      } catch (parseErr) {
        // Fallback if JSON format was slightly imperfect
        parsedData = {
          transcription: textInput || "Audio input received",
          reply: responseText,
        };
      }

      return res.json({
        success: true,
        transcription: parsedData.transcription || (textInput ? textInput : "Audio transcribed"),
        reply: parsedData.reply || "Acknowledge.",
      });
    } catch (error: any) {
      console.error("Error processing voice chat:", error);
      return res.status(500).json({
        error: error.message || "Failed to process audio transcription and reply.",
      });
    }
  });

  // Text-only fallback route for quick chat
  app.post("/api/chat", async (req, res) => {
    try {
      const { message } = req.body;
      if (!message) {
        return res.status(400).json({ error: "Message is required" });
      }

      const ai = getGenAI();
      const response = await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents: message,
        config: {
          systemInstruction: "You are a brutalist AI companion: direct, sharp, clean, no filler.",
        },
      });

      return res.json({
        success: true,
        reply: response.text || "",
      });
    } catch (error: any) {
      console.error("Error in chat:", error);
      return res.status(500).json({ error: error.message || "Internal server error" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[BRUTALIST SERVER] running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
