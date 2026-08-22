# 🌐 Web Deployment Guide for Voice RAG Application

This guide walks you through deploying your Voice RAG application to the cloud so anyone can access it via a public URL with a working microphone interface.

---

## 🔑 Required Environment Variables
Before deploying, make sure you have your two API keys ready:
1. `SARVAM_API_KEY`: Your Sarvam AI key (from [dashboard.sarvam.ai](https://dashboard.sarvam.ai)) for Indian Language STT (`saarika:v2.5`).
2. `GROQ_API_KEY`: Your Groq Cloud key (from [console.groq.com](https://console.groq.com)) for ultra-fast LLM generation.

---

## 🚀 Option 1: Deploy on Render.com (Recommended Free/Easy)

1. Push this repository to your **GitHub** account.
2. Go to **[Render.com](https://dashboard.render.com)** and sign in.
3. Click **New +** $\rightarrow$ **Web Service**.
4. Select your GitHub repository (`Voice`).
5. Configure the service settings:
   - **Runtime:** `Python 3` or `Docker`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -m uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Under **Environment Variables**, add:
   - `SARVAM_API_KEY` = `your_sarvam_key`
   - `GROQ_API_KEY` = `your_groq_key`
   - `GROQ_MODEL` = `openai/gpt-oss-20b`
   - `PYTHON_VERSION` = `3.10.12`
7. Click **Create Web Service**.
8. Once built, Render will give you a public URL (e.g. `https://voice-rag-app.onrender.com`).

---

## 🚀 Option 2: Deploy on Railway.app (1-Click Docker)

1. Go to **[Railway.app](https://railway.app)** and log in with GitHub.
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select your `Voice` repository.
4. Go to **Variables** tab and add:
   - `SARVAM_API_KEY`
   - `GROQ_API_KEY`
   - `PORT` = `8000`
5. Railway will automatically detect the [Dockerfile](file:///d:/GitHub/Voice/Dockerfile) and build your live web server.
6. Under **Settings** $\rightarrow$ **Generate Domain** to get your public HTTPS URL.

---

## 🚀 Option 3: Deploy on Hugging Face Spaces (Free Docker Hosting)

1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** and click **Create new Space**.
2. Space SDK: Select **Docker** (Blank).
3. Clone the space repo or push this codebase to the Space repository.
4. In Space **Settings** $\rightarrow$ **Variables and secrets**, add:
   - `SARVAM_API_KEY`
   - `GROQ_API_KEY`
5. Your Space will build and launch your real-time Voice RAG UI at `https://huggingface.co/spaces/<username>/<spacename>`.

---

## 🐳 Option 4: Self-Hosted Cloud VM / Docker (AWS, DigitalOcean, GCP)

If running on a Linux VPS (Ubuntu/Debian):

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Voice

# 2. Configure .env file
echo "SARVAM_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env

# 3. Start container with Docker Compose
docker compose up -d --build

# 4. Access at http://<your-server-ip>:8000
```

---

## 🎙️ Note on Browser Microphone Permissions
Modern browsers require **HTTPS** (or `localhost`) to access the microphone.
Platforms like **Render**, **Railway**, and **Hugging Face Spaces** automatically provide free **SSL/HTTPS domains**, allowing the microphone recording to work on all desktop and mobile devices.
