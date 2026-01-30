# Maximo Primo

A premium, localized voice synthesis application powered by **Qwen3-TTS**.

## Features

- **Premium UI**: Dark-themed, high-end## ✨ Premium Experience: Maximo Primo
This release includes the **Maximo Primo** premium interface, a high-end web UI designed for the Qwen3-TTS engine.

### How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the application:

   ```bash
   python app.py
   ```

3. Open your browser to: `http://localhost:8000`

### Included Features

- **Aesthetic Design**: Glassmorphism, radial gradients, and fluid animations.
- **Voice-First**: Integrated microphone controls and visualizer.
- **Engine Core**: Powered by Qwen-Audio technology.

## 🚀 Deployment Options

### 1. Vercel (Frontend Only)

You can deploy just the beautiful UI to Vercel:

1. Push this repo to your GitHub.
2. Connect it to Vercel.
3. In your Vercel project settings, set an **Environment Variable**:
   - `MAXIMO_API_URL`: Your VPS backend URL (e.g., `http://your-vps-ip:8000`).
4. Vercel will serve the UI, and it will connect to your remote VPS engine.

### 2. VPS (Full Stack with Docker)

For a production-ready setup on your VPS:

1. Clone the repo to your VPS.
2. Ensure you have Docker and Docker Compose installed.
3. Run:

   ```bash
   docker-compose up -d --build
   ```

4. Access the app at `http://your-vps-ip:8000`.

---
*Developed for the panyeroa1/joler project.*
