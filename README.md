# Self-Hosted ChatGPT Clone with llama.cpp

---

## Irish Greyhound Racing Predictive Intelligence System

ML-powered prediction dashboard for Irish greyhound racing. Uses LightGBM + XGBoost on 99 features (form, pace, track fit, breeding) with a Seven-Dimensional Outcome Engine UI.

### Prerequisites

- Python 3.10+
- Node.js 18+
- Chrome (for Paddy Power odds scraper only)

### Quick Start

**One command (Windows):**
```batch
greyhound-start.bat
```

**Manual:**
```bash
# Terminal 1: API
python -m greyhound serve

# Terminal 2: Frontend
cd greyhound/frontend && npm install && npm run dev

# Open http://localhost:5173
```

### Key Commands

| Command | Description |
|---------|-------------|
| `python -m greyhound stats` | Database summary |
| `python -m greyhound train` | Train ML models |
| `python -m greyhound serve` | Start API (port 8000) |
| `python -m greyhound scrape-upcoming` | Fetch upcoming race cards from GRI |
| `python -m greyhound scrape-dogs` | Enrich dog profiles (Level 2) |
| `python -m greyhound scrape-extras` | Trainer/owner/sire stats (Level 3) |

**Pipeline:** Run `scrape-extras` first, then `train` to incorporate Level 3 features. The API returns 503 while a scraper holds the DB lock.

### UI

- **Dashboard**: Race cards by date/track, "Get New Races" button
- **Race Detail**: 7D analysis, TOP PICK badge, Kelly stakes, live odds
- **Performance**: Win rate, ROI, calibration

---

## 🎯 Project Overview

A production-ready, self-hosted ChatGPT-like web application that runs entirely on your own hardware using llama.cpp for inference. Built for Coursera capstone submission.

**Demo**: [Your public URL after deployment]

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Public Internet                       │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTPS (port 443)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Nginx Reverse Proxy                     │
│         - Routes / to frontend (static files)            │
│         - Routes /api to FastAPI backend                 │
│         - HTTPS/SSL termination                          │
└─────────────┬───────────────────┬───────────────────────┘
              │                   │
      ┌───────▼────────┐  ┌──────▼──────────┐
      │   Frontend     │  │   FastAPI       │
      │   (React +     │  │   Backend       │
      │    Vite)       │  │   (Python)      │
      │   Port 5173    │  │   Port 8000     │
      └────────────────┘  └──────┬──────────┘
                                 │ HTTP (localhost only)
                                 ▼
                    ┌────────────────────────┐
                    │   llama.cpp Server     │
                    │   (RTX 5090)           │
                    │   Port 8080            │
                    │   NOT exposed publicly │
                    └────────────────────────┘
```

### Data Flow

1. User opens `https://yourdomain.com` in browser
2. Nginx serves React frontend (static files)
3. User types message, React sends POST to `/api/chat`
4. Nginx forwards `/api/*` to FastAPI backend
5. FastAPI validates request, calls llama.cpp via HTTP
6. llama.cpp generates tokens using GPU
7. FastAPI streams tokens back via Server-Sent Events (SSE)
8. React displays tokens in real-time (streaming effect)
9. Conversation state stored in browser localStorage

## 🔑 Key Design Decisions

### Why FastAPI?
- Native async/await for streaming
- Excellent SSE support
- Fast performance
- Built-in OpenAPI docs
- Easy to extend

### Why llama.cpp?
- Direct GPU inference (RTX 5090)
- No Python overhead
- Battle-tested and fast
- Flexible API (supports OpenAI-compatible format)

### Why SSE (Server-Sent Events)?
- Simpler than WebSockets for one-way streaming
- HTTP-based (works through proxies/CDNs)
- Built-in reconnection in browsers
- Perfect for chat streaming

### Why Client-Side Storage?
- No database needed (simpler deployment)
- Fast (no network roundtrip)
- Privacy-friendly (data stays in browser)
- Easy session management with UUIDs

### Production Upgrade Path
- Add PostgreSQL for persistent conversations
- Add Redis for rate limiting across instances
- Add user authentication (JWT)
- Add conversation sharing
- Add multi-model support
- Add usage analytics

## 📁 Project Structure

```
ufo/
├── frontend/                    # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── LoadingIndicator.jsx
│   │   │   └── MarkdownRenderer.jsx
│   │   ├── services/
│   │   │   └── chatService.js
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── main.py             # FastAPI entry point
│   │   ├── llama_adapter.py    # llama.cpp API adapter
│   │   ├── rate_limiter.py     # Simple rate limiting
│   │   └── models.py           # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
│
├── nginx/
│   └── nginx.conf              # Reverse proxy config
│
├── docker-compose.yml           # Optional: containerized deployment
├── .env.example                 # Environment variables template
├── deployment/
│   ├── setup-https.sh          # Let's Encrypt setup
│   └── cloudflare-tunnel.sh    # Alternative: Cloudflare Tunnel
│
├── docs/
│   └── COURSERA_SUBMISSION.md  # Project writeup
│
└── README.md                    # This file
```

## 🚀 Technologies Used

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool (fast HMR)
- **Tailwind CSS**: Utility-first styling
- **marked.js**: Markdown rendering
- **highlight.js**: Code syntax highlighting

### Backend
- **FastAPI**: Modern Python web framework
- **uvicorn**: ASGI server
- **httpx**: Async HTTP client for llama.cpp calls
- **python-dotenv**: Environment configuration

### Infrastructure
- **Nginx**: Reverse proxy + static file serving
- **Let's Encrypt**: Free HTTPS certificates
- **Cloudflare Tunnel**: Alternative public access (no port forwarding)

### Model Inference
- **llama.cpp**: GPU-accelerated inference on RTX 5090

## 🔒 Security Features

1. **Private Model Server**: llama.cpp only accessible from localhost
2. **Rate Limiting**: Prevent API abuse (100 req/min per IP)
3. **Request Validation**: Pydantic models validate all inputs
4. **CORS Protection**: Only whitelisted origins allowed
5. **Request Size Limits**: 100KB max payload
6. **HTTPS**: All public traffic encrypted
7. **No Secrets in Frontend**: All config in backend

## 💰 Cost Analysis

**Hosting Costs: ~$0-10/month**
- Self-hosted on your hardware: $0
- Domain name: ~$10-15/year
- Cloudflare Tunnel: $0 (free tier)
- Let's Encrypt: $0 (free)

**vs AWS Serverless ChatGPT:**
- Lambda: $20-50/month (10K requests)
- OpenAI API: $20-100/month (10K requests)
- Total: $40-150/month

**Your solution is 100% free to run** (after hardware investment).

## 📊 Performance

- **Latency**: 50-200ms first token (depends on model size)
- **Throughput**: 30-100 tokens/sec (RTX 5090)
- **Concurrent Users**: 5-20 (depends on model size and VRAM)
- **Bottleneck**: GPU (single user gets full resources)

## 🎓 Coursera Learning Objectives

This project demonstrates:

✅ **Full-Stack Development**: React frontend + Python backend
✅ **API Design**: RESTful endpoints, streaming responses
✅ **Real-Time Communication**: Server-Sent Events (SSE)
✅ **System Architecture**: Multi-tier design, reverse proxy patterns
✅ **GPU Computing**: Hardware acceleration for AI inference
✅ **DevOps**: Deployment, HTTPS, reverse proxy configuration
✅ **Security**: Private backend, public frontend separation
✅ **Modern Web**: Responsive UI, real-time updates, markdown rendering

## 🚀 Quick Start

```bash
# 1. Start llama.cpp server (separate terminal)
./llama-server -m models/llama-3-8b.gguf -c 4096 --port 8080 --host 127.0.0.1

# 2. Start backend (separate terminal)
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Start frontend (separate terminal)
cd frontend
npm install
npm run dev

# 4. Open browser
http://localhost:5173
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment.

## 📝 License

MIT - Free for educational use

## 👤 Author

Coursera Cloud Computing Capstone Project
