# CLAUDE.md

## 🎯 Mission & Success Criteria

**Goal:** Build a production-ready, portfolio-quality chord detection platform that demonstrates full-stack Docker orchestration and modern ML deployment patterns.

**Timeline:** 4-6 weeks (10-15 hours/week)

**Success Metrics:**
- ✅ 3 minutes usage deliveres pleasant experience without failures
- ✅ Real-time chord detection with <200ms latency
- ✅ Multi-user support with Google OAuth
- ✅ Deployed to Azure with CI/CD pipeline
- ✅ RAG-powered documentation assistant functional
- ✅ All services containerized with health checks

**Optimize for:**
- Clean architecture (this is portfolio material)
- Reusable patterns (apply to future projects)
- Learning Docker/K8s/microservices deeply
- Observable system (logs, metrics, traces)

**Do NOT optimize for:**
- Massive scale (single-instance services OK)
- Feature completeness (focus on core loop)
- Perfect UI polish (functional > pretty)

## Project Overview

Chord Detector is an application that detects guitar chords from audio input using deep learning. It currently runs as a PyQt5 desktop application for live chord recognition.

**Current Validation Accuracy:** 0.921

## Current State

### Dataset

Training data was recorded manually using a Fender FA-15 3/4 Acoustic guitar. Samples are labeled and preprocessed into Mel-spectrograms before training. The dataset is available on Hugging Face at `severyn-k/isolated-guitar-chords`.

### Model

The model uses a CRNN architecture with a custom loss function (`GuitarChordDistanceLoss`) that is musically aware:

- Models 12 root notes on a chromatic circle with circular semitone distance
- Penalizes major/minor confusion differently than unrelated chord confusion
- Includes a dedicated Noise class treated as maximally distant from all chords
- Combines cross-entropy with distance-based penalty controlled by `alpha`, `root_weight`, `temperature`, and `noise_distance` parameters

### Current Interface

PyQt5 desktop application (`app.py`) with:

- Input device selector
- Start/Stop button for detection
- Current chord display
- Confidence/status area

### Training

- Local training: `python train.py`
- Dataset preparation: `python -m scripts.prepare_dataset`
- Colab training available via `train_colab.ipynb`

## Deployment Goals

### Target Architecture

| Service                   | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| Frontend (Next.js)        | Web UI, mic capture via WebAudio API           |
| API Gateway (FastAPI)     | Auth, routing, validation, user management     |
| ML Worker (FastAPI)       | CRNN inference only                            |
| PostgreSQL                | User accounts, session data, usage history     |
| Redis                     | Task queue, session cache, rate limiting       |
| Nginx                     | Reverse proxy, SSL termination, load balancing |
| Vector DB (Qdrant/Chroma) | RAG for documentation Q&A                      |

### Implementation Phases

**Phase 1: API Contracts & Data Flow Design**

1. Define WebSocket message format (frontend ↔ API ↔ ML service)
2. Define WAV chunk encoding specification (22050 Hz, mono, ~200ms chunks)
3. Define JSON response schema for predictions
4. Document connection lifecycle (connect, stream, disconnect)

**Phase 2: ML Service Implementation**

1. Create `ml-service/` folder structure
2. Extract inference logic from `app.py` into dedicated modules
3. Implement WebSocket handler with per-connection rolling buffer (1s window)
4. Implement WAV decoding and mel-spectrogram extraction
5. Reference model architecture from `src/model.py`
6. Create Dockerfile (Python 3.11, PyTorch, librosa)
7. Test with sample WAV files

**Phase 3: API Gateway Implementation**

1. Create `api/` folder structure
2. Implement stateless WebSocket relay (frontend ↔ ML service)
3. Implement health check endpoint
4. Create Dockerfile
5. Test relay with mock messages

**Phase 4: Frontend Implementation (Next.js 15)**

1. Initialize Next.js 15 project in `frontend/`
2. Implement WebAudio microphone capture
3. Implement resampling to 22050 Hz
4. Implement WAV encoding for chunks
5. Implement WebSocket client
6. Create chord display UI (similar to PyQt5 app)
7. Copy chord images from `assets/UI/pictures/chords/`
8. Create Dockerfile (Node.js, nginx for production)

**Phase 5: Docker Compose Integration**

1. Create `docker-compose.yml` with all 3 services
2. Configure internal Docker network
3. Add health checks for all services
4. Test end-to-end flow: mic → frontend → API → ML → API → frontend → display
5. Document startup instructions

**Phase 6: Database & Auth**

1. Add PostgreSQL container
2. Connect API service to Postgres
3. Implement user model (SQLAlchemy/SQLModel)
4. Add Google OAuth flow
5. Store user sessions, track inference history

**Phase 7: Production Infrastructure**

1. Add Nginx as reverse proxy
2. Add Redis for caching/sessions
3. Implement WebSocket handling through Nginx
4. Add health checks and structured logging
5. SSL termination configuration

**Phase 8: Advanced Features & Deployment**

1. Add Celery/RQ workers with Redis backend
2. Introduce worker pattern for async ML jobs
3. Add Vector DB + RAG pipeline for docs Q&A
4. Implement rate limiting
5. Azure deployment (ACI or AKS)
6. CI/CD pipeline setup

### Architecture Overview (Phase 1-5)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js 15)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ WebAudio API captures microphone                                    │ │
│  │ Resamples to 22050 Hz, mono                                         │ │
│  │ Encodes chunks as WAV                                               │ │
│  │ Sends via WebSocket every ~200ms                                    │ │
│  │ Displays chord predictions received from server                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket: WAV chunks (~200ms audio)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI) - STATELESS                    │
│                                                                          │
│  - No buffers                                                            │
│  - No ML logic                                                           │
│  - Pure WebSocket relay between Frontend and ML Service                  │
│  - Future: auth, rate limiting, routing (Phase 6+)                       │
│                                                                          │
│  Endpoints:                                                              │
│  - WS /ws/audio  →  relay to ML Service                                  │
│  - GET /health   →  health check                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ WebSocket: relay WAV chunks
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          ML SERVICE (FastAPI)                             │
│                                                                          │
│  - Maintains per-connection rolling buffer (1s window)                   │
│  - Receives WAV chunks, decodes, appends to buffer                       │
│  - Every inference hop: extract features, run CRNN inference             │
│  - Sends prediction JSON back through WebSocket                          │
│                                                                          │
│  Inference logic extracted from existing app.py:                         │
│  - Audio normalization (RMS)                                             │
│  - Mel-spectrogram extraction (librosa)                                  │
│  - CRNN model with smoothing                                             │
│                                                                          │
│  Endpoints:                                                              │
│  - WS /ws/inference  →  audio in, predictions out                        │
│  - GET /health       →  health check                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Audio Specification

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 22050 Hz | Model trained on this rate |
| Channels | Mono | Single channel |
| Format | WAV (PCM) | Model trained on WAV |
| Chunk Duration | ~200ms | INFERENCE_HOP_SEC |
| Chunk Samples | 4,410 | 0.2s × 22050 Hz |
| Buffer Window | 1.0s | INFERENCE_WIN_SEC |
| Buffer Samples | 22,050 | 1.0s × 22050 Hz |
| Acceptable Latency | ≤500ms | User requirement |

### Folder Structure

```
chord-detector/
├── api/                          # API Gateway service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py               # FastAPI app, WebSocket relay
│       └── config.py             # ML service URL, etc.
│
├── ml-service/                   # ML Inference service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py               # FastAPI app, WebSocket handler
│       ├── inference.py          # Buffer management, CRNN inference
│       ├── audio.py              # WAV decoding, mel-spectrogram
│       ├── config.py             # Audio params from src/config.py
│       └── model/
│           └── crnn_best.pt      # Model checkpoint (copied or mounted)
│
├── frontend/                     # Next.js 15 web application
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   └── page.tsx          # Main chord detection UI
│       ├── components/
│       │   ├── ChordDisplay.tsx  # Chord visualization
│       │   ├── AudioCapture.tsx  # WebAudio + WebSocket client
│       │   └── DeviceSelector.tsx
│       ├── lib/
│       │   ├── audioProcessor.ts # Resampling, WAV encoding
│       │   └── websocket.ts      # WebSocket connection management
│       └── public/
│           └── assets/           # Chord images from assets/UI/
│
├── docker-compose.yml            # Orchestration
├── src/                          # Existing model architecture (shared)
├── checkpoints/                  # Existing model checkpoints
├── app.py                        # Existing PyQt5 app (kept for reference)
└── CLAUDE.md
```

### WebSocket Message Formats

**Frontend → API → ML Service:**
```
Binary frame: WAV audio chunk
- Header: standard WAV header (44 bytes)
- Data: PCM 16-bit or Float32, mono, 22050 Hz
- Duration: ~200ms (4,410 samples)
```

**ML Service → API → Frontend:**
```json
{
  "type": "prediction",
  "chord": "Am",
  "confidence": 0.92,
  "timestamp": 1704067200.123
}
```

**Connection Lifecycle:**
```json
// On connect (ML Service → Frontend)
{"type": "connected", "message": "Ready for audio"}

// On error
{"type": "error", "message": "Invalid audio format"}

// On disconnect
{"type": "disconnected", "reason": "Client closed connection"}
```

### Dependencies

**ML Service (requirements.txt):**
```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
torch>=2.0.0
librosa>=0.10.0
numpy<2.0
```

**API Gateway (requirements.txt):**
```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
httpx>=0.26.0
```

**Frontend (package.json dependencies):**
```json
{
  "next": "^15.0.0",
  "react": "^19.0.0",
  "react-dom": "^19.0.0",
  "typescript": "^5.0.0"
}
```

### Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Browser sample rate mismatch (48kHz vs 22kHz) | High | Implement resampling in AudioWorklet |
| WebSocket connection drops | Medium | Implement reconnection logic in frontend |
| librosa Docker dependencies | Medium | Use slim Python image with libsndfile |
| Model loading time on startup | Low | Load model once at service startup |
| Per-connection buffer memory | Low | Limit max connections, cleanup on disconnect |

### Success Criteria for Phase 1-5

- [ ] All 3 services start with `docker-compose up`
- [ ] Frontend captures microphone audio in browser
- [ ] Audio streams through API to ML service
- [ ] ML service returns chord predictions
- [ ] Predictions display in frontend UI
- [ ] End-to-end latency ≤500ms
- [ ] Services restart cleanly after container restart

### Production Architecture (Phase 7+)

```
[Browser] --WebSocket--> [Nginx] --> [API] --Redis Queue--> [ML Worker]
    ↑                                           |
    └───────────── Redis Pub/Sub ───────────────┘
```

### Service Justifications

**Redis:**

- Task queue: Audio upload → Redis queue → ML worker processes
- Session store: JWT tokens, active WebSocket connections
- Rate limiting: X inferences per minute per user
- Pub/sub: Push chord detection results back to frontend in real-time

**Nginx:**

- SSL termination: HTTPS at nginx, internal traffic is HTTP
- Load balancing: Round-robin between multiple ML workers
- Static file caching: Serve Next.js built assets
- WebSocket proxy: Route `/ws` connections for real-time audio streaming

**Vector DB for RAG:**

1. Chunk README, docstrings, comments into text segments
2. Embed each chunk
3. Store vectors in Qdrant/Chroma
4. User query → embed → find similar chunks → feed to LLM as context

### Worker Pattern

```
API receives audio → Pushes to Redis queue → Worker pulls job → Runs inference → Pushes result to Redis → API streams back to user
```

## License

Apache License 2.0
