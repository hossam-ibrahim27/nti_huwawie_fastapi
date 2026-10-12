# VisionAI Backend

FastAPI service behind VisionAI. It exposes two endpoints:

- **`POST /api/v1/ask`**: RAG over the NICE guideline *Cataracts in adults (NG77)*. Answers come only from the guideline, with section and page citations.
- **`POST /api/v1/detect`**: eye-image detection with a trained YOLO model (4 classes).

> **Disclaimer:** This service summarises a published guideline and runs an automated screening model. It is not a medical device and does not replace clinical judgement or a diagnosis by a qualified ophthalmologist.

## Tech stack

| Area      | Technology |
|-----------|------------|
| API       | Python 3.12, FastAPI, Uvicorn |
| LLM       | Groq API via `langchain-groq` (TODO: model name) |
| Retrieval | LangChain, Cohere (`langchain-cohere`), Chroma (`langchain-chroma`) |
| Ingestion | `pypdf` (TODO: confirm the ingestion script / path) |
| Vision    | Ultralytics YOLO, Pillow |

## Project structure

```
FastAPI/
├── main.py                 # App entrypoint, CORS, router registration (TODO: confirm)
├── routes/
│   └── detect.py           # POST /api/v1/detect
├── models/best.pt          # Trained YOLO weights (TODO: confirm path)
├── ...                     # RAG code and Chroma index (TODO: list folders)
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## API

Interactive docs once the server is running: `http://localhost:8000/docs`

### `POST /api/v1/ask?k=5`

Request body:

```json
{ "question": "What types of anaesthesia are recommended for cataract surgery?" }
```

Response fields: `answer`, `refused`, `refusal_reason` (`out_of_scope` or `personal_medical_advice`), `confidence`, `top_score`, `retrieved_chunks[]`, `sources[]`, `claim_validation`, `warning`, `disclaimer`.

### `POST /api/v1/detect`

`multipart/form-data` with a `file` field (JPG, PNG or WEBP, max 10 MB). Optional query parameter `conf` (default `0.25`).

Response:

```json
{
  "image_width": 1024,
  "image_height": 768,
  "detections": [
    {
      "class_id": 0,
      "class_name": "cataract",
      "confidence": 0.93,
      "box": { "x1": 0.12, "y1": 0.2, "x2": 0.8, "y2": 0.9 }
    }
  ]
}
```

Box coordinates are normalised (0 to 1).

Error responses: `415` unsupported file type, `413` file too large, `400` unreadable image.

## Run locally

Requirements: Python 3.12, a Groq API key and a Cohere API key.

```bash
cd FastAPI
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (TODO: confirm its location, e.g. `FastAPI/.env`) and **never commit it**:

```env
GROQ_API_KEY=your_groq_key
COHERE_API_KEY=your_cohere_key
```

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

### Use CPU-only PyTorch (recommended)

`ultralytics` pulls PyTorch, and the default Linux build bundles CUDA libraries (several GB). For a smaller install, install the CPU build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Environment variables

| Variable          | Required | Description |
|-------------------|----------|-------------|
| `GROQ_API_KEY`    | yes      | Groq API key for the LLM |
| `COHERE_API_KEY`  | yes      | Cohere API key (TODO: embeddings and/or rerank) |
| `PORT`            | no       | Port to listen on (set automatically by most hosts, default `8000`) |

Variable names must match what the code reads. Commit a `.env.example` with empty values, never the real `.env`.

## Docker

```bash
cd FastAPI
docker build -t visionai-api .
docker run --rm -p 8000:8000 --env-file .env visionai-api
```

The image uses CPU-only PyTorch and listens on `$PORT` (default `8000`). `.env` is excluded by `.dockerignore`, so secrets are passed at runtime instead of baked into the image.

Make sure the model weights (`best.pt`) and the Chroma index are inside `FastAPI/` and committed (use Git LFS for large files), because the image is built from the repository.

## Deploy (Railway)

1. Push the repository to GitHub.
2. Create a service from the repo and set **Root Directory** to `/FastAPI` (Railway then builds the `Dockerfile`).
3. Add `GROQ_API_KEY` and `COHERE_API_KEY` under **Variables**.
4. Generate a public domain under **Networking**.
5. Add the frontend origin to the CORS `allow_origins` list in `main.py`, then set `VITE_API_URL` in the frontend project to the new backend URL and redeploy it.

Notes:

- This service is **not** suitable for Vercel serverless functions: the dependency bundle is far larger than their 500 MB limit and requests are capped at 4.5 MB, below the 10 MB image-upload limit.
- Free tiers may sleep after inactivity, so the first request can be slow while models load.
- ML libraries need memory. If the service is killed or runs out of memory, increase the plan's RAM.

## Troubleshooting

| Problem | Cause and fix |
|---------|---------------|
| `CUDA error: no kernel image is available for execution on the device` | The installed PyTorch build does not support your GPU. Install a matching build, or run on CPU with `device="cpu"` in `routes/detect.py`. |
| Dependency resolution fails on `multipart` | The package is named `python-multipart` (imported as `multipart`). Use `python-multipart` in `requirements.txt`. |
| GitHub rejects a push with `GH013` / "Push cannot contain secrets" | An API key was committed. Remove the file from git (`git rm --cached`), add `.env` to `.gitignore`, rewrite the history, and rotate the exposed key. |
| Detection returns "cataract" for non-eye photos | An object detector always picks one of its known classes. Upload clear close-ups of a single eye (a pre-check that rejects non-eye images is a planned improvement). |

## Limitations

- Answers are limited to NG77. Anything outside it is refused or flagged.
- The detection model was trained on a small public Roboflow dataset (`empty-ka3qc/cataract-gtqym`) and some training images appear to be animal eyes. Validate it on your own data before relying on it for human patients.
- Output is a screening aid, not a diagnosis.

## Credits

- [NICE guideline NG77: Cataracts in adults](https://www.nice.org.uk/guidance/ng77)
- Roboflow Universe dataset `cataract-gtqym` by `empty-ka3qc`
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)