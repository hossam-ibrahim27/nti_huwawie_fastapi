import logging

import groq
from fastapi import APIRouter, File, HTTPException, UploadFile
from groq import Groq

transcribeRoute = APIRouter(prefix="/api/v1", tags=["api_v1"])
logger = logging.getLogger(__name__)

EXT = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}
MAX_BYTES = 10 * 1024 * 1024

_client: Groq | None = None


def get_client() -> Groq:
    # Created lazily so the app can start even before env vars are loaded
    global _client
    if _client is None:
        _client = Groq()  # reads GROQ_API_KEY from the environment
    return _client


@transcribeRoute.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: str | None = None):
    # Browsers send e.g. "audio/webm;codecs=opus", so drop the codec part
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in EXT:
        raise HTTPException(415, "Unsupported audio format.")

    data = file.file.read()
    if not data:
        raise HTTPException(400, "The recording is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Recording is too large.")

    kwargs = {"language": language} if language else {}
    try:
        result = get_client().audio.transcriptions.create(
            file=(f"voice.{EXT[ctype]}", data),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0,
            **kwargs,
        )
    except groq.RateLimitError:
        raise HTTPException(429, "Too many voice requests. Please try again in a moment.")
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(502, "Transcription failed. Please try again.")

    return {"text": (result.text or "").strip()}