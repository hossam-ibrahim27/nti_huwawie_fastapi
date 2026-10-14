from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import ask, base , data , detect , transcribe

app = FastAPI(title="Lumina Model", description="FastAPI Backend for Lumina Model", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.baseRoute)
app.include_router(data.dataRoute)
app.include_router(ask.askRoute)
app.include_router(detect.detectRoute)
app.include_router(transcribe.transcribeRoute)