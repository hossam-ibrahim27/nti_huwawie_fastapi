from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import ask, base , data , detect

app = FastAPI(title="FastAPI Backend", description="FastAPI Backend", version="1.0.0")

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