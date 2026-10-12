from fastapi import APIRouter

baseRoute = APIRouter(prefix="/api/v1", tags=["api_v1"])


@baseRoute.get("/")
async def root():
    return {
        "status": "Successfully Run FastAPI Backend",
        "version": "1.0.0",
    }