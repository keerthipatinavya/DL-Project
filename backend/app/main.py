import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.api.endpoints import router as api_router
from ml.config import BASE_DIR

app = FastAPI(
    title="Badminton AI Rally & Shot Analytics API",
    description="Deep Learning Based Spatiotemporal Video Analysis, Next-Shot Prediction, and Rally Outcome Forecasting.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for uploaded and annotated videos
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Include API Router
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "title": "Badminton AI Rally & Shot Prediction API",
        "docs": "/docs",
        "health": "/api/health",
        "model_info": "/api/model/info"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
