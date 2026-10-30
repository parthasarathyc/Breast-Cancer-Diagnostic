import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

app = FastAPI(
    title="Breast CARE-MA Multi-Agent Decision Support API",
    description="Explainable 6-Agent AI Breast Cancer Diagnostic Decision Support System using native FastAPI orchestration.",
    version="2.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports directory for PDF downloads if needed
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/static/reports", StaticFiles(directory=reports_dir), name="reports")

# Include API routes
app.include_router(router)

@app.get("/")
def read_root():
    return {
        "project": "Breast CARE-MA",
        "description": "Explainable Multi-Agent AI Framework for Multimodal Breast Cancer Diagnostic Decision Support",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
