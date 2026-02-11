from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.loader import load_models
from app.routers import ALL_ROUTERS

app = FastAPI(
    title="Patient Record Prediction API",
    description="API for managing patient records and predicting medical outcomes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_models()

for router in ALL_ROUTERS:
    app.include_router(router)

