from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.inference import router as inference_router
from app.utils.logging_config import setup_logging


def create_app() -> FastAPI:
    setup_logging()

    api = FastAPI(
        title="Brain Tumor Segmentation + Survival Prediction API",
        version="1.0.0",
    )

    origins_env = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api.include_router(health_router)
    api.include_router(inference_router)
    return api


app = create_app()

