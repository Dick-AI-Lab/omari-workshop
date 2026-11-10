"""Optional live-inference server for the website (real checkpoints, any parameter setting).

    pip install "fastapi>=0.110" "uvicorn>=0.27"
    python -m omari_us.serve --config configs/export.yaml --port 8000

The website probes http://localhost:8000/health (override with ?backend=URL). When reachable, controls that
have no precomputed result offer "Run live inference", and outputs are badged "Live inference".
Endpoints
  GET  /health                     {"status": "ok", "models": [...], "checkpoints": {...}}
  POST /predict  {image_png_base64, models?}          → {"predictions": {model: {probs, pred}}}
  POST /explain  {image_png_base64, model, method, target_class, layer?} → {"map": [[...]], "raw_min", "raw_max"}
The image is the *displayed* grey image; each model applies its own documented eval transform.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:  # optional dependency; request models must live at module level so FastAPI can resolve the annotations
    from pydantic import BaseModel

    class PredictReq(BaseModel):
        image_png_base64: str
        models: Optional[List[str]] = None

    class ExplainReq(BaseModel):
        image_png_base64: str
        model: str
        method: str
        target_class: int
        layer: Optional[str] = None
except ImportError:  # pragma: no cover
    PredictReq = ExplainReq = None  # type: ignore


def create_app(cfg: Dict):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from . import explainability as X
    from .export.website_export import Exporter, predict_one, to_input

    ex = Exporter(cfg)
    ex.load_models()
    app = FastAPI(title="OMARI workshop inference")
    app.add_middleware(CORSMiddleware, allow_origins=cfg.get("cors_origins", ["http://localhost:5173", "http://localhost:4173", "*"]), allow_methods=["*"], allow_headers=["*"])

    def decode(b64: str) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
                return np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        except Exception as e:  # pragma: no cover
            raise HTTPException(400, f"invalid image: {e}")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": list(ex.models), "checkpoints": {k: v.ckpt_id for k, v in ex.models.items()}}

    @app.post("/predict")
    def predict(req: PredictReq):
        g = decode(req.image_png_base64)
        ids = [m for m in (req.models or list(ex.models)) if m in ex.models]
        with torch.no_grad():
            return {"predictions": {m: predict_one(ex.models[m], g, ex.device) for m in ids}}

    @app.post("/explain")
    def explain(req: ExplainReq):
        if req.model not in ex.models:
            raise HTTPException(404, f"model {req.model} not loaded")
        lm = ex.models[req.model]
        g = decode(req.image_png_base64)
        try:
            m = X.explain(req.method, req.model, lm.model, to_input(lm, g, ex.device), req.target_class, req.layer)
        except (X.IncompatibleMethod, NotImplementedError) as e:
            raise HTTPException(422, str(e))
        a = m.numpy()
        return {"map": a.tolist(), "raw_min": float(a.min()), "raw_max": float(a.max()), "checkpoint": lm.ckpt_id}

    return app


def main():
    from .utils import load_yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(load_yaml(a.config)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
