import io
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse

from .config import settings
from .inference import inference_engine
from .schemas import HealthResponse, InferenceMetadataResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if inference_engine.model is not None else "degraded",
        model_loaded=inference_engine.model is not None,
        device=str(inference_engine.device),
        weights_path=settings.MODEL_WEIGHTS_PATH
    )

@router.post("/api/v1/inference")
async def run_inference(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    if inference_engine.model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    try:
        flair_bytes = io.BytesIO(await flair.read())
        t1_bytes = io.BytesIO(await t1.read())
        t1ce_bytes = io.BytesIO(await t1ce.read())
        t2_bytes = io.BytesIO(await t2.read())

        nii_bytes, metadata = inference_engine.infer_volumes(
            flair_bytes,
            t1_bytes,
            t1ce_bytes,
            t2_bytes
        )

        return Response(
            content=nii_bytes,
            media_type="application/gzip",
            headers={
                "Content-Disposition": "attachment; filename=prediction_seg.nii.gz",
                "X-Inference-Metadata": json.dumps(metadata),
                "Access-Control-Expose-Headers": "X-Inference-Metadata"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v1/inference/metrics", response_model=InferenceMetadataResponse)
async def run_inference_metrics_only(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    if inference_engine.model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    try:
        flair_bytes = io.BytesIO(await flair.read())
        t1_bytes = io.BytesIO(await t1.read())
        t1ce_bytes = io.BytesIO(await t1ce.read())
        t2_bytes = io.BytesIO(await t2.read())

        _, metadata = inference_engine.infer_volumes(
            flair_bytes,
            t1_bytes,
            t1ce_bytes,
            t2_bytes
        )

        return JSONResponse(content=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

