import time
import logging
import traceback
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

from .config import settings
from .inference import inference_engine
from .schemas import HealthResponse

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

def cleanup_old_temp_files(temp_dir: Path, max_age_seconds: int = 86400) -> None:
    now = time.time()
    for file_path in temp_dir.glob("*.nii.gz"):
        try:
            if now - file_path.stat().st_mtime > max_age_seconds:
                file_path.unlink()
        except Exception:
            pass

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if inference_engine.model is not None else "degraded",
        model_loaded=inference_engine.model is not None,
        device=str(inference_engine.device),
        weights_path=settings.MODEL_WEIGHTS_PATH
    )

@router.post("/api/v1/inference")
async def run_uq_inference_endpoint(
    background_tasks: BackgroundTasks,
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    if inference_engine.model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    try:
        flair_bytes = await flair.read()
        t1_bytes = await t1.read()
        t1ce_bytes = await t1ce.read()
        t2_bytes = await t2.read()

        result = inference_engine.run_uq_inference(
            flair_bytes,
            t1_bytes,
            t1ce_bytes,
            t2_bytes,
            num_passes=20
        )

        background_tasks.add_task(cleanup_old_temp_files, inference_engine.temp_dir)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Inference processing failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/mask/{task_id}")
async def get_mask_endpoint(
    task_id: str,
    p: int = Query(50, description="Percentile level: 5, 50, or 95")
):
    mask_path = inference_engine.get_mask_file_path(task_id, p)
    if mask_path is None or not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Mask for task_id '{task_id}' with percentile p={p} was not found"
        )

    return FileResponse(
        path=str(mask_path),
        media_type="application/gzip",
        filename=f"{task_id}_p{p}.nii.gz"
    )
