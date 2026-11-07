from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    weights_path: str

class ClassStats(BaseModel):
    voxel_count: int
    volume_mm3: float

class TumorMetrics(BaseModel):
    whole_tumor: ClassStats
    tumor_core: ClassStats
    enhancing_tumor: ClassStats
    peritumoral_edema: ClassStats
    necrotic_core: ClassStats

class InferenceMetadataResponse(BaseModel):
    success: bool
    input_shape: list[int]
    output_shape: list[int]
    voxel_dimensions: list[float]
    metrics: TumorMetrics

