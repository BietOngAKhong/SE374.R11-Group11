from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Brain Tumor 3D UNet Worker"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    MODEL_WEIGHTS_PATH: str = str(Path(__file__).resolve().parent.parent / "weights" / "unet3d_final_fp32.pth")
    DEVICE: str = "cuda"
    PATCH_DEPTH: int = 128
    PATCH_HEIGHT: int = 128
    PATCH_WIDTH: int = 128
    OVERLAP: float = 0.5
    SECOND_LCC_MIN_VOXELS: int = 100
    ET_MIN_VOXELS: int = 50
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()

