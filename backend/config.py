from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    APP_NAME: str = "SecureVision"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DEFAULT_DEPTH_M: float = 10.0
    DEFAULT_TURBIDITY: float = 0.5
    DEFAULT_WATER_TYPE: str = "bay_of_bengal"

    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    CONFIDENCE_THRESHOLD: float = 0.45
    NMS_THRESHOLD: float = 0.5

    MODEL_DIR: Path = Path("backend/models")
    DATA_DIR: Path = Path("backend/data")

    # GPS & Map
    ENABLE_GPS: bool = True
    GPS_TIMEOUT_SECONDS: int = 10
    DEFAULT_LAT: float = 13.0827
    DEFAULT_LON: float = 80.2707
    MAP_TILE_PROVIDER: str = "openstreetmap"
    OCEAN_LOOKUP_PRECISION_KM: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()