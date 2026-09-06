from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vision_api_host: str = "0.0.0.0"
    vision_api_port: int = 8000
    vision_frame_queue_capacity: int = 4
    vision_result_history: int = 30
    vision_max_frame_bytes: int = 6_000_000
    glasses_frame_interval_seconds: float = 1.0
    vision_backend: str = "mock"
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_model: str = ""
    vlm_timeout_seconds: float = 15.0
    component_knowledge_path: str | None = None
    component_knowledge_top_k: int = 3
    workflow_definitions_dir: str | None = None
    speech_mode: Literal["off", "computer", "glasses", "both"] = "off"


settings = Settings()
