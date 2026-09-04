"""
Platform settings.

This module defines the application's configuration using
Pydantic Settings.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Platform configuration."""

    environment: str = "development"
    debug: bool = False
    random_seed: int = 42

    train_split: float = 0.8
    validation_split: float = 0.2

    log_level: str = "INFO"

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_registry_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "default"

    serving_model_name: str = "f1-predictor"
    serving_model_alias: str = "champion"
    serving_host: str = "0.0.0.0"
    serving_port: int = Field(default=8000, ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
