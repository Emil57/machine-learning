import pytest

from ml_platform.config import settings
from ml_platform.config.settings import Settings


def test_default_environment():
    assert settings.environment == "development"


def test_random_seed():
    assert settings.random_seed == 42


def test_debug_default():
    assert settings.debug is False


def test_environment_override(monkeypatch):
    monkeypatch.setenv("RANDOM_SEED", "123")

    settings = Settings()

    assert settings.random_seed == 123


def test_mlflow_default_settings():
    settings = Settings()

    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_registry_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_experiment_name == "default"


def test_mlflow_tracking_uri_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://localhost:5000"


def test_mlflow_registry_uri_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_REGISTRY_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_registry_uri == "http://localhost:5000"


def test_mlflow_experiment_name_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        "ca_house_prediction",
    )

    settings = Settings()

    assert settings.mlflow_experiment_name == "ca_house_prediction"


def test_serving_default_settings():
    settings = Settings()

    assert settings.serving_model_name == "f1-predictor"
    assert settings.serving_model_alias == "champion"
    assert settings.serving_host == "0.0.0.0"
    assert settings.serving_port == 8000


def test_serving_model_name_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_MODEL_NAME",
        "my-model",
    )

    settings = Settings()
    assert settings.serving_model_name == "my-model"


def test_serving_model_alias_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_MODEL_ALIAS",
        "production",
    )

    settings = Settings()
    assert settings.serving_model_alias == "production"


def test_serving_host_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_HOST",
        "127.0.0.1",
    )

    settings = Settings()

    assert settings.serving_host == "127.0.0.1"


def test_serving_port_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_PORT",
        "9000",
    )

    settings = Settings()

    assert settings.serving_port == 9000


def test_serving_port_must_be_valid():
    with pytest.raises(ValueError):
        Settings(serving_port=0)


def test_serving_port_cannot_exceed_maximum():
    with pytest.raises(ValueError):
        Settings(serving_port=65536)
