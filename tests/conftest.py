"""Shared test configuration."""

import pytest

from app.config import Settings


@pytest.fixture(scope="session", autouse=True)
def disable_local_dotenv_loading():
    """Keep developer-specific .env values from changing test expectations."""
    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
