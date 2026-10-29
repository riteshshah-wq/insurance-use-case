"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration for the Insurance AI application.

    All settings can be overridden via environment variables.
    The LLM client is designed to work with:
    - Small self-hosted models (TinyLlama, opt-125m) via GoldenRay MaaS
    - Future GPU models via GoldenRay MaaS
    - Cloud models (GPT-4o, Claude) via GoldenRay External Models gateway
    Just change GOLDENRAY_NAMESPACE and GOLDENRAY_ENDPOINT.
    """

    # GoldenRay MaaS API
    goldenray_api_url: str = "http://10.0.0.113:30888"
    goldenray_namespace: str = "demo-tenant"
    goldenray_endpoint: str = "TinyLlama"
    goldenray_api_key: Optional[str] = None

    # Alternative: External MaaS URL
    goldenray_external_url: str = "http://maas.goldenray.ai:30090"

    # MLflow tracking
    mlflow_tracking_uri: str = "http://10.0.0.113:30500"

    # Keycloak auth
    keycloak_url: str = "http://10.0.0.113:30808"
    keycloak_realm: str = "goldenray"

    # Application
    app_name: str = "Insurance AI Use Case"
    app_version: str = "1.0.0"
    debug: bool = False
    max_tokens: int = 200
    temperature: float = 0.3

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
