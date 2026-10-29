"""Flyte workflow: Claims Processing Pipeline.

Pipeline steps: extract → classify → route → log to MLflow

To register and run:
    pyflyte register pipelines/claims_pipeline.py
    pyflyte run pipelines/claims_pipeline.py claims_processing_workflow --claim_text "..."
"""

import json
import logging
from datetime import datetime
from typing import Optional

try:
    from flytekit import task, workflow, dynamic
    from flytekit.types.file import FlyteFile
except ImportError:
    # Stub decorators when flytekit isn't installed
    def task(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def workflow(fn=None, **kwargs):
        return fn if fn else lambda f: f

logger = logging.getLogger(__name__)


@task(cache=False, retries=2)
def extract_claim_fields(claim_text: str) -> dict:
    """Extract structured fields from claim text using LLM."""
    import asyncio
    from app.claims.extract import extract_claim

    result = asyncio.run(extract_claim(claim_text))
    return result.model_dump()


@task(cache=False, retries=2)
def classify_claim_severity(extraction: dict) -> dict:
    """Classify claim severity based on extracted data."""
    import asyncio
    from app.claims.classify import classify_claim
    from app.common.models import ClaimExtraction

    claim = ClaimExtraction(**extraction)
    result = asyncio.run(classify_claim(claim))
    return result.model_dump()


@task(cache=False)
def route_to_adjuster(extraction: dict, classification: dict) -> dict:
    """Route claim to appropriate adjuster team."""
    from app.claims.route import route_claim
    from app.common.models import ClaimExtraction, ClaimClassification

    claim = ClaimExtraction(**extraction)
    severity = ClaimClassification(**classification)
    result = route_claim(claim, severity)
    return result.model_dump()


@task(cache=False)
def log_to_mlflow(
    extraction: dict,
    classification: dict,
    routing: dict,
) -> str:
    """Log claim processing results to MLflow for tracking."""
    try:
        import mlflow

        mlflow.set_tracking_uri("http://10.0.0.113:30500")
        mlflow.set_experiment("insurance-claims-processing")

        with mlflow.start_run(run_name=f"claim-{extraction.get('claim_id', 'unknown')}"):
            # Log parameters
            mlflow.log_param("claim_type", extraction.get("claim_type", "unknown"))
            mlflow.log_param("severity", classification.get("severity", "unknown"))
            mlflow.log_param("adjuster_team", routing.get("adjuster_team", "unknown"))

            # Log metrics
            if extraction.get("claim_amount"):
                mlflow.log_metric("claim_amount", extraction["claim_amount"])
            mlflow.log_metric("confidence", classification.get("confidence", 0))
            mlflow.log_metric("priority", routing.get("priority", 0))

            # Log artifacts
            mlflow.log_dict(extraction, "extraction.json")
            mlflow.log_dict(classification, "classification.json")
            mlflow.log_dict(routing, "routing.json")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Logged to MLflow run: {run_id}")
            return run_id

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        return f"mlflow-unavailable-{datetime.now().isoformat()}"


@workflow
def claims_processing_workflow(claim_text: str) -> dict:
    """Full claims processing pipeline.

    Steps:
    1. Extract structured fields from claim text (LLM)
    2. Classify severity (LLM + rules)
    3. Route to adjuster team (rules)
    4. Log everything to MLflow
    """
    extraction = extract_claim_fields(claim_text=claim_text)
    classification = classify_claim_severity(extraction=extraction)
    routing = route_to_adjuster(extraction=extraction, classification=classification)
    mlflow_run_id = log_to_mlflow(
        extraction=extraction,
        classification=classification,
        routing=routing,
    )

    return {
        "extraction": extraction,
        "classification": classification,
        "routing": routing,
        "mlflow_run_id": mlflow_run_id,
    }
