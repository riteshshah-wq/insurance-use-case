"""Flyte workflow: Model Training Pipeline.

Pipeline steps: load data → fine-tune → evaluate → register model

This pipeline demonstrates how to use Flyte + MLflow + GoldenRay
for model training workflows in the insurance domain.

To register and run:
    pyflyte register pipelines/training_pipeline.py
    pyflyte run pipelines/training_pipeline.py training_workflow --dataset_path "data/"
"""

import json
import logging
from typing import List, Optional

try:
    from flytekit import task, workflow, Resources
except ImportError:
    def task(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def workflow(fn=None, **kwargs):
        return fn if fn else lambda f: f
    class Resources:
        def __init__(self, **kwargs): pass

logger = logging.getLogger(__name__)


@task(cache=True, cache_version="1.0")
def load_training_data(dataset_path: str) -> dict:
    """Load and prepare training data from sample claims.

    In production, this would load from a data warehouse or
    labeled dataset stored in object storage.
    """
    import os
    from pathlib import Path

    claims_dir = Path(dataset_path) / "sample_claims"
    training_examples = []

    if claims_dir.exists():
        for claim_file in sorted(claims_dir.glob("*.txt")):
            text = claim_file.read_text(encoding="utf-8")
            training_examples.append({
                "id": claim_file.stem,
                "text": text,
                "source": str(claim_file),
            })

    logger.info(f"Loaded {len(training_examples)} training examples")

    return {
        "examples": training_examples,
        "count": len(training_examples),
        "source": dataset_path,
    }


@task(
    cache=False,
    retries=1,
    # requests=Resources(cpu="4", mem="16Gi", gpu="1"),
)
def fine_tune_model(
    training_data: dict,
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    epochs: int = 3,
    learning_rate: float = 2e-5,
) -> dict:
    """Fine-tune a model on insurance domain data.

    In production, this would use LoRA/QLoRA fine-tuning with a
    GPU-equipped Flyte task. For demo purposes, this simulates
    the training process.
    """
    logger.info(f"Fine-tuning {base_model} on {training_data['count']} examples")
    logger.info(f"Epochs: {epochs}, LR: {learning_rate}")

    # Simulated training metrics (in production, these come from actual training)
    training_result = {
        "base_model": base_model,
        "training_examples": training_data["count"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "final_loss": 0.42,
        "training_time_seconds": 1200,
        "model_path": f"models/insurance-{base_model.split('/')[-1]}-finetuned",
        "status": "simulated",  # Change to "completed" with real training
    }

    return training_result


@task(cache=False)
def evaluate_model(
    training_result: dict,
    training_data: dict,
) -> dict:
    """Evaluate the fine-tuned model on held-out examples.

    Metrics:
    - Extraction accuracy: Can the model extract claim fields?
    - Classification accuracy: Does severity match ground truth?
    - Response quality: Is the output well-structured JSON?
    """
    logger.info(f"Evaluating model: {training_result['model_path']}")

    # Simulated evaluation metrics
    evaluation = {
        "model_path": training_result["model_path"],
        "metrics": {
            "extraction_accuracy": 0.78,
            "classification_accuracy": 0.85,
            "json_parse_rate": 0.92,
            "avg_latency_ms": 450,
            "eval_examples": min(training_data["count"], 5),
        },
        "status": "simulated",
    }

    return evaluation


@task(cache=False)
def register_model(
    training_result: dict,
    evaluation: dict,
) -> str:
    """Register the model in MLflow Model Registry.

    If evaluation metrics meet thresholds, register as a new version.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri("http://10.0.0.113:30500")
        mlflow.set_experiment("insurance-model-training")

        with mlflow.start_run(run_name=f"train-{training_result['base_model'].split('/')[-1]}"):
            # Log training params
            mlflow.log_param("base_model", training_result["base_model"])
            mlflow.log_param("epochs", training_result["epochs"])
            mlflow.log_param("learning_rate", training_result["learning_rate"])

            # Log metrics
            for key, value in evaluation["metrics"].items():
                mlflow.log_metric(key, value)
            mlflow.log_metric("final_loss", training_result["final_loss"])

            # Log artifacts
            mlflow.log_dict(training_result, "training_result.json")
            mlflow.log_dict(evaluation, "evaluation.json")

            # Register model if quality thresholds met
            extraction_acc = evaluation["metrics"]["extraction_accuracy"]
            if extraction_acc >= 0.70:
                model_name = "insurance-claims-model"
                logger.info(f"Registering model {model_name} (accuracy: {extraction_acc})")
                # In production: mlflow.register_model(model_uri, model_name)

            run_id = mlflow.active_run().info.run_id
            return run_id

    except Exception as e:
        logger.warning(f"MLflow registration failed: {e}")
        return "mlflow-unavailable"


@workflow
def training_workflow(
    dataset_path: str = "data/",
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    epochs: int = 3,
) -> dict:
    """End-to-end model training pipeline.

    Steps:
    1. Load training data from sample claims
    2. Fine-tune base model on insurance domain
    3. Evaluate on held-out examples
    4. Register in MLflow if quality meets thresholds
    """
    training_data = load_training_data(dataset_path=dataset_path)
    training_result = fine_tune_model(
        training_data=training_data,
        base_model=base_model,
        epochs=epochs,
    )
    evaluation = evaluate_model(
        training_result=training_result,
        training_data=training_data,
    )
    run_id = register_model(
        training_result=training_result,
        evaluation=evaluation,
    )

    return {
        "training_result": training_result,
        "evaluation": evaluation,
        "mlflow_run_id": run_id,
    }
