"""Flyte workflow: Insurance Claims Severity Classifier.

Self-contained pipeline that:
1. Takes a list of insurance claim texts
2. Sends each to the LLM (GoldenRay inference API) for severity classification
3. Logs results to MLflow (experiment: insurance-claims/severity-classifier)
4. Returns a summary

Register & run:
    pyflyte register pipelines/severity_classifier.py --project insurance --domain development --copy none --version v1
    pyflyte run --remote pipelines/severity_classifier.py classify_claims_workflow \
        --claims '["Car accident on highway", "House fire destroyed kitchen"]'
"""

import json
import logging
from datetime import datetime
from typing import List

from flytekit import task, workflow, dynamic

logger = logging.getLogger(__name__)

# --- Configuration ---
GOLDENRAY_INFERENCE_URL = (
    "http://goldenray-ui-backend.goldenray-ui.svc.cluster.local:8000"
    "/api/v1/inference/demo-tenant/ollama-llama-model"
)
MLFLOW_TRACKING_URI = "http://goldenray-mlflow.mlops.svc.cluster.local:80"
MLFLOW_EXPERIMENT = "insurance-claims/severity-classifier"

SEVERITY_PROMPT_TEMPLATE = """You are an insurance claims severity classifier.
Classify the following insurance claim into exactly one severity level:
- LOW: minor damage, no injuries, estimated cost < $5,000
- MEDIUM: moderate damage, minor injuries, estimated cost $5,000-$50,000
- HIGH: major damage, serious injuries, estimated cost $50,000-$200,000
- CRITICAL: catastrophic damage, life-threatening injuries, estimated cost > $200,000

Claim: {claim_text}

Respond ONLY with a JSON object like:
{{"severity": "LOW|MEDIUM|HIGH|CRITICAL", "confidence": 0.0-1.0, "reasoning": "brief explanation", "estimated_cost": 12345}}"""


@task(cache=False, retries=1)
def classify_single_claim(claim_text: str, claim_index: int) -> dict:
    """Send a single claim to the LLM for severity classification."""
    import urllib.request

    prompt = SEVERITY_PROMPT_TEMPLATE.format(claim_text=claim_text)
    payload = json.dumps({
        "model": "ollama-llama-model",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 300,
    }).encode()

    req = urllib.request.Request(
        GOLDENRAY_INFERENCE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result = {
        "claim_index": claim_index,
        "claim_text": claim_text[:200],
        "severity": "UNKNOWN",
        "confidence": 0.0,
        "reasoning": "",
        "estimated_cost": 0,
        "llm_raw": "",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "pending",
    }

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read().decode())

        llm_text = ""
        if "choices" in body and body["choices"]:
            llm_text = body["choices"][0].get("message", {}).get("content", "")
        elif "response" in body:
            llm_text = body["response"]
        elif "content" in body:
            llm_text = body["content"]
        elif isinstance(body, str):
            llm_text = body

        result["llm_raw"] = llm_text[:500]

        try:
            start = llm_text.find("{")
            end = llm_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(llm_text[start:end])
                result["severity"] = parsed.get("severity", "UNKNOWN").upper()
                result["confidence"] = float(parsed.get("confidence", 0.5))
                result["reasoning"] = parsed.get("reasoning", "")[:200]
                result["estimated_cost"] = int(parsed.get("estimated_cost", 0))
                result["status"] = "classified"
            else:
                upper = llm_text.upper()
                for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    if sev in upper:
                        result["severity"] = sev
                        result["confidence"] = 0.5
                        result["reasoning"] = "Extracted from unstructured response"
                        result["status"] = "classified_fallback"
                        break
        except (json.JSONDecodeError, ValueError) as e:
            result["reasoning"] = f"JSON parse failed: {e}"
            result["status"] = "parse_error"

    except Exception as e:
        result["reasoning"] = f"LLM call failed: {str(e)[:200]}"
        result["status"] = "error"

    return result


@task(cache=False)
def build_summary_and_log(classifications: List[dict]) -> dict:
    """Aggregate results and log to MLflow in a single task."""
    # Aggregate
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "UNKNOWN": 0}
    total_cost = 0
    successful = 0

    for c in classifications:
        sev = c.get("severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        total_cost += c.get("estimated_cost", 0)
        if c.get("status") in ("classified", "classified_fallback"):
            successful += 1

    summary = {
        "total_claims": len(classifications),
        "successful_classifications": successful,
        "severity_distribution": severity_counts,
        "total_estimated_cost": total_cost,
        "classifications": classifications,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Log to MLflow
    mlflow_run_id = "not-logged"
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        run_name = f"severity-batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("total_claims", summary["total_claims"])
            mlflow.log_param("pipeline", "flyte-severity-classifier")
            mlflow.log_metric("successful_classifications", successful)
            mlflow.log_metric("total_estimated_cost", total_cost)
            for sev, count in severity_counts.items():
                mlflow.log_metric(f"severity_{sev.lower()}_count", count)
            mlflow.log_dict(summary, "classification_results.json")
            mlflow_run_id = mlflow.active_run().info.run_id
    except Exception as e:
        mlflow_run_id = f"mlflow-error-{str(e)[:100]}"

    summary["mlflow_run_id"] = mlflow_run_id
    return summary


@dynamic
def classify_all_claims(claims: List[str]) -> List[dict]:
    """Dynamically classify each claim (allows iteration over list)."""
    results = []
    for i, claim in enumerate(claims):
        result = classify_single_claim(claim_text=claim, claim_index=i)
        results.append(result)
    return results


@workflow
def classify_claims_workflow(claims: List[str]) -> dict:
    """Classify insurance claims by severity using LLM.

    Args:
        claims: List of insurance claim text descriptions

    Returns:
        Summary dict with severity distribution, costs, and MLflow run ID
    """
    classifications = classify_all_claims(claims=claims)
    summary = build_summary_and_log(classifications=classifications)
    return summary
