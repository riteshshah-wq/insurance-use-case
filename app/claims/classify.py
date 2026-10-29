"""Classify insurance claim severity."""

import json
import logging
from typing import Optional

from ..common.llm_client import generate
from ..common.models import ClaimExtraction, ClaimClassification, Severity
from .prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_PROMPT

logger = logging.getLogger(__name__)


async def classify_claim(extraction: ClaimExtraction) -> ClaimClassification:
    """Classify claim severity based on extracted data.

    Uses both rule-based logic and LLM analysis.

    Args:
        extraction: Previously extracted claim data.

    Returns:
        ClaimClassification with severity, confidence, and reasoning.
    """
    # Rule-based classification as primary (reliable with small models)
    severity = _rule_based_severity(extraction.claim_amount, extraction.claim_type)

    # Try LLM-based classification for richer reasoning
    reasoning = f"Rule-based: amount=${extraction.claim_amount}, type={extraction.claim_type}"
    confidence = 0.85

    try:
        prompt = CLASSIFICATION_USER_PROMPT.format(
            claim_type=extraction.claim_type or "unknown",
            claim_amount=extraction.claim_amount or 0,
            description=extraction.description or "No description available",
        )

        raw = await generate(
            prompt=prompt,
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.1,
        )

        data = _parse_json(raw)
        if data:
            llm_severity = data.get("severity", "").lower()
            if llm_severity in [s.value for s in Severity]:
                severity = Severity(llm_severity)
            reasoning = data.get("reasoning", reasoning)
            confidence = float(data.get("confidence", confidence))

    except Exception as e:
        logger.warning(f"LLM classification failed, using rules: {e}")

    return ClaimClassification(
        severity=severity,
        confidence=min(max(confidence, 0.0), 1.0),
        reasoning=reasoning,
    )


def _rule_based_severity(amount: Optional[float], claim_type: Optional[str]) -> Severity:
    """Determine severity from amount and type using business rules."""
    if amount is None:
        return Severity.MEDIUM

    if amount > 100_000:
        return Severity.CRITICAL
    elif amount > 25_000:
        return Severity.HIGH
    elif amount > 5_000:
        return Severity.MEDIUM
    else:
        return Severity.LOW


def _parse_json(text: str) -> Optional[dict]:
    """Safely parse JSON from LLM response."""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return None
