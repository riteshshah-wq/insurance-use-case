"""Extract structured data from insurance claim documents using LLM."""

import json
import logging
from typing import Optional

from ..common.llm_client import generate
from ..common.models import ClaimExtraction
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT

logger = logging.getLogger(__name__)


async def extract_claim(claim_text: str) -> ClaimExtraction:
    """Extract structured claim fields from raw claim text.

    Sends the claim document to the LLM and parses the structured response.

    Args:
        claim_text: Raw text of the insurance claim document.

    Returns:
        ClaimExtraction with parsed fields.
    """
    prompt = EXTRACTION_USER_PROMPT.format(claim_text=claim_text)

    raw_response = await generate(
        prompt=prompt,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        max_tokens=300,
        temperature=0.1,
    )

    logger.info(f"LLM extraction response: {raw_response[:200]}")

    # Parse the LLM response
    try:
        # Try to find JSON in the response
        data = _parse_json_response(raw_response)
        return ClaimExtraction(
            claim_id=data.get("claim_id"),
            claimant_name=data.get("claimant_name"),
            incident_date=data.get("incident_date"),
            claim_amount=_parse_amount(data.get("claim_amount")),
            claim_type=data.get("claim_type"),
            description=data.get("description"),
            raw_llm_response=raw_response,
        )
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return ClaimExtraction(raw_llm_response=raw_response)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


def _parse_amount(value) -> Optional[float]:
    """Parse a dollar amount from various formats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None
