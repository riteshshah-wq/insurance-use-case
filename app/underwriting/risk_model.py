"""Risk scoring model for insurance underwriting."""

import json
import logging
from typing import List

from ..common.llm_client import generate
from ..common.models import ApplicantProfile, RiskAssessment, ClaimType
from .prompts import RISK_ANALYSIS_SYSTEM_PROMPT, RISK_ANALYSIS_USER_PROMPT

logger = logging.getLogger(__name__)


async def assess_risk(applicant: ApplicantProfile) -> RiskAssessment:
    """Score risk factors for an insurance applicant.

    Combines rule-based scoring with LLM narrative generation.

    Args:
        applicant: The applicant profile to assess.

    Returns:
        RiskAssessment with score, category, factors, and narrative.
    """
    # Rule-based risk scoring (reliable, deterministic)
    score, factors = _calculate_risk_score(applicant)
    category = _score_to_category(score)

    # Attempt LLM narrative enrichment
    narrative = None
    try:
        prompt = RISK_ANALYSIS_USER_PROMPT.format(
            name=applicant.name,
            age=applicant.age,
            coverage_type=applicant.coverage_type.value,
            coverage_amount=applicant.coverage_amount,
            health_conditions=", ".join(applicant.health_conditions) or "None reported",
            driving_record=applicant.driving_record or "Not applicable",
            property_value=f"${applicant.property_value:,.2f}" if applicant.property_value else "N/A",
            location=applicant.location or "Not specified",
            occupation=applicant.occupation or "Not specified",
        )
        raw = await generate(
            prompt=prompt,
            system_prompt=RISK_ANALYSIS_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.3,
        )
        data = _parse_json(raw)
        if data and "narrative" in data:
            narrative = data["narrative"]
    except Exception as e:
        logger.warning(f"LLM risk narrative failed: {e}")
        narrative = f"Rule-based assessment: {category} risk (score: {score:.1f}/100)."

    return RiskAssessment(
        risk_score=round(score, 1),
        risk_category=category,
        risk_factors=factors,
        narrative=narrative,
    )


def _calculate_risk_score(applicant: ApplicantProfile) -> tuple[float, List[str]]:
    """Calculate a composite risk score from applicant factors."""
    score = 25.0  # Base score
    factors = []

    # Age risk
    if applicant.age < 25:
        score += 15
        factors.append(f"Young driver/applicant (age {applicant.age})")
    elif applicant.age > 65:
        score += 10
        factors.append(f"Senior applicant (age {applicant.age})")

    # Health conditions
    high_risk_conditions = {"diabetes", "heart disease", "cancer", "copd", "stroke"}
    for condition in applicant.health_conditions:
        if condition.lower() in high_risk_conditions:
            score += 12
            factors.append(f"High-risk health condition: {condition}")
        else:
            score += 5
            factors.append(f"Health condition: {condition}")

    # Driving record (for auto)
    if applicant.coverage_type == ClaimType.AUTO:
        driving = (applicant.driving_record or "").lower()
        if driving == "dui":
            score += 30
            factors.append("DUI on driving record")
        elif driving == "major_violations":
            score += 20
            factors.append("Major driving violations")
        elif driving == "minor_violations":
            score += 8
            factors.append("Minor driving violations")

    # Coverage amount relative risk
    if applicant.coverage_amount > 500_000:
        score += 10
        factors.append(f"High coverage amount: ${applicant.coverage_amount:,.0f}")
    elif applicant.coverage_amount > 1_000_000:
        score += 15
        factors.append(f"Very high coverage amount: ${applicant.coverage_amount:,.0f}")

    return min(score, 100.0), factors


def _score_to_category(score: float) -> str:
    """Map numeric score to category."""
    if score <= 25:
        return "low"
    elif score <= 50:
        return "moderate"
    elif score <= 75:
        return "high"
    else:
        return "very_high"


def _parse_json(text: str) -> dict | None:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return None
