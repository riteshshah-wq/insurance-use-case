"""Premium calculation based on risk assessment."""

from ..common.models import ApplicantProfile, RiskAssessment, PremiumCalculation, ClaimType


# Base annual premium rates by coverage type
BASE_RATES = {
    ClaimType.AUTO: 0.008,      # 0.8% of coverage
    ClaimType.HEALTH: 0.012,    # 1.2% of coverage
    ClaimType.PROPERTY: 0.005,  # 0.5% of coverage
    ClaimType.LIFE: 0.006,      # 0.6% of coverage
}


def calculate_premium(
    applicant: ApplicantProfile,
    risk: RiskAssessment,
) -> PremiumCalculation:
    """Calculate insurance premium based on applicant profile and risk.

    Args:
        applicant: The applicant profile.
        risk: The risk assessment result.

    Returns:
        PremiumCalculation with base, adjusted, monthly, and annual premiums.
    """
    base_rate = BASE_RATES.get(applicant.coverage_type, 0.01)
    base_premium = applicant.coverage_amount * base_rate

    # Risk multiplier: score 0-100 maps to 0.7x-2.5x
    risk_multiplier = 0.7 + (risk.risk_score / 100.0) * 1.8

    # Age adjustment
    age_factor = 1.0
    if applicant.age < 25:
        age_factor = 1.3
    elif applicant.age > 65:
        age_factor = 1.2

    # Coverage amount tier discount
    tier_factor = 1.0
    if applicant.coverage_amount > 500_000:
        tier_factor = 0.95  # 5% volume discount
    if applicant.coverage_amount > 1_000_000:
        tier_factor = 0.90  # 10% volume discount

    adjusted = base_premium * risk_multiplier * age_factor * tier_factor
    annual = round(adjusted, 2)
    monthly = round(annual / 12, 2)

    adjustments = {
        "base_rate": base_rate,
        "risk_multiplier": round(risk_multiplier, 3),
        "age_factor": age_factor,
        "tier_factor": tier_factor,
    }

    return PremiumCalculation(
        base_premium=round(base_premium, 2),
        risk_adjusted_premium=annual,
        monthly_premium=monthly,
        annual_premium=annual,
        risk_score=risk.risk_score,
        adjustments=adjustments,
    )
