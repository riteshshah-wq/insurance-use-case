"""Route classified claims to the appropriate adjuster team."""

from ..common.models import ClaimExtraction, ClaimClassification, ClaimRouting, Severity, ClaimType


# Adjuster team routing matrix
ROUTING_MATRIX = {
    ClaimType.AUTO: {
        Severity.LOW: ("Auto Express Team", 1, 3),
        Severity.MEDIUM: ("Auto Standard Team", 2, 7),
        Severity.HIGH: ("Auto Senior Adjusters", 3, 14),
        Severity.CRITICAL: ("Auto Special Investigations", 5, 30),
    },
    ClaimType.HEALTH: {
        Severity.LOW: ("Health Quick Claims", 1, 5),
        Severity.MEDIUM: ("Health Review Team", 2, 10),
        Severity.HIGH: ("Health Senior Review", 4, 21),
        Severity.CRITICAL: ("Health Special Cases", 5, 45),
    },
    ClaimType.PROPERTY: {
        Severity.LOW: ("Property Express", 1, 5),
        Severity.MEDIUM: ("Property Adjusters", 2, 10),
        Severity.HIGH: ("Property Senior Team", 3, 21),
        Severity.CRITICAL: ("Property Catastrophe Team", 5, 30),
    },
    ClaimType.LIFE: {
        Severity.LOW: ("Life Benefits Team", 2, 14),
        Severity.MEDIUM: ("Life Claims Team", 3, 21),
        Severity.HIGH: ("Life Senior Review", 4, 30),
        Severity.CRITICAL: ("Life Special Investigations", 5, 60),
    },
}

DEFAULT_ROUTING = ("General Claims Pool", 3, 14)


def route_claim(
    extraction: ClaimExtraction,
    classification: ClaimClassification,
) -> ClaimRouting:
    """Route a classified claim to the appropriate adjuster team.

    Args:
        extraction: Extracted claim data.
        classification: Severity classification.

    Returns:
        ClaimRouting with team assignment, priority, and timeline.
    """
    claim_type = extraction.claim_type
    severity = classification.severity

    if claim_type and claim_type in ROUTING_MATRIX:
        team, priority, days = ROUTING_MATRIX[claim_type].get(
            severity, DEFAULT_ROUTING
        )
    else:
        team, priority, days = DEFAULT_ROUTING

    notes = _build_routing_notes(extraction, classification)

    return ClaimRouting(
        adjuster_team=team,
        priority=priority,
        estimated_processing_days=days,
        notes=notes,
    )


def _build_routing_notes(
    extraction: ClaimExtraction,
    classification: ClaimClassification,
) -> str:
    """Generate routing notes for the adjuster."""
    parts = [f"Severity: {classification.severity.value} (confidence: {classification.confidence:.0%})"]

    if extraction.claim_amount:
        parts.append(f"Amount: ${extraction.claim_amount:,.2f}")
    if extraction.claim_type:
        parts.append(f"Type: {extraction.claim_type.value}")
    if classification.reasoning:
        parts.append(f"Classification basis: {classification.reasoning}")

    return " | ".join(parts)
