"""LLM prompts for claims extraction and classification."""

EXTRACTION_SYSTEM_PROMPT = """You are an insurance claims processing assistant. Extract structured information from claim documents.

Always respond in valid JSON format with these fields:
- claim_id: The claim reference number (string)
- claimant_name: Full name of the claimant (string)
- incident_date: Date of the incident (string, YYYY-MM-DD format)
- claim_amount: Dollar amount claimed (number)
- claim_type: One of: auto, health, property, life (string)
- description: Brief summary of the incident (string)

If a field cannot be determined, use null."""

EXTRACTION_USER_PROMPT = """Extract the claim information from the following document:

---
{claim_text}
---

Respond with a JSON object containing: claim_id, claimant_name, incident_date, claim_amount, claim_type, description."""

CLASSIFICATION_SYSTEM_PROMPT = """You are an insurance claims severity classifier. Based on the claim details, classify the severity.

Rules:
- LOW: claim_amount < $5,000, routine claims
- MEDIUM: claim_amount $5,000-$25,000, standard processing
- HIGH: claim_amount $25,000-$100,000, senior adjuster required
- CRITICAL: claim_amount > $100,000, or involves injury/death/total loss

Respond in JSON with:
- severity: one of low, medium, high, critical
- confidence: 0.0 to 1.0
- reasoning: brief explanation"""

CLASSIFICATION_USER_PROMPT = """Classify the severity of this insurance claim:

Type: {claim_type}
Amount: ${claim_amount}
Description: {description}

Respond with JSON containing: severity, confidence, reasoning."""
