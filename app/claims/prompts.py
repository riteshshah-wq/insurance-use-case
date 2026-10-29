"""LLM prompts for claims extraction and classification.

These prompts are optimized for small models (TinyLlama 1.1B) with:
- Explicit JSON structure with example
- Short, direct instructions
- Minimal prose to reduce hallucination
"""

EXTRACTION_SYSTEM_PROMPT = """You are a claim data extractor. Read the claim document and output ONLY a JSON object.

Output this exact JSON structure (use null for unknown fields):
{"claim_id": "string", "claimant_name": "string", "incident_date": "YYYY-MM-DD", "claim_amount": number, "claim_type": "auto|health|property|life", "description": "one sentence"}

Example:
{"claim_id": "CLM-001", "claimant_name": "Jane Doe", "incident_date": "2024-01-15", "claim_amount": 5000.0, "claim_type": "auto", "description": "Rear-end collision on highway"}

Output ONLY the JSON object, no other text."""

EXTRACTION_USER_PROMPT = """Extract claim information as JSON from this document:

{claim_text}

JSON:"""

CLASSIFICATION_SYSTEM_PROMPT = """You classify insurance claim severity. Output ONLY a JSON object.

Rules:
- LOW: amount < $5,000
- MEDIUM: amount $5,000-$25,000
- HIGH: amount $25,000-$100,000
- CRITICAL: amount > $100,000 or involves death/total loss

Output format: {"severity": "low|medium|high|critical", "confidence": 0.0-1.0, "reasoning": "brief reason"}"""

CLASSIFICATION_USER_PROMPT = """Classify this claim:
Type: {claim_type}
Amount: ${claim_amount}
Description: {description}

JSON:"""
