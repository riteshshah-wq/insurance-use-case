"""LLM prompts for underwriting risk analysis.

Optimized for small models (TinyLlama 1.1B) with explicit structure.
"""

RISK_ANALYSIS_SYSTEM_PROMPT = """You are an insurance risk analyst. Analyze the applicant and output ONLY a JSON object.

Output format:
{"risk_score": 0-100, "risk_category": "low|moderate|high|very_high", "risk_factors": ["factor1", "factor2"], "narrative": "2-3 sentence analysis"}

Higher score = more risk. Consider age, health, driving record, occupation, and coverage amount."""

RISK_ANALYSIS_USER_PROMPT = """Analyze this insurance applicant:

Name: {name}
Age: {age}
Coverage Type: {coverage_type}
Coverage Amount: ${coverage_amount:,.2f}
Health Conditions: {health_conditions}
Driving Record: {driving_record}
Property Value: {property_value}
Location: {location}
Occupation: {occupation}

JSON:"""
