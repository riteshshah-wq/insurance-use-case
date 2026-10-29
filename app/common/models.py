"""Pydantic models for insurance domain objects."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import date


# --- Claims Models ---

class ClaimType(str, Enum):
    AUTO = "auto"
    HEALTH = "health"
    PROPERTY = "property"
    LIFE = "life"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClaimExtraction(BaseModel):
    claim_id: Optional[str] = None
    claimant_name: Optional[str] = None
    incident_date: Optional[str] = None
    claim_amount: Optional[float] = None
    claim_type: Optional[ClaimType] = None
    description: Optional[str] = None
    raw_llm_response: Optional[str] = None


class ClaimClassification(BaseModel):
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ClaimRouting(BaseModel):
    adjuster_team: str
    priority: int = Field(ge=1, le=5)
    estimated_processing_days: int
    notes: str


class ClaimProcessResult(BaseModel):
    extraction: ClaimExtraction
    classification: ClaimClassification
    routing: ClaimRouting


# --- Underwriting Models ---

class ApplicantProfile(BaseModel):
    name: str
    age: int
    gender: Optional[str] = None
    health_conditions: List[str] = Field(default_factory=list)
    driving_record: Optional[str] = None  # clean, minor_violations, major_violations, dui
    property_value: Optional[float] = None
    coverage_type: ClaimType
    coverage_amount: float
    location: Optional[str] = None
    occupation: Optional[str] = None


class RiskAssessment(BaseModel):
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_category: str  # low, moderate, high, very_high
    risk_factors: List[str]
    narrative: Optional[str] = None


class PremiumCalculation(BaseModel):
    base_premium: float
    risk_adjusted_premium: float
    monthly_premium: float
    annual_premium: float
    risk_score: float
    adjustments: dict


# --- Customer Agent Models ---

class ChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = Field(default_factory=list)


# --- API Request/Response Models ---

class ClaimTextRequest(BaseModel):
    text: str


class UnderwritingRequest(BaseModel):
    applicant: ApplicantProfile


class HealthResponse(BaseModel):
    status: str
    version: str
    model_endpoint: str
    namespace: str
