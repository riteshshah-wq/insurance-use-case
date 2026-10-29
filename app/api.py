"""FastAPI application for Insurance AI Use Cases."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .common.config import settings
from .common.models import (
    ClaimTextRequest,
    UnderwritingRequest,
    ChatRequest,
    ClaimProcessResult,
    HealthResponse,
)
from .claims.extract import extract_claim
from .claims.classify import classify_claim
from .claims.route import route_claim
from .underwriting.risk_model import assess_risk
from .underwriting.premium import calculate_premium
from .customer_agent.agent import handle_chat
from .customer_agent.rag import load_policy_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Insurance AI use cases powered by GoldenRay AI Platform",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load policy documents for RAG on startup."""
    count = load_policy_documents()
    logger.info(f"Loaded {count} policy documents for RAG")


# --- Health Check ---

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_endpoint=settings.goldenray_endpoint,
        namespace=settings.goldenray_namespace,
    )


# --- Claims Processing ---

@app.post("/api/claims/extract")
async def api_extract_claim(request: ClaimTextRequest):
    """Extract structured fields from claim document text."""
    try:
        result = await extract_claim(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/claims/classify")
async def api_classify_claim(request: ClaimTextRequest):
    """Extract and classify a claim's severity."""
    try:
        extraction = await extract_claim(request.text)
        classification = await classify_claim(extraction)
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/claims/process", response_model=ClaimProcessResult)
async def api_process_claim(request: ClaimTextRequest):
    """Full claims pipeline: extract → classify → route."""
    try:
        extraction = await extract_claim(request.text)
        classification = await classify_claim(extraction)
        routing = route_claim(extraction, classification)
        return ClaimProcessResult(
            extraction=extraction,
            classification=classification,
            routing=routing,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Underwriting ---

@app.post("/api/underwriting/assess")
async def api_assess_risk(request: UnderwritingRequest):
    """Perform risk assessment on an applicant."""
    try:
        result = await assess_risk(request.applicant)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/underwriting/premium")
async def api_calculate_premium(request: UnderwritingRequest):
    """Calculate premium for an applicant."""
    try:
        risk = await assess_risk(request.applicant)
        premium = calculate_premium(request.applicant, risk)
        return premium
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Customer Agent ---

@app.post("/api/agent/chat")
async def api_agent_chat(request: ChatRequest):
    """Customer service chat with RAG-powered responses."""
    try:
        result = await handle_chat(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
