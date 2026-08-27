from contextlib import asynccontextmanager
from typing import List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline_utils import MODEL_VERSION
from scoring_service import (
    ScoringService,
    ValidationError as ServiceValidationError,
    DomainValidationError,
    ModelLoadError,
    PolicyLoadError,
    AuditFailureError,
    ScoringServiceError
)

# Global service instance
scoring_service = None
service_initialization_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scoring_service, service_initialization_error
    try:
        # Initialize the scoring service once during application startup
        scoring_service = ScoringService()
    except Exception as e:
        service_initialization_error = str(e)
    yield
    # Cleanup (if needed)

app = FastAPI(
    title="Razorpay Return Risk Scorer API",
    description="Defense-only risk scoring API estimating return/RTO risk using synthetic data assumptions. All decisions are deterministically audited.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---

class OrderRequest(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order")
    amount_inr: float = Field(..., description="Order amount in INR")
    method: str = Field(..., description="Payment method (e.g., cod, upi, netbanking)")
    category: str = Field(..., description="Product category (e.g., apparel, electronics)")
    is_new_customer: int = Field(..., description="1 if new customer, 0 otherwise")
    past_orders: int = Field(..., description="Number of past successful orders")
    past_return_rate: float = Field(..., description="Historical return rate (0.0 to 1.0)")
    order_hour: int = Field(..., description="Hour of order creation (0-23)")
    is_weekend: int = Field(..., description="1 if weekend, 0 otherwise")
    is_late_night: int = Field(..., description="1 if late night order, 0 otherwise")
    delivery_distance_km: float = Field(..., description="Delivery distance in km")
    checkout_time_sec: float = Field(..., description="Checkout time in seconds")

    model_config = {
        "allow_inf_nan": False,
        "json_schema_extra": {
            "example": {
                "order_id": "demo_888",
                "amount_inr": 12000,
                "method": "cod",
                "category": "apparel",
                "is_new_customer": 1,
                "past_orders": 0,
                "past_return_rate": 0.1,
                "order_hour": 18,
                "is_weekend": 1,
                "is_late_night": 0,
                "delivery_distance_km": 15,
                "checkout_time_sec": 30
            }
        }
    }

class Contribution(BaseModel):
    feature: str
    raw_value: Any
    contribution: float
    direction: str
    reason_code: str
    reason_text: str

class ScoringResponse(BaseModel):
    order_id: str
    risk_score: float
    decision: str
    top_positive_model_contributions: List[Contribution]
    top_negative_model_contributions: List[Contribution]
    reason_codes: List[str]
    domain_signals: List[str]
    risk_score_reason: str
    review_threshold: float
    hold_threshold: float
    audit_id: str
    model_version: str
    policy_version: str

class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    policy_version: Optional[str] = None
    error: Optional[str] = None

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Returns the readiness status of the scoring service.
    Verifies that the model and policy configurations were loaded successfully.
    """
    if scoring_service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "NOT_READY",
                "error": "Service initialization failed. See server logs."
            }
        )
    
    return HealthResponse(
        status="READY",
        model_version=MODEL_VERSION,
        policy_version=scoring_service.policy_config.get('policy_version', 'UNKNOWN')
    )

@app.post("/score-order", response_model=ScoringResponse, tags=["Scoring"])
async def score_order(order: OrderRequest):
    """
    Scores an incoming order against the frozen return risk model.
    Produces a risk probability, a bounded decision (APPROVE/REVIEW/HOLD), 
    filtered mathematically-derived explanations, and records a permanent audit event.
    """
    if scoring_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Scoring service is currently unavailable due to initialization failure."
        )

    try:
        result = scoring_service.score_order(order.model_dump())
        return ScoringResponse(**result)

    except (ServiceValidationError, DomainValidationError) as e:
        # Invalid input structure or synthetic domain constraints violated
        raise HTTPException(status_code=400, detail=str(e))
        
    except (ModelLoadError, PolicyLoadError) as e:
        # Core components unavailable
        raise HTTPException(status_code=503, detail="Service configuration unavailable.")
        
    except AuditFailureError as e:
        # The decision could not be audited. Do NOT return a successful response!
        raise HTTPException(status_code=503, detail="Decision could not be durably recorded. Aborting.")
        
    except ScoringServiceError as e:
        # Unexpected internal service errors
        raise HTTPException(status_code=500, detail="Internal scoring failure.")
        
    except Exception as e:
        # Generic unhandled exception shield. Do not leak stack trace to client.
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")
