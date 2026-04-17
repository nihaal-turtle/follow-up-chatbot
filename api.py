from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    content: str
    score: float

class RAGResponse(BaseModel):
    query: str
    retrieved_chunks: list[RetrievedChunk]


# --- Mock knowledge base ---

CHUNKS = [
    {
        "chunk_id": "hi_001",
        "source": "Health Insurance Basics Guide, Section 2",
        "content": (
            "A deductible is the amount you pay for covered health care services "
            "before your insurance plan starts to pay. For example, if your deductible "
            "is $1,000, you pay the first $1,000 of covered services yourself. After "
            "you meet your deductible, you usually pay only a copayment or coinsurance "
            "for covered services, and your insurance company pays the rest."
        ),
    },
    {
        "chunk_id": "hi_002",
        "source": "Plan Comparison Document, Section 1",
        "content": (
            "HMO (Health Maintenance Organization) plans generally have lower premiums "
            "and require you to choose a primary care physician (PCP). You need a referral "
            "from your PCP to see a specialist. Out-of-network care is typically not covered "
            "except in emergencies. PPO (Preferred Provider Organization) plans offer more "
            "flexibility — you can see any doctor without a referral, but staying in-network "
            "costs less."
        ),
    },
    {
        "chunk_id": "hi_003",
        "source": "Open Enrollment FAQ, Section 4",
        "content": (
            "Open enrollment is the yearly period when you can sign up for health insurance "
            "or change your existing plan. For plans on the Health Insurance Marketplace, "
            "open enrollment typically runs from November 1 to January 15. Outside of open "
            "enrollment, you can only enroll if you qualify for a Special Enrollment Period "
            "(SEP) due to a qualifying life event such as losing job-based coverage, getting "
            "married, or having a baby."
        ),
    },
    {
        "chunk_id": "hi_004",
        "source": "Cost-Sharing Explained, Section 3",
        "content": (
            "An out-of-pocket maximum is the most you have to pay for covered services in "
            "a plan year. After you spend this amount on deductibles, copayments, and "
            "coinsurance, your health plan pays 100% of the costs of covered benefits. "
            "For 2024, the out-of-pocket maximum for Marketplace plans is $9,450 for an "
            "individual and $18,900 for a family."
        ),
    },
    {
        "chunk_id": "hi_005",
        "source": "Prescription Drug Coverage Guide, Section 1",
        "content": (
            "Most health insurance plans use a drug formulary — a tiered list of covered "
            "medications. Tier 1 includes low-cost generic drugs, Tier 2 covers preferred "
            "brand-name drugs, and Tier 3 includes non-preferred brand-name drugs. Some plans "
            "have a Tier 4 for specialty drugs, which can have significantly higher cost-sharing. "
            "Always check if your medication is on the plan's formulary before enrolling."
        ),
    },
    {
        "chunk_id": "hi_006",
        "source": "Preventive Care Coverage Policy, Section 2",
        "content": (
            "Under the Affordable Care Act (ACA), most health insurance plans are required "
            "to cover a set of preventive services at no cost to you — meaning no copay or "
            "coinsurance — even if you haven't met your deductible. These services include "
            "annual wellness visits, blood pressure screenings, cholesterol checks, certain "
            "cancer screenings, and recommended vaccines."
        ),
    },
    {
        "chunk_id": "hi_007",
        "source": "Network Coverage Explainer, Section 5",
        "content": (
            "In-network providers have agreed to negotiated rates with your insurance company, "
            "meaning your cost-sharing will be lower. Out-of-network providers have not agreed "
            "to these rates. If you see an out-of-network provider, you may be responsible for "
            "the difference between what your insurer pays and what the provider charges — this "
            "is called balance billing. Always verify a provider's network status before "
            "scheduling an appointment."
        ),
    },
    {
        "chunk_id": "hi_008",
        "source": "HSA and FSA Overview, Section 1",
        "content": (
            "A Health Savings Account (HSA) is a tax-advantaged account available to people "
            "enrolled in a High-Deductible Health Plan (HDHP). You can contribute pre-tax "
            "dollars, and funds roll over year to year with no expiration. A Flexible Spending "
            "Account (FSA) is employer-established and also pre-tax, but typically has a "
            "use-it-or-lose-it rule — unused funds at year-end may be forfeited."
        ),
    },
]

QUERY_ROUTING = {
    "deductible":      ["hi_001", "hi_004"],
    "hmo":             ["hi_002"],
    "ppo":             ["hi_002"],
    "plan type":       ["hi_002"],
    "open enrollment": ["hi_003"],
    "out-of-pocket":   ["hi_004"],
    "maximum":         ["hi_004"],
    "prescription":    ["hi_005"],
    "drug":            ["hi_005"],
    "preventive":      ["hi_006"],
    "network":         ["hi_007"],
    "provider":        ["hi_007"],
    "hsa":             ["hi_008"],
    "fsa":             ["hi_008"],
    "savings account": ["hi_008"],
}


def mock_retrieve(query: str, top_k: int) -> list[dict]:
    query_lower = query.lower()
    matched_ids = []

    for keyword, chunk_ids in QUERY_ROUTING.items():
        if keyword in query_lower:
            for cid in chunk_ids:
                if cid not in matched_ids:
                    matched_ids.append(cid)

    if not matched_ids:
        matched_ids = random.sample([c["chunk_id"] for c in CHUNKS], min(top_k, len(CHUNKS)))

    matched_ids = matched_ids[:top_k]

    results = []
    for rank, cid in enumerate(matched_ids):
        chunk = next(c for c in CHUNKS if c["chunk_id"] == cid)
        score = round(0.95 - rank * 0.07 + random.uniform(-0.02, 0.02), 4)
        results.append({**chunk, "score": score})
    return results


@app.get("/rag/query")
def rag_query(query:str,top_k:int=3):
    retrieved = mock_retrieve(query, top_k)
    top_chunk_id = retrieved[0]["chunk_id"]

    return {
        "query":query,
        "retrieved_chunks":[RetrievedChunk(**c) for c in retrieved]
    }


@app.get("/")
async def root():
    return {"message":"Welcome to API"}
