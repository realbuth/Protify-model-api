from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re

app = FastAPI(title="Protify Model API")

# Validate amino acid sequence
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.I)

class PredictIn(BaseModel):
    sequence: str

class Region(BaseModel):
    start: int
    end: int
    importance: float

class PredictOut(BaseModel):
    function: str
    confidence: float
    application: str
    important_regions: Optional[List[Region]] = None
    key_amino_acids: Optional[List[int]] = None

def clean(seq: str) -> str:
    s = (seq or "").strip()
    if s.startswith(">"):
        s = "\n".join(ln for ln in s.splitlines() if not ln.startswith(">"))
    s = s.replace(" ", "").replace("\n", "").upper()
    if not s or not AA.match(s):
        raise HTTPException(status_code=400, detail="Invalid sequence (use standard amino acids).")
    return s

# ---------------- Your model logic (placeholder now) ----------------
def predict_with_model(seq: str) -> Dict[str, Any]:
    """
    Replace this later with your real ESM-2 inference.
    """
    function = "Enzyme catalysis" if ("H" in seq or "D" in seq) else "Structural protein"
    confidence = 0.87
    application = "Biotechnology" if function.startswith("Enzyme") else "Agriculture"
    return {
        "function": function,
        "confidence": confidence,
        "application": application,
        "important_regions": [
            {"start": 10, "end": 25, "importance": 0.9},
            {"start": 50, "end": 70, "importance": 0.8}
        ],
        "key_amino_acids": [15, 23, 58, 62]
    }
# -------------------------------------------------------------------

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    seq = clean(inp.sequence)
    if len(seq) > 5000:
        raise HTTPException(status_code=422, detail="Sequence too long (max 5000 aa).")
    result = predict_with_model(seq)

    fn = str(result.get("function"))
    cf = float(result.get("confidence"))
    app_field = str(result.get("application"))
    if app_field not in {"Medicine", "Agriculture", "Biotechnology"}:
        app_field = "Biotechnology"
    cf = max(0.0, min(1.0, cf))

    out = {
        "function": fn,
        "confidence": cf,
        "application": app_field
    }
    if "important_regions" in result:
        out["important_regions"] = result["important_regions"]
    if "key_amino_acids" in result:
        out["key_amino_acids"] = result["key_amino_acids"]

    return out