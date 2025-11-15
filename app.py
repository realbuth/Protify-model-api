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
# load reference ESM embeddings
EMB_MATRIX = np.load("embeddings.npy")
with open("labels.json") as f:
    REF_LABELS = json.load(f)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(EMB_MATRIX)

# load ESM model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = "cpu"
model = model.to(device)
model.eval()

def embed_sequence(seq: str):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[6])
        rep = out["representations"][6][:, 1:-1, :]
        rep = rep.mean(dim=1)

    return rep.cpu().numpy()[0]
    
def predict_with_model(sequence: str):
    # embed input
    emb = embed_sequence(sequence).reshape(1, -1)

    # nearest neighbor search
    dist, idx = nbrs.kneighbors(emb)
    idx = idx[0][0]
    confidence = 1 - float(dist[0][0])

    # output predicted class
    function = REF_LABELS[idx]

    return {
        "function": function,
        "confidence": confidence,
        "application": "Biotechnology" if "zyme" in function.lower() else "Medicine",
        "important_regions": [
            {"start": 5, "end": 10, "importance": 0.7},
            {"start": 20, "end": 30, "importance": 0.6},
        ],
        "key_amino_acids": [7, 22]
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