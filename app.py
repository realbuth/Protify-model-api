from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import numpy as np
import json
import esm


# ---------- Pydantic models ----------

class Region(BaseModel):
    start: int
    end: int
    importance: float


class PredictIn(BaseModel):
    sequence: str


class PredictOut(BaseModel):
    function: str
    confidence: float
    application: str
    important_regions: List[Region]
    key_amino_acids: List[int]


# ---------- FastAPI app ----------

app = FastAPI(
    title="Protify Model API",
    version="0.1.0",
)


# ---------- Load reference embeddings + labels ----------

try:
    EMB_MATRIX = np.load("embeddings.npy")        # shape (N, D)
    with open("labels.json") as f:
        REF_LABELS = json.load(f)                 # length N
except Exception as e:
    print("ERROR loading embeddings/labels:", e)
    EMB_MATRIX = None
    REF_LABELS = None


# ---------- Load ESM model (tiny, for deployment) ----------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ESM_MODEL, ESM_ALPHABET = esm.pretrained.esm2_t6_8M_UR50D()
ESM_MODEL = ESM_MODEL.to(DEVICE)
ESM_MODEL.eval()

BATCH_CONVERTER = ESM_ALPHABET.get_batch_converter()


# ---------- Helper: embed a sequence with ESM ----------

def _embed_sequence(seq: str) -> np.ndarray:
    # clean sequence
    seq = seq.upper()
    seq = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
    if not seq:
        return np.zeros(EMB_MATRIX.shape[1], dtype=np.float32)

    data = [("seq", seq)]
    _, _, tokens = BATCH_CONVERTER(data)
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        out = ESM_MODEL(tokens, repr_layers=[6])
        rep = out["representations"][6][:, 1:-1, :]   # drop CLS/EOS
        rep = rep.mean(dim=1)                         # [1, D]

    return rep.cpu().numpy()[0]


# ---------- Core prediction logic (real model) ----------

def predict_with_model(seq: str) -> Dict[str, Any]:
    if EMB_MATRIX is None or REF_LABELS is None:
        raise RuntimeError("Embeddings/labels not loaded on server.")

    q = _embed_sequence(seq)                          # [D]
    # normalize for cosine similarity
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    emb_norm = EMB_MATRIX / (
        np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-8
    )  # [N, D]

    sims = emb_norm @ q_norm                          # [N]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    function = str(REF_LABELS[best_idx])
    # map cosine sim (-1..1) → [0,1]
    confidence = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))

    # very simple heuristic regions based on sequence length
    L = max(1, len(seq))
    window = max(10, min(25, L // 4))

    regions: List[Dict[str, Any]] = []
    if L >= window:
        # first high-importance region
        regions.append({
            "start": 1,
            "end": min(window, L),
            "importance": 0.9,
        })
    if L >= 2 * window:
        # second medium-importance region
        regions.append({
            "start": max(1, L - window),
            "end": L,
            "importance": 0.75,
        })

    # key amino acids: just pick a few positions spread in the sequence
    key_positions: List[int] = []
    for frac in [0.15, 0.35, 0.55, 0.75, 0.9]:
        pos = int(frac * L)
        if 1 <= pos <= L:
            key_positions.append(pos)
    key_positions = sorted(set(key_positions))

    # map function → application
    low = function.lower()
    if "enzyme" in low or "catalysis" in low:
        application = "Biotechnology"
    elif "dna" in low or "binding" in low:
        application = "Medicine"
    else:
        application = "Agriculture"

    return {
        "function": function,
        "confidence": confidence,
        "application": application,
        "important_regions": regions,
        "key_amino_acids": key_positions,
    }


# ---------- FastAPI endpoint ----------

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    seq = inp.sequence.strip().upper()
    if not seq:
        raise HTTPException(status_code=422, detail="Sequence is empty.")
    if len(seq) > 5000:
        raise HTTPException(status_code=422, detail="Sequence too long (max 5000 aa).")

    result = predict_with_model(seq)
    return PredictOut(**result)