
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import joblib

from embed_index import search_topk, load_index

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CLARIFIER_PATH = ARTIFACTS_DIR / "clarifier.json"

def _best_margin_threshold(examples: List[Dict[str,Any]]) -> float:
    # Simple sweep over margins to maximize F1
    margins = [e["margin"] for e in examples]
    if not margins:
        return 0.1
    cand_ts = np.linspace(0.0, max(margins), 31)
    best_t, best_f1 = 0.1, -1.0
    for t in cand_ts:
        tp = sum(1 for e in examples if e["needs"] and e["margin"] <= t)
        fp = sum(1 for e in examples if (not e["needs"]) and e["margin"] <= t)
        fn = sum(1 for e in examples if e["needs"] and e["margin"] > t)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def tune_from_ambiguous_json(json_path: str, k:int=3):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    ex = []
    for item in data:
        q = item["query"]
        needs = bool(item.get("needs_clarification", False))
        cands = search_topk(q, k=k)
        if len(cands) < 2:
            continue
        margin = float(cands[0]["score"] - cands[1]["score"])
        ex.append({"query": q, "needs": needs, "margin": margin})
    t = _best_margin_threshold(ex) if ex else 0.1
    Path(CLARIFIER_PATH).write_text(json.dumps({"margin_t": t}, indent=2), encoding="utf-8")
    return {"margin_t": t, "num_examples": len(ex)}

def is_ambiguous(candidates: List[Dict[str,Any]], margin_t: float) -> bool:
    if len(candidates) < 2:
        return False
    margin = float(candidates[0]["score"] - candidates[1]["score"])
    return margin <= margin_t
