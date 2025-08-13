
import json
import random
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

from embed_index import load_index, search_topk

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RANKER_PATH = ARTIFACTS_DIR / "ranker.pkl"

@dataclass
class Pair:
    q: str
    a: str
    y: int  # 1 good, 0 bad

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b) + 1e-9)
    return float((na * nb).sum())

def _embed(texts: List[str], embedder):
    return np.array(embedder.encode(texts, normalize_embeddings=True))

def _build_training_pairs(training_queries: List[Dict[str, Any]], docs: List[Dict[str, Any]], id2doc: Dict[int, Dict[str,Any]], n_neg: int = 3) -> List[Pair]:
    pairs = []
    for item in training_queries:
        q = item["query"]
        tgt = int(item["intent_id"])
        if tgt not in id2doc:
            continue
        pos = id2doc[tgt]["steps"]
        pairs.append(Pair(q, pos, 1))
        # negatives
        other_ids = [d["intent_id"] for d in docs if d["intent_id"] != tgt]
        neg_ids = random.sample(other_ids, k=min(n_neg, len(other_ids)))
        for nid in neg_ids:
            pairs.append(Pair(q, id2doc[nid]["steps"], 0))
    return pairs

def train_ranker(training_json_path: str):
    # Load embedder and docs
    nn, X, docs, embedder = load_index()
    id2doc = {d["intent_id"]: d for d in docs}

    data = json.loads(Path(training_json_path).read_text(encoding="utf-8"))
    training_queries = data.get("training_queries", [])

    pairs = _build_training_pairs(training_queries, docs, id2doc, n_neg=3)

    # Features: cosine similarity between q and a
    q_emb = _embed([p.q for p in pairs], embedder)
    a_emb = _embed([p.a for p in pairs], embedder)
    cos = np.array([_cosine(q_emb[i], a_emb[i]) for i in range(len(pairs))]).reshape(-1,1)
    y = np.array([p.y for p in pairs])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(cos, y)

    joblib.dump(clf, RANKER_PATH)

def score_candidates(query: str, candidates: List[Dict[str,Any]]):
    # Load ranker & embedder
    from embed_index import _load_embedder
    clf = joblib.load(RANKER_PATH)
    embedder = _load_embedder()
    qv = _embed([query], embedder)[0]
    scores = []
    for c in candidates:
        av = _embed([c["steps"]], embedder)[0]
        sim = np.dot(qv/ (np.linalg.norm(qv)+1e-9), av/ (np.linalg.norm(av)+1e-9))
        proba = float(clf.predict_proba([[sim]])[0,1])
        scores.append({**c, "rank_score": proba})
    scores.sort(key=lambda x: x["rank_score"], reverse=True)
    return scores
