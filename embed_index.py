
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    # Fallback to sklearn NearestNeighbors to avoid extra deps
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    NearestNeighbors = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_PATH = ARTIFACTS_DIR / "embed_index.joblib"
DOCS_PATH = ARTIFACTS_DIR / "docs.json"

def _load_embedder():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    return SentenceTransformer(MODEL_NAME)

def _encode(embedder, texts: List[str]) -> np.ndarray:
    return np.array(embedder.encode(texts, normalize_embeddings=True))

def build_from_csv(csv_path: str) -> None:
    """
    Build the intent document index from enhanced_ground_truth.csv.
    Each 'document' = intent_name + steps (optionally category).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    # Compose document text
    docs = []
    for _, row in df.iterrows():
        intent_id = int(row["intent_id"])
        title = str(row.get("intent_name", ""))
        steps = str(row.get("steps", ""))
        cat = str(row.get("category", ""))
        text = f"{title}\nCategory: {cat}\nSteps:\n{steps}"
        docs.append({
            "intent_id": intent_id,
            "intent_name": title,
            "text": text,
            "steps": steps,
            "primary_link": str(row.get("primary_link","")),
            "secondary_link": str(row.get("secondary_link","")),
            "category": cat
        })

    embedder = _load_embedder()
    X = _encode(embedder, [d["text"] for d in docs])

    # Fit simple NN index
    if NearestNeighbors is None:
        raise RuntimeError("sklearn not installed. pip install scikit-learn")
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(X)

    joblib.dump({"nn": nn, "X": X}, EMBED_PATH)
    DOCS_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index():
    data = joblib.load(EMBED_PATH)
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    return data["nn"], np.array(data["X"]), docs, _load_embedder()

def search_topk(query: str, k: int = 5):
    nn, X, docs, embedder = load_index()
    qv = _encode(embedder, [query])
    # sklearn cosine distance => smaller is more similar
    dists, idxs = nn.kneighbors(qv, n_neighbors=min(k, len(docs)))
    idxs = idxs[0].tolist(); dists = dists[0].tolist()
    out = []
    for i, dist in zip(idxs, dists):
        doc = docs[i]
        score = 1.0 - float(dist)  # convert distance to similarity
        out.append({
            "intent_id": doc["intent_id"],
            "intent_name": doc["intent_name"],
            "score": score,
            "steps": doc["steps"],
            "primary_link": doc["primary_link"],
            "secondary_link": doc["secondary_link"],
            "category": doc["category"]
        })
    return out
