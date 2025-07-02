import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Global Caching
# ----------------------------
EMBED_PATH = "math_guardrail_embeddings.pkl"
_cached_embeddings = None
_cached_texts = None

# ----------------------------
# Load Once into Memory
# ----------------------------
def load_saved_embeddings_once(filepath: str):
    global _cached_embeddings, _cached_texts
    if _cached_embeddings is None or _cached_texts is None:
        try:
            data = joblib.load(filepath)
            _cached_embeddings = data["embeddings"]
            _cached_texts = data["texts"]
            print(f"[✓] Loaded {len(_cached_texts)} examples from: {filepath}")
        except Exception as e:
            print(f"[✗] Error loading embeddings: {e}")
            _cached_embeddings = []
            _cached_texts = []

# ----------------------------
# Initialize Embedding Model
# ----------------------------
def initialize_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embedding_model
    except Exception as e:
        print(f"[✗] Error loading model: {e}")
        return None

# ----------------------------
# Query Similarity Check
# ----------------------------
def check_query_similarity(user_query: str, threshold=0.7):
    load_saved_embeddings_once(EMBED_PATH)
    model = initialize_embedding_model()
    if model is None or not _cached_embeddings:
        return False, 0.0

    try:
        query_vector = model.embed_query(user_query)
        similarities = cosine_similarity([query_vector], _cached_embeddings)
        max_sim_idx = np.argmax(similarities)
        max_sim_val = similarities[0, max_sim_idx]
        is_math = max_sim_val >= threshold
        return is_math, float(max_sim_val)
    except Exception as e:
        print(f"Error during similarity check: {e}")
        return False, 0.0

# ----------------------------
# Query Wrapper
# ----------------------------
def querying(query: str):
    return check_query_similarity(query)
