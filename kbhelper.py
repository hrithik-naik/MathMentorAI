import os
import torch
from langchain_community.document_loaders import DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Config paths
pdf_dir = "J:/Assignment/Aiplanet/testingknowledgebase/pdfs"
save_path = "enhanced_math_vector_db"
query_file = "J:/Assignment/Aiplanet/testingknowledgebase/queries.txt"

# Global vector store cache
_vector_db = None

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        print("🔄 Initializing FAISS vector DB...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",  # Good balance of speed/quality
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        _vector_db = FAISS.load_local(
            save_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS loaded.")
    return _vector_db


def is_out_of_knowledge_base(query, threshold_score=0.6):
    try:
        vector_db = get_vector_db()
        results = vector_db.similarity_search_with_relevance_scores(query, k=3)

        if not results:
            return True, []

        # Unpack top match and score
        top_doc, score = results[0]

        # If similarity score is too high (i.e., low similarity), it's OOB
        is_oob = score > threshold_score

        return is_oob, [doc for doc, _ in results],score

    except Exception as e:
        print(f"Error during knowledge base check: {e}")
        return True, []

