import os
import dspy
import torch
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
pdf_dir = "J:/Assignment/Aiplanet/testingknowledgebase/pdfs"
save_path = "enhanced_math_vector_db"
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Configure DSPy with Gemini
lm = dspy.LM(
    "gemini/gemini-2.0-flash",
    api_key=api_key,
    max_tokens=1000,
    temperature=0.7,
)
dspy.configure(lm=lm)

# Load FAISS vector database
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True
    },
    encode_kwargs={"normalize_embeddings": True}
)
vector_db = FAISS.load_local(
    save_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# DSPy signature for answering questions
class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="Relevant context from knowledge base")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The detailed answer to the question")

# FAISS-based retriever
def faiss_retriever(query, k=3):
    try:
        results = vector_db.similarity_search(query, k=k)
        passages = [doc.page_content for doc in results]
        return passages
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return ["No relevant context found."]

# Simple feedback function
def provide_feedback(is_positive):
    if is_positive:
        print("✓ Positive feedback received - Answer was helpful!")
    else:
        print("✗ Negative feedback received - Answer needs improvement!")

# RAG Module using DSPy
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        retrieved_passages = faiss_retriever(question, k=3)
        context = "\n\n".join(retrieved_passages)
        result = self.generate_answer(context=context, question=question)
        result.retrieved_passages = retrieved_passages
        return result

rag = RAG()

def ask_question(question):
    try:
        response = rag(question=question)
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print("\nRetrieved Context:")
        if hasattr(response, 'retrieved_passages'):
            for i, passage in enumerate(response.retrieved_passages, 1):
                print(f"{i}. {passage}...")
        print("\n" + "="*80 + "\n")
        return response
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
        return None

def process_queries_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"Processing {len(queries)} queries from {file_path}\n")
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}:")
            result = ask_question(query)
            results.append(result)
        return results
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def test_system():
    sample_questions = [
        "What is the main topic discussed in the documents?",
        "Can you summarize the key findings?",
        "What are the main conclusions?"
    ]
    print("Testing RAG system with sample questions:\n")
    for question in sample_questions:
        ask_question(question)

if __name__ == "__main__":
    if vector_db:
        print("FAISS vector database loaded successfully!")
        print(f"Number of documents in vector store: {vector_db.index.ntotal}")
    
    queries_file = "J:/Assignment/Aiplanet/testingknowledgebase/queries.txt"
    
    if os.path.exists(queries_file):
        results = process_queries_from_file(queries_file)
    else:
        print(f"Queries file not found at {queries_file}")
        print("Running test with sample questions instead:\n")
        test_system()
    
    print("\n" + "="*50)
    print("FEEDBACK SYSTEM EXAMPLE")
    print("="*50)
    ask_question("What is machine learning?")
    # Uncomment to give feedback:
    # provide_feedback(True)
    # provide_feedback(False)
