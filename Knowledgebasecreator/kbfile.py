import os
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathKnowledgeBase:
    def __init__(self, pdf_dir: str, save_path: str, query_file: str):
        self.pdf_dir = Path(pdf_dir)
        self.save_path = Path(save_path)
        self.query_file = Path(query_file)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True
            },
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_db = None
    
    def clean_math_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'(\$[^$]+\$)', r' \1 ', text)
        text = re.sub(r'(\\\[.*?\\\])', r' \1 ', text, flags=re.DOTALL)
        return text.strip()
    
    def load_and_process_documents(self) -> list[Document]:
        logger.info(f"📂 Loading PDFs from: {self.pdf_dir}")
        loader = DirectoryLoader(
            path=str(self.pdf_dir),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        raw_documents = loader.load()
        logger.info(f"📄 Loaded {len(raw_documents)} raw documents")
        
        processed_docs = []
        for doc in tqdm(raw_documents, desc="Processing documents"):
            cleaned_content = self.clean_math_text(doc.page_content)
            if len(cleaned_content.strip()) < 50:
                continue
                
            doc.metadata.update({
                'char_count': len(cleaned_content),
                'source_file': Path(doc.metadata.get('source', '')).name
            })
            
            doc.page_content = cleaned_content
            processed_docs.append(doc)
        
        logger.info(f"✅ Processed {len(processed_docs)} documents")
        return processed_docs
    
    def create_chunks(self, documents: list[Document]) -> list[Document]:
        logger.info("🔗 Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        filtered_chunks = [
            chunk for chunk in chunks
            if 100 <= len(chunk.page_content.strip()) <= 1500
        ]
        
        logger.info(f"📦 Created {len(filtered_chunks)} optimized chunks")
        return filtered_chunks
    
    def build_vector_database(self, force_rebuild: bool = True):
        if force_rebuild and self.save_path.exists():
            logger.info("🗑️ Removing existing vector database...")
            shutil.rmtree(self.save_path)
        
        documents = self.load_and_process_documents()
        chunks = self.create_chunks(documents)
        
        if not chunks:
            logger.error("❌ No chunks created. Check your PDF directory.")
            return
        
        logger.info("⚙️ Building FAISS vector database...")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        batch_size = 100
        if len(chunks) > batch_size:
            logger.info(f"📊 Processing {len(chunks)} chunks in batches of {batch_size}")
            self.vector_db = FAISS.from_documents(chunks[:batch_size], self.embedding_model)
            
            for i in tqdm(range(batch_size, len(chunks), batch_size), desc="Adding batches"):
                batch = chunks[i:i + batch_size]
                batch_db = FAISS.from_documents(batch, self.embedding_model)
                self.vector_db.merge_from(batch_db)
        else:
            self.vector_db = FAISS.from_documents(chunks, self.embedding_model)
        
        self.vector_db.save_local(str(self.save_path))
        logger.info(f"✅ Vector database saved to: {self.save_path}")
    
    def load_vector_database(self):
        if not (self.save_path / "index.faiss").exists():
            logger.error(f"❌ No vector database found at: {self.save_path}")
            return False
        
        logger.info("🔁 Loading existing FAISS index...")
        self.vector_db = FAISS.load_local(
            str(self.save_path),
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        return True
    
    def run_queries(self, top_k: int = 3):
        if not self.query_file.exists():
            logger.error(f"❌ Query file not found: {self.query_file}")
            return
        
        with open(self.query_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        if not queries:
            logger.warning("⚠️ No valid queries found in query file")
            return
        
        logger.info(f"🔎 Running {len(queries)} queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"🧪 Query {i}: {query}")
            print('='*80)
            
            try:
                results = self.vector_db.similarity_search_with_score(query, k=top_k)
                if not results:
                    print("❌ No results found")
                    continue
                
                for j, (doc, score) in enumerate(results, 1):
                    print(f"\n📄 Match {j} (Similarity: {score:.4f})")
                    print(f"📁 Source: {doc.metadata.get('source_file', 'Unknown')}")
                    print(f"📄 Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"\n📝 Content:\n{doc.page_content}")
                    print(f"\n{'-'*60}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing query '{query}': {str(e)}")

def main():
    config = {
        "pdf_dir": "J:/Assignment/Aiplanet/testingknowledgebase/pdfs",
        "save_path": "enhanced_math_vector_db",
        "query_file": "J:/Assignment/Aiplanet/testingknowledgebase/queries.txt"
    }
    
    kb = MathKnowledgeBase(**config)
    kb.build_vector_database(force_rebuild=True)
    
    if kb.vector_db:
        kb.run_queries(top_k=3)
    else:
        logger.error("❌ Failed to create/load vector database")

if __name__ == "__main__":
    main()
