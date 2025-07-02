# MathMentorAI: Agentic RAG Math Tutor

**MathMentorAI** is an agentic, retrieval-augmented chatbot designed to behave like a math professor. Built using [LangGraph](https://github.com/langchain-ai/langgraph), FAISS, and Gemini Pro (via Google Generative AI), it intelligently classifies, retrieves, explains, and refines mathematical queries — complete with web search fallback and human-in-the-loop feedback.

---

## 🚀 Features

- 🔎 **Math Query Classification** (LLM + Embedding-based)
- 📚 **In-Knowledge-Base Retrieval** using FAISS
- 🌐 **Web Search Fallback** via SerpAPI
- 🧠 **Step-by-Step Math Solution Generation**
- 🧹 **Chunk Cleaning Agent** for OCR & structure correction
- 🗣️ **Human-in-the-Loop Feedback Capture**
- 🎛️ Streamlit UI for chatting and feedback

---

## 🧱 Architecture

### Powered by LangGraph's agent framework:

![workflow](workflow.png)

## **🧰 Technologies Used**

* **🧠 LangGraph:** Multi-agent workflow
* **🌐 LangChain**
* **🔎 FAISS:** Semantic retrieval
* **🤖 Gemini:** Google Generative AI
* **📄 PyMuPDF:** For better PDF parsing
* **🔍 SerpAPI:** Live Google search results
* **🧼 Custom cleaning agent:** For OCR/noise removal
* **🌐 Streamlit:** For the web frontend

**📘 Knowledge Base**

The knowledge base is built using math textbooks from CBSE Classes 10, 11, and 12, covering:

* Algebra
* Geometry
* Trigonometry
* Calculus
* Probability
* Coordinate Geometry
* and more...

The textbooks were processed using:

* ✅ Custom PDF cleaning
* ✅ Text chunking with overlap for mathematical continuity
* ✅ Embedding and indexing using SentenceTransformers + FAISS

**💬 How it Works**

1.  User enters a query via the Streamlit chat interface.
2.  Query is passed to LangGraph for node-based processing.
3.  `classify_math_query`: Determines if the input is a math query.
4.  `handle_math_query`: Checks if the KB has relevant content.
5.  Depending on coverage:
    * KB → `Retrievalagent` → `SolutionGenerator`
    * No KB → `webRetrieval` → `SolutionGenerator`
6.  `FeedbackCollector`: Allows user to submit human feedback.

## 📸 UI Preview

![workflow](sampleoutput.png)





