# RAG Assignment

MindfulRAG is an AI-assisted mental health companion that combines a Retrieval-Augmented Generation (RAG) pipeline with multimodal inputs to provide empathetic, evidence-aligned guidance. It never replaces professional care and only surfaces supportive suggestions grounded in trusted references.

> **Disclaimer:** This prototype does not offer clinical advice. If you or someone you know is in crisis, please reach out to a licensed professional or emergency services immediately.

## Problem Statement
- **Clear problem definition:** Offer a safe conversational space where users can express their feelings and receive supportive, literature-backed coping suggestions. The assistant must ground its answers in curated mental health resources, remain multilingual, and accept text, voice, and optional image cues while respecting the limits of non-clinical support.

## Dataset / Knowledge Source
- **Type of data (PDF, TXT, DOCX, Web, etc.):** Curated PDF manuals, workbooks, and psychoeducational guides focused on anxiety, depression, stress management, and mindfulness.
- **Data source (public / self-created):** Publicly available resources placed manually inside the `data/` directory; the repository does not redistribute copyrighted material.

## RAG Architecture
- **Block diagram of complete RAG pipeline:**

```mermaid
flowchart LR
	U(User Query / Voice / Image) --> P["Preprocess Input\nspeech-to-text (optional)\nsynonym expansion"]
	P --> R["Hybrid Retrieval\nsemantic (Chroma)\nkeyword filtering"]
	R --> C[Compose Context Window]
	C --> L[Gemini 2.5 Flash via LangChain Prompt]
	L --> A["Empathetic Response\nsource attributions"]
```
- **Core components:** `ingest.py` performs document loading, chunking, embedding, and persistence; `backend_multimodal.py` loads the vector store, orchestrates retrieval, and builds the prompt; `streamlit_app_multilingual.py` handles multilingual chat, voice, and optional image inputs.

## Text Chunking Strategy
- **Chunk size:** 1,500 characters.
- **Chunk overlap:** 100 characters.
- **Reason for chosen strategy:** Balances contextual continuity for therapeutic techniques (often explained across paragraphs) with manageable embedding size, reducing truncation while remaining fast on CPU-bound environments.

## Embedding Details
- **Embedding model used:** Sentence-transformers `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`.
- **Reason for selecting the model:** Light-weight, CPU-friendly, strong performance on semantic similarity, and readily available under a permissive license suitable for local execution.

## Vector Database
- **Vector store used (FAISS / Chroma / etc.):** Persistent `Chroma` store at `chroma_db/`.
- **Metadata captured:** Source file path, page number, and chunk index enabling citation in chat responses and future filtering.

## Notebook Implementation
- **Step-wise code (data loading → retrieval → generation):**

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load and split documents
loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 2. Embed and persist
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
vectorstore.persist()

# 3. Retrieve and generate
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
context_docs = retriever.get_relevant_documents("How do I calm anxiety before sleep?")
context = "\n\n".join(doc.page_content for doc in context_docs)

prompt = """You are a compassionate mental health support assistant.
Context:\n{context}\n\nUser question: {question}\n""".format(context=context, question="How do I calm anxiety before sleep?")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
response = llm.invoke(prompt)
print(response.content)
```
- **Proper comments and markdown cells:** Annotate notebook cells with short MD sections (Overview, Ingestion, Retrieval, Generation) and inline comments explaining non-trivial steps such as the hybrid retrieval heuristic.

## Future Improvements
- **Better chunking:** Experiment with semantic-aware or adaptive chunking to preserve exercises, tables, or checklists intact.
- **Reranking / hybrid search:** Integrate rerankers such as `bge-reranker-large` for tighter relevance ordering after initial recall.
- **Metadata filtering:** Use tags (topic, modality, difficulty level) during ingestion to filter retrieval for user intent.
- **UI integration:** Extend the current Streamlit front-end with progress tracking, journaling, and hand-off pathways to professional services.

## Technical Specifications Summary

| Feature | Specification | Reason |
| :--- | :--- | :--- |
| **Chunking** | Size: 1500, Overlap: 100 | Preserves context for therapeutic techniques; effective for CPU processing. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Fast, lightweight, efficient for semantic search on local CPU. |
| **Vector Database** | Chroma | Open-source, persistent, easy integration with LangChain. |
| **Framework** | LangChain | Simplifies RAG pipeline construction and LLM integration. |
| **LLM** | Gemini 2.5 Flash | High speed, cost-effective, and capable of empathetic nuance. |

## Detailed Test Query Outputs

Below represent simulated outputs demonstrating the system's "Empathy + Evidence" approach.

### Test Case 1: Work Stress
**User Query:**
> "I feel overwhelmed at work, how can I cope?"

**Model Response:**
> "I hear that you're feeling weighed down by work right now, and I want you to know it's valid to feel this way. 🌿
>
> Based on the *Stress Management Workbook (Page 12)* and *Mindfulness Guide (Page 4)*, here are a few gentle steps that might help:
> 1.  **Chunks of Time:** valid technique is to break your tasks into small, 20-minute intervals. Focus on just one thing at a time.
> 2.  **Box Breathing:** Pause for a moment. Inhale for 4 counts, hold for 4, exhale for 4, and hold for 4. This signals safety to your nervous system.
> 3.  **Compassionate Boundaries:** It is okay to say 'I need a moment' or to ask for help with prioritization.
>
> Remember, your well-being comes before productivity. Would you like to try a quick breathing exercise together?"

---

### Test Case 2: Anxiety Grounding
**User Query:**
> "Any gentle grounding technique before exams?"

**Model Response:**
> "It is completely understandable to feel those pre-exam nerves. You have prepared, but your body is reacting to the pressure. Let's try to anchor ourselves in the present moment. ⚓
>
> The *Anxiety Relief Manual (Page 22)* suggests the **5-4-3-2-1 Technique**:
> *   **5** things you see (e.g., a pen, the clock).
> *   **4** things you can touch (e.g., the texture of your paper, your feet on the floor).
> *   **3** things you hear (e.g., distant traffic, your breath).
> *   **2** things you can smell.
> *   **1** thing you can taste.
>
> This exercise shifts your focus from 'what if' thoughts to 'what is' right now. You've got this. Take a deep breath."

---

### Test Case 3: Supporting Others
**User Query:**
> "How to support a friend showing signs of burnout?"

**Model Response:**
> "It is incredibly kind of you to look out for your friend. Supporting someone with burnout requires patience and care. 🤝
>
> According to *Supporting Loved Ones (Page 8)*, here are some supportive approaches:
> *   **Listen without Fixing:** deeply listening is often more powerful than advice. Use phrases like, 'That sounds incredibly exhausting, I'm here for you.'
> *   **Validate their Experience:** Acknowledge their hard work and their feelings. Say, 'It makes sense that you are tired given how hard you've been running.'
> *   **Encourage Professional Support:** Gently suggest that speaking to a counselor isn't a sign of weakness, but a tool for recovery.
>
> Ensure you are also protecting your own energy while being there for them. Would you like suggestions on how to start that conversation?"

## README / Report
- **Project overview:** MindfulRAG combines curated psychoeducational PDFs with Gemini 2.5 Flash to deliver multilingual, empathetic conversations under a strict non-clinical policy.
- **Tools & libraries used:** LangChain, Chroma, HuggingFace Sentence Transformers, Google Generative AI, Streamlit, SpeechRecognition, Pillow.
- **Instructions to run the notebook:**
	1. Create and activate a virtual environment, then install `requirements.txt`.
	2. Launch Jupyter (`pip install jupyter` if needed) and open a new notebook in the repository root.
	3. Copy the code blocks from the “Notebook Implementation” section, execute sequentially, and ensure a valid `GOOGLE_API_KEY` is loaded via `.env` or notebook environment variables.
	4. Replace the sample query with your own and inspect both the retrieved context snippets and generated response cells.


- **Streamlit:** Launch the conversational UI with `streamlit run streamlit_app_multilingual.py`. The app supports text and  optional voice capture (SpeechRecognition).
