# backend_multimodal.py
"""
Backend logic for MindfulRAG: model loading, search, and preprocessing functions
for the mental health support assistant.
"""
import asyncio
import base64
import io
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

load_dotenv()


def ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Return an active event loop for the current thread."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


if os.name == "nt":  # Streamlit runs on a background thread on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_models():
    ensure_event_loop()
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Chroma database not found at '{VECTOR_STORE_PATH}'. Please run 'python ingest.py' first.")
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt_template = """
You are a compassionate mental health support assistant. You provide empathetic, non-judgmental emotional support grounded in evidence-based techniques from the provided reference material.

Important guidelines:
- You are NOT a therapist or medical professional. Never diagnose or prescribe.
- Use warm, simple, human language.
- Validate feelings without reinforcing hopelessness or negativity.
- Use the provided context from mental health literature to inform your responses.
- Reference previous parts of the conversation when relevant.
- Offer optional coping techniques (breathing, grounding, reflection) when appropriate.
- If distress sounds severe, gently suggest seeking professional human support.
- Keep your responses supportive, calm, and respectful.

Previous conversation:
{conversation_history}

Reference material:
{context}

User's message: {question}

Your Response:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "context", "question"])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    return vectorstore, chain

def hybrid_search(vectorstore, query, k=5):
    semantic_docs = vectorstore.similarity_search_with_score(query, k=k)
    all_docs = vectorstore.get()
    keyword_matches = []
    query_words = set(query.lower().split())
    if 'documents' in all_docs and 'metadatas' in all_docs:
        for i, (doc_text, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                from langchain.schema import Document
                doc = Document(page_content=doc_text, metadata=metadata)
                score = 1.0 / (overlap + 1)
                keyword_matches.append((doc, score))
    all_results = semantic_docs + keyword_matches
    seen_content = set()
    unique_results = []
    for doc, score in all_results:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    return sorted(unique_results, key=lambda x: x[1])[:k]

def preprocess_query(query):
    processed_query = query.lower().strip()
    mental_health_synonyms = {
        "anxiety": ["anxious", "worry", "nervous", "panic", "fear"],
        "depression": ["depressed", "sad", "hopeless", "low mood", "despair"],
        "stress": ["stressed", "overwhelmed", "burnout", "pressure", "tension"],
        "therapy": ["counseling", "treatment", "psychotherapy", "intervention"],
        "coping": ["managing", "handling", "dealing", "strategies", "techniques"],
        "trauma": ["ptsd", "traumatic", "distress", "adverse experience"],
        "sleep": ["insomnia", "sleepless", "rest", "fatigue"],
        "mindfulness": ["meditation", "breathing", "grounding", "relaxation"],
        "self-care": ["wellness", "wellbeing", "self-help", "resilience"],
        "grief": ["loss", "bereavement", "mourning", "sorrow"]
    }
    query_terms = processed_query.split()
    expanded_terms = []
    for term in query_terms:
        expanded_terms.append(term)
        for key, synonyms in mental_health_synonyms.items():
            if term in key or key in term:
                expanded_terms.extend(synonyms)
    expanded_query = " ".join(expanded_terms)
    return query, expanded_query

def process_image_input(uploaded_image):
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            return {
                "image": image,
                "base64": img_base64,
                "description": "User uploaded an image related to their question"
            }
        except Exception as e:
            return None
    return None
