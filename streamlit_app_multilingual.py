import os
import streamlit as st
import json
from backend_multimodal import load_models, hybrid_search, preprocess_query, process_image_input

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MindfulRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* ===== Dark-mode calming palette ===== */
:root {
    --primary: #7EB8E0;
    --primary-soft: rgba(126,184,224,0.12);
    --accent: #8ECFA8;
    --accent-soft: rgba(142,207,168,0.12);
    --text-primary: #E2E8F0;
    --text-secondary: #A0AEC0;
    --surface: #1E2A3A;
    --surface-raised: #253347;
    --surface-hover: #2D3E54;
    --border: rgba(255,255,255,0.08);
    --border-focus: rgba(126,184,224,0.4);
}

/* Main app background */
.stApp {
    background: linear-gradient(160deg, #0F1923 0%, #152231 40%, #131F2B 100%) !important;
}

/* ===== Header / Hero ===== */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
    margin-bottom: 1rem;
}
.hero-icon {
    font-size: 3.5rem;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ===== Chat messages ===== */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem 1.2rem !important;
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: var(--text-primary) !important;
}

/* ===== Chat input ===== */
[data-testid="stChatInput"] {
    background: transparent !important;
    border-color: transparent !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 2px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1.2rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-soft) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-secondary) !important;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #152231 0%, #172535 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* Sidebar card-like sections */
.sidebar-card {
    background: var(--surface-raised);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    color: var(--text-primary);
}
.sidebar-card b {
    color: var(--text-primary);
}

/* ===== Buttons ===== */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: var(--surface-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--surface-hover) !important;
    border-color: var(--primary) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(126,184,224,0.15) !important;
}

/* ===== Quick-action chips ===== */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0;
}
.chip {
    display: inline-block;
    background: var(--surface-raised);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.45rem 1rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    cursor: default;
    transition: all 0.2s;
}
.chip:hover {
    background: var(--surface-hover);
    border-color: var(--primary);
    color: var(--text-primary);
}

/* ===== Breathing animation (sidebar) ===== */
@keyframes breathe {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50% { transform: scale(1.15); opacity: 1; }
}
.breathe-circle {
    width: 60px; height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    margin: 0 auto;
    animation: breathe 6s ease-in-out infinite;
}
.breathe-label {
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

/* ===== Expander ===== */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

/* ===== Crisis banner ===== */
.crisis-banner {
    background: rgba(229, 62, 62, 0.1);
    border-left: 4px solid #E53E3E;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: #FC8181;
}
.crisis-banner b {
    color: #FEB2B2;
}

/* Hide default Streamlit header & footer for cleaner look */
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }

/* Hide textarea scrollbar and resize handle */
[data-testid="stChatInput"] textarea {
    overflow: hidden !important;
    resize: none !important;
}

/* Metric styling */
[data-testid="stMetric"] {
    background: var(--surface-raised);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--border);
    color: var(--text-primary);
}
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)





# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""
if "user_language" not in st.session_state:
    st.session_state.user_language = "en"


vectorstore = None
chain = None
backend_init_error = None

try:
    vectorstore, chain = load_models()
except Exception as e:
    backend_init_error = str(e)
    st.warning(
        "⚠️ Backend is not ready yet. Add a valid GOOGLE_API_KEY and make sure the vector DB is available."
    )





def process_voice_input():
    """
    Process voice input using speech recognition.
    Returns the transcribed text or None if failed.
    """
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        try:
            microphone_names = sr.Microphone.list_microphone_names()
        except OSError as device_error:
            st.error(f"🎤 Unable to access microphones: {device_error}")
            return None

        if "voice_device_index" not in st.session_state:
            st.session_state.voice_device_index = None

        def is_input_device(label: str) -> bool:
            lowered = label.lower()
            return any(keyword in lowered for keyword in ["mic", "input", "capture", "record"])

        indexed_devices = [(idx, name) for idx, name in enumerate(microphone_names)]
        input_devices = [(idx, name) for idx, name in indexed_devices if is_input_device(name)]
        device_pool = input_devices or indexed_devices

        device_choices = [("System default", None)]
        device_choices.extend([(f"{idx}: {name}", idx) for idx, name in device_pool])

        choice_labels = [label for label, _ in device_choices]

        default_choice = 0
        if st.session_state.voice_device_index is not None:
            for option_idx, (_, value) in enumerate(device_choices):
                if value == st.session_state.voice_device_index:
                    default_choice = option_idx
                    break
        else:
            for option_idx, (_, value) in enumerate(device_choices):
                if option_idx == 0:
                    continue
                selected_name = device_choices[option_idx][0].lower()
                if any(keyword in selected_name for keyword in ["microphone", "mic", "input"]):
                    default_choice = option_idx
                    break

        selected_label = st.selectbox(
            "Select a microphone",
            choice_labels,
            index=default_choice,
            key="voice_device_selector",
        )
        device_index = dict(device_choices)[selected_label]
        st.session_state.voice_device_index = device_index

        st.caption("Choose the input that matches your physical microphone. Speakers/headphones won't work for recording.")

        # Create audio recording interface
        st.info("🎤 Click the button below and speak your question...")

        if st.button("🎙️ Start Recording", key="voice_button"):
            with st.spinner("Listening... Speak now!"):
                try:
                    # Use microphone as source
                    mic_kwargs = {"device_index": device_index} if device_index is not None else {}
                    with sr.Microphone(**mic_kwargs) as source:
                        r.adjust_for_ambient_noise(source, duration=1)
                        audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    
                    with st.spinner("Processing speech..."):
                        # Use Google's speech recognition
                        text = r.recognize_google(audio)
                        
                        st.success(f"🎯 I heard: '{text}'")
                        
                        return text
                        
                except sr.WaitTimeoutError:
                    st.warning("⏰ No speech detected. Please try again.")
                    return None
                except sr.UnknownValueError:
                    st.warning("🤔 Sorry, I couldn't understand what you said. Please try again.")
                    return None
                except sr.RequestError as e:
                    st.error(f"❌ Speech recognition error: {e}")
                    return None
                except AttributeError as attr_error:
                    st.error("🎤 That device does not provide a microphone stream. Pick another input or reconfigure Windows recording devices.")
                    return None
                except OSError as mic_error:
                    st.error(f"🎤 Microphone error: {mic_error}")
                    return None
                except Exception as e:
                    st.warning(f"🎤 Voice input not available: {e}")
                    return None
    except ImportError:
        st.warning("🎤 Voice input requires additional packages. Please install: pip install speechrecognition pyaudio")
        return None
    
    return None



def create_multimodal_prompt(text_query, image_data=None, conversation_history=""):
    """
    Create a prompt for a mental health support chatbot.
    The assistant provides empathetic, non-clinical emotional support.
    """

    system_instructions = """
You are a mental health support chatbot.
Your role is to provide empathetic, non-judgmental emotional support.
You do NOT diagnose, provide medical advice, or replace professional help.

Guidelines:
- Use warm, simple, human language
- Validate feelings without reinforcing hopelessness or negativity
- Ask gentle, open-ended questions
- Offer optional coping techniques (breathing, grounding, reflection)
- Avoid absolutes, labels, or assumptions
- If distress sounds severe, gently suggest seeking human support

Never:
- Say you are a therapist or professional
- Give clinical diagnoses
- Encourage isolation or withdrawal
"""

    if image_data:
        return f"""
{system_instructions}

Previous conversation:
{conversation_history}

User message:
{text_query}

The user has also shared an image.
If the image contains emotional cues (e.g., written thoughts, drawings, symbols),
acknowledge them gently. If the image is unclear or irrelevant, focus on the text.

Your response should:
- Center on emotional understanding
- Be supportive, calm, and respectful
- Avoid interpretation beyond what the user expresses

Your response:
"""
    else:
        return f"""
{system_instructions}

Previous conversation:
{conversation_history}

User message:
{text_query}

Your response should:
- Acknowledge the user’s feelings
- Respond with empathy and clarity
- Ask at most one gentle follow-up question
- Avoid giving advice unless framed as an optional suggestion

Your response:
"""




# --- STREAMLIT APP ---
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🧠</div>
    <div class="hero-title">MindfulRAG</div>
    <div class="hero-subtitle">
        A safe, supportive space to talk things through.<br>
        You're not alone — share what's on your mind.
    </div>
</div>
""", unsafe_allow_html=True)

# Quick-start suggestion chips (only show when chat is empty)
if not st.session_state.messages:
    st.markdown("""
    <div class="chip-container" style="justify-content:center;">
        <span class="chip">😔 I'm feeling low today</span>
        <span class="chip">😰 Help me with anxiety</span>
        <span class="chip">🧘 Guide me through breathing</span>
        <span class="chip">💭 I just need someone to talk to</span>
        <span class="chip">😴 I can't sleep well</span>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.write(f"- **{source['source']}** (Page: {source['page']})")
        # Display images in user messages
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="Uploaded Image", width=300)

# --- INPUT SECTION ---

# Text input (always visible at bottom)
text_input = st.chat_input("How are you feeling today? 💬")

# Process the input
prompt = text_input
image_data = None

if prompt or image_data:
    if vectorstore is None or chain is None:
        st.error("Backend models are not loaded. Please check configuration and try again.")
        if backend_init_error:
            st.caption(f"Details: {backend_init_error}")
    else:
        # Prepare the user message
        user_message = {"role": "user", "content": prompt or "Please analyze this image"}
        if image_data:
            user_message["image"] = image_data["image"]
            user_message["has_image"] = True
        
        # Add user message to chat history
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            if prompt:
                st.markdown(prompt)
            if image_data:
                st.image(image_data["image"], caption="Uploaded Image", width=300)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("✨ Reflecting on what you shared..."):
                try:
                    # Use the prompt or default for image-only queries
                    query_text = prompt or "Please analyze this image and explain what you see"
                    
                    # Preprocess the query
                    original_query, expanded_query = preprocess_query(query_text)
                    
                    # 1. Try hybrid search first
                    try:
                        docs_with_scores = hybrid_search(vectorstore, original_query, k=5)
                    except Exception as e:
                        # Fallback to regular semantic search if hybrid search fails
                        docs_with_scores = vectorstore.similarity_search_with_score(original_query, k=5)
                    
                    # 2. If scores are too high (not similar enough), try with expanded query
                    if docs_with_scores and docs_with_scores[0][1] > 1.0:
                        try:
                            expanded_docs = hybrid_search(vectorstore, expanded_query, k=3)
                        except:
                            expanded_docs = vectorstore.similarity_search_with_score(expanded_query, k=3)
                        
                        # Combine and deduplicate results
                        all_docs = docs_with_scores + expanded_docs
                        seen_content = set()
                        unique_docs = []
                        for doc, score in all_docs:
                            content_hash = hash(doc.page_content[:100])
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                unique_docs.append((doc, score))
                        docs_with_scores = sorted(unique_docs, key=lambda x: x[1])[:5]
                    
                    # 3. Filter documents by similarity threshold
                    similarity_threshold = 1.2
                    relevant_docs = [doc for doc, score in docs_with_scores if score < similarity_threshold]
                    
                    # If no documents meet the threshold, use the top 3 anyway
                    if not relevant_docs:
                        relevant_docs = [doc for doc, score in docs_with_scores[:3]]
                    
                    # 4. Prepare context with more information
                    context_parts = []
                    for i, doc in enumerate(relevant_docs):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        context_parts.append(f"Source {i+1} ({source}, Page {page}):\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # 5. Prepare conversation history
                    conversation_history = ""
                    if len(st.session_state.messages) > 1:  # More than just the current message
                        recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges (6 messages)
                        history_parts = []
                        for msg in recent_messages[:-1]:  # Exclude current message
                            role = "User" if msg["role"] == "user" else "Assistant"
                            content = msg['content']
                            if msg.get('has_image'):
                                content += " [User also shared an image]"
                            history_parts.append(f"{role}: {content}")
                        conversation_history = "\n".join(history_parts)
                    
                    # 6. Generate response
                    result = chain.invoke(
                        {
                            "conversation_history": conversation_history,
                            "context": context,
                            "question": original_query,
                        }
                    )
                    
                    # Display the response
                    st.markdown(result)
                    
                    # Prepare sources information
                    sources = []
                    for doc in relevant_docs:
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        source_info = {
                            "source": source_file,
                            "page": doc.metadata.get('page', 'N/A')
                        }
                        if source_info not in sources:
                            sources.append(source_info)
                    
                    # Display sources
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.write(f"- **{source['source']}** (Page: {source['page']})")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result,
                        "sources": sources
                    })
                    
                    # Check if the answer indicates no relevant information was found
                    if "cannot find" in result.lower() or "no relevant information" in result.lower():
                        st.info("💡 **Tip**: Try rephrasing your question or use more specific terms!")

                except Exception as e:
                    error_msg = f"I'm sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- SIDEBAR ---
with st.sidebar:
    # Breathing exercise widget
    st.markdown("""
    <div class="sidebar-card" style="text-align:center;">
        <div style="font-size:0.9rem; font-weight:600; color:#E2E8F0; margin-bottom:0.6rem;">
            🫧 Take a moment to breathe
        </div>
        <div class="breathe-circle"></div>
        <div class="breathe-label">Breathe in… and out…</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🔄 Start a New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.rerun()
    
    # Language selector
    languages = {
        "en": "🇺🇸 English",
        "hi": "🇮🇳 हिंदी (Hindi)",
        "bn": "🇧🇩 বাংলা (Bengali)", 
        "te": "🇮🇳 తెలుగు (Telugu)",
        "mr": "🇮🇳 मराठी (Marathi)",
        "ta": "🇮🇳 தமிழ் (Tamil)",
        "ur": "🇵🇰 اردو (Urdu)",
        "gu": "🇮🇳 ગુજરાતી (Gujarati)",
        "kn": "🇮🇳 ಕನ್ನಡ (Kannada)",
        "ml": "🇮🇳 മലയാളം (Malayalam)",
        "es": "🇪🇸 Español",
        "fr": "🇫🇷 Français",
        "de": "🇩🇪 Deutsch",
        "zh": "🇨🇳 中文",
        "ar": "🇸🇦 العربية"
    }

    # Chat statistics
    if st.session_state.messages:
        st.markdown("---")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("💬 Messages", len(st.session_state.messages))
        with col_s2:
            st.metric("🙋 Your messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.markdown("---")
    
    # About section
    st.markdown("""
    <div class="sidebar-card">
        <div style="font-weight:600; margin-bottom:0.4rem; color:#E2E8F0;">🧠 About MindfulRAG</div>
        <div style="font-size:0.85rem; color:#A0AEC0; line-height:1.5;">
            A multilingual, multimodal mental health support assistant powered by RAG.
            Share your thoughts through text, voice, or images in your preferred language.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How to use
    st.markdown("""
    <div class="sidebar-card">
        <div style="font-weight:600; margin-bottom:0.4rem; color:#E2E8F0;">🎯 Ways to Connect</div>
        <div style="font-size:0.85rem; color:#A0AEC0; line-height:1.7;">
            💬 <b>Text</b> — Type in any language<br>
            🎤 <b>Voice</b> — Speak naturally<br>
            🖼️ <b>Image</b> — Share journal entries or drawings<br>
            🌍 <b>Multilingual</b> — 15+ languages supported
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Conversation starters
    st.markdown("""
    <div class="sidebar-card">
        <div style="font-weight:600; margin-bottom:0.4rem; color:#E2E8F0;">💡 Try Saying…</div>
        <div style="font-size:0.85rem; color:#A0AEC0; line-height:1.8;">
            • "I've been feeling really stressed lately"<br>
            • "Can you guide me through a breathing exercise?"<br>
            • "मुझे बहुत चिंता हो रही है"<br>
            • "আমি একটু কথা বলতে চাই"<br>
            • Share a journal photo and ask for reflection
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Crisis banner
    st.markdown("""
    <div class="crisis-banner">
        <b>🚨 In crisis?</b> This assistant is not a substitute for professional care.
        If you're in danger, please contact a <b>mental health helpline</b> or emergency services immediately.
    </div>
    """, unsafe_allow_html=True)
    
    # Debug
    with st.expander("🔧 Advanced", expanded=False):
        st.write(f"Messages: {len(st.session_state.messages)}")
        st.write(f"Context length: {len(st.session_state.conversation_context)}")
