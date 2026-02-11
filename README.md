# MindfulRAG: Mental Health Support Assistant using RAG

MindfulRAG is an AI-powered mental health support assistant that provides empathetic, evidence-informed responses grounded in mental health literature. It uses a Retrieval-Augmented Generation (RAG) pipeline to ensure responses are informed by trusted reference material, while maintaining a warm, non-judgmental tone.

> **Disclaimer:** This tool is NOT a substitute for professional mental health care. If you are in crisis, please contact a helpline or a mental health professional.

## Features

- **Empathetic conversations** — warm, supportive, non-clinical emotional support
- **RAG-powered** — responses grounded in ingested mental health literature
- **Multimodal input** — text, voice, and image support
- **Multilingual** — supports 15+ languages including major Indian languages
- **Conversation memory** — maintains context across the chat session

## How to Run This Project

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Set Up Your Project Folder

Your folder structure should look like this:

```
MindfulRAG/
├── data/
│   └── (Your mental health resource PDFs go here)
├── backend_multimodal.py
├── streamlit_app_multilingual.py
├── ingest.py
├── requirements.txt
└── README.md
```

### Step 2: Create a Virtual Environment

```bash
cd MindfulRAG
python -m venv venv

# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your API Key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### Step 5: Process Your Documents

Add your mental health resource PDFs into the `data/` folder, then run:

```bash
python ingest.py
```

### Step 6: Run the App

```bash
streamlit run streamlit_app_multilingual.py
```

Your browser will open with the MindfulRAG assistant ready to chat.