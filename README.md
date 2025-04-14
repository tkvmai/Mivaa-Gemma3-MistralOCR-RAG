# AI-Powered Document Assistant for Subsurface Geoscience Data  
### OCR + RAG + Gemini (Gemma 3) → Intelligent Document QA for Well Logs, Drilling Reports, and Seismic Data

---

## Overview

This project brings together cutting-edge AI technologies to help upstream oil & gas professionals unlock insights from traditionally unstructured subsurface geoscience data like scanned well logs, drilling reports, and seismic survey documents.

Built for Geoscientists, Well Engineers, and Data Managers — this AI-powered Document Assistant uses:

- Mistral OCR → For extracting text & images from scanned documents  
- Retrieval-Augmented Generation (RAG) → For smart context retrieval from large documents  
- Google Gemini (Gemma 3) → For generating human-like, accurate answers to domain-specific queries  

---

## Why This Project?

Upstream oil & gas workflows deal with:

- Decades-old well logs  
- Legacy scanned reports  
- PDF-based drilling & seismic documents  
- Field notes & hand-drawn data  

Traditional document search fails in this domain because:
- The data is messy  
- The format is inconsistent  
- Manual review is slow & error-prone  

This AI Assistant solves that.

---

## Key Features

| Feature | Description |
|---------|-------------|
|OCR (Mistral)|Extracts text & embedded images from PDFs, images, or URLs.|
|RAG Framework|Breaks content into smart text chunks and retrieves context for queries.|
|Gemini LLM (Gemma 3)|Generates context-aware answers for technical queries like formation data, depth, hazards etc.|
|Streamlit Interface|Easy-to-use chat-based UI for interacting with your document.|

---

## Architecture

```plaintext
PDF / Image Upload
        ↓
Mistral OCR Extraction
        ↓
Data Chunking & RAG
        ↓
Gemini (Gemma 3) LLM
        ↓
Interactive Chat Interface
```

---

## Use Case Examples

| Use Case | Benefit |
|----------|---------|
|Extract Formation Depths|Quickly get depth & interval info from scanned well logs.|
|Summarize Drilling Reports|Extract incidents, hazards, and operational notes automatically.|
|Understand Seismic Reports|Ask about subsurface structures or survey highlights.|
|Field Data Integration|Transform handwritten notes into structured data.|

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2. Create & Activate Python Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root folder.

```
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

---

## Run the App

```bash
streamlit run gemma_ocr_assistant.py
```

---

## File Structure

```bash
├── gemma_ocr_assistant.py          # Main App Logic
├── requirements.txt                # Python Dependencies
├── .env                            # API Keys
├── utils/                          # OCR, RAG, Prompt Helpers
├── assets/                         # Sample PDFs, Images
└── README.md
```

---

## Future Enhancements

- Multi-document QA  
- Auto Entity Extraction (Formation Names, Depth, Fluid Type)  
- Summarization Mode  
- Excel/JSON Export of Extracted Data  
- Visualization Dashboard for Insights  

---

## Demo Video

https://youtu.be/quiAFph86O8

---

## Credits

Built by [Mivaa](https://deepdatawithmivaa.com)  
For the Subsurface Geoscience Community  
Empowering Data-Driven Upstream Workflows

---

## License

MIT License

---