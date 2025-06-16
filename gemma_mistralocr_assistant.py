import streamlit as st
import base64
import tempfile
import os
from mistralai import Mistral
from PIL import Image
import io
from mistralai import DocumentURLChunk, ImageURLChunk
from mistralai.models import OCRResponse
from dotenv import find_dotenv, load_dotenv
import google.generativeai as genai
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Load environment variables from .env if present
load_dotenv(find_dotenv())

# Read API keys from environment
api_key = os.environ.get("MISTRAL_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

# Database setup
DATABASE_URL = "sqlite:///ocr_documents.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    filetype = Column(String)
    content = Column(LargeBinary)  # For files
    url = Column(String)           # For URLs
    ocr_text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_mistral_client(api_key):
    try:
        return Mistral(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Mistral client: {e}")
        return None

def test_google_api(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemma-3-27b-it')
        # Simple test prompt
        _ = model.generate_content("Hello, how are you?")
        return True, "connected successfully"
    except Exception as e:
        return False, str(e)

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# OCR Processing Functions
def upload_pdf(client, content, filename):
    """
    Uploads a PDF to Mistral's API and retrieves a signed URL for processing.

    Parameters:
    - client: Initialized Mistral API client.
    - content: Binary content of the uploaded file.
    - filename: Name of the file being uploaded.

    Returns:
    - signed_url (str): A signed URL from Mistral for OCR processing.
    """
    if client is None:
        raise ValueError("Mistral client is not initialized")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "wb") as tmp:
            tmp.write(content)

        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )

            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url

        except Exception as e:
            raise ValueError(f"Error uploading PDF: {str(e)}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded image strings.

    Parameters:
    - markdown_str (str): The markdown content containing image placeholders.
    - images_dict (dict): Dictionary where keys are image names and values are base64-encoded image strings.

    Returns:
    - str: Markdown with embedded base64 images.
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine markdown content from all pages of the OCR response,
    embedding base64 images in place of image placeholders.

    Parameters:
    - ocr_response (OCRResponse): The response object returned by Mistral OCR.

    Returns:
    - str: Complete markdown string for all pages with inline images.
    """
    markdowns: list[str] = []

    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64

        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)


def process_ocr(client, document_source):
    """
    Process a document or image using Mistral's OCR API based on the source type.

    Parameters:
    - client: Initialized Mistral API client.
    - document_source (dict): Should contain 'type' and corresponding URL key:
        - type: "document_url" or "image_url"
        - document_url/image_url: The signed URL

    Returns:
    - OCRResponse: Structured OCR response from Mistral
    """
    if client is None:
        raise ValueError("Mistral client is not initialized")

    if document_source["type"] == "document_url":
        return client.ocr.process(
            document=DocumentURLChunk(document_url=document_source["document_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

    elif document_source["type"] == "image_url":
        return client.ocr.process(
            document=ImageURLChunk(image_url=document_source["image_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

    else:
        raise ValueError(f"Unsupported document source type: {document_source['type']}")

def generate_response(context, query):
    """
    Generate a response using Google's Gemini API (Gemma family).

    Parameters:
    - context (str): The OCR-extracted document content.
    - query (str): User's question about the document.

    Returns:
    - str: AI-generated response based on document content.
    """
    try:
        # Initialize the Google Gemini API
        genai.configure(api_key=google_api_key)

        # Validate context
        if not context or len(context) < 10:
            return "Error: No document content available to answer your question."

        # Create a detailed prompt
        prompt = f"""I have a document with the following content:

{context}

Based on this document, please answer the following question:
{query}

If you can find information related to the query in the document, please answer based on that information.
If the document doesn't specifically mention the exact information asked, please try to infer from related content or clearly state that the specific information isn't available in the document.
"""

        # Debug logs (optional)
        print(f"Sending prompt with {len(context)} characters of context")
        print(f"First 500 chars of context: {context[:500]}...")

        # Initialize the model
        model = genai.GenerativeModel('gemma-3-27b-it')

        # Generation config
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        # Safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        # Generate and return the response
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return response.text

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error generating response: {str(e)}"


def main():
    global api_key, google_api_key

    st.set_page_config(page_title="Document OCR & Chat", layout="wide")

    # Sidebar: Show settings and API key status only
    with st.sidebar:
        st.header("Settings")

        # Initialize Mistral client with the API key
        mistral_client = None
        if api_key:
            mistral_client = initialize_mistral_client(api_key)
            if mistral_client:
                st.success("✅ Mistral API connected successfully")
        else:
            st.error("❌ Mistral API key not found in environment variables.")

        # Google API key validation
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            if is_valid:
                st.success(f"✅ Google API {message}")
            else:
                st.error(f"❌ Google API: {message}")
                google_api_key = None
        else:
            st.error("❌ Google API key not found in environment variables.")

        # Display warnings for missing API keys
        if not api_key or mistral_client is None:
            st.warning("⚠️ Valid Mistral API key required for document processing")

        if not google_api_key:
            st.warning("⚠️ Google API key required for chat functionality")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_content" not in st.session_state:
        st.session_state.document_content = ""

    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False

    # Document upload section
    st.subheader("Document Upload")

    session = SessionLocal()

    # Only show document upload if Mistral client is initialized
    if mistral_client:
        input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"])

        document_sources = []
        uploaded_files = []
        uploaded_images = []
        urls = []

        if input_method == "URL":
            url = st.text_input("Document URL:")
            if url and st.button("Load Document from URL"):
                document_sources.append({
                    "type": "document_url",
                    "document_url": url
                })
                urls.append(url)

        elif input_method == "PDF Upload":
            uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
            if uploaded_files and st.button("Process PDFs"):
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read()
                    try:
                        doc_url = upload_pdf(mistral_client, content, uploaded_file.name)
                        document_sources.append({
                            "type": "document_url",
                            "document_url": doc_url,
                            "filename": uploaded_file.name,
                            "content": content
                        })
                    except Exception as e:
                        st.error(f"Error uploading {uploaded_file.name}: {str(e)}")

        elif input_method == "Image Upload":
            uploaded_images = st.file_uploader("Choose Image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if uploaded_images and st.button("Process Images"):
                for uploaded_image in uploaded_images:
                    try:
                        image = Image.open(uploaded_image)
                        st.image(image, caption=f"Uploaded Image: {uploaded_image.name}", use_column_width=True)
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        document_sources.append({
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}",
                            "filename": uploaded_image.name,
                            "content": buffered.getvalue()
                        })
                    except Exception as e:
                        st.error(f"Error processing image {uploaded_image.name}: {str(e)}")

        # Process all document sources
        if document_sources:
            for doc in document_sources:
                with st.spinner(f"Processing {doc.get('filename', doc.get('document_url', 'URL'))}..."):
                    try:
                        ocr_response = process_ocr(mistral_client, doc)
                        if ocr_response and ocr_response.pages:
                            raw_content = []
                            for page in ocr_response.pages:
                                page_content = page.markdown.strip()
                                if page_content:
                                    raw_content.append(page_content)
                            final_content = "\n\n".join(raw_content)
                            # Save to database
                            db_doc = Document(
                                filename=doc.get('filename', None),
                                filetype=doc['type'],
                                content=doc.get('content', None),
                                url=doc.get('document_url', None),
                                ocr_text=final_content
                            )
                            session.add(db_doc)
                            session.commit()
                            st.success(f"Processed and saved: {doc.get('filename', doc.get('document_url', 'URL'))}")
                        else:
                            st.warning(f"No content extracted from {doc.get('filename', doc.get('document_url', 'URL'))}.")
                    except Exception as e:
                        st.error(f"Processing error for {doc.get('filename', doc.get('document_url', 'URL'))}: {str(e)}")
    session.close()

    # Main area: Display chat interface
    st.title("Document OCR & Chat")

    # Document preview area
    if "document_loaded" in st.session_state and st.session_state.document_loaded:
        with st.expander("Document Content", expanded=False):
            # Show the display version with page numbers
            if "display_content" in st.session_state:
                st.markdown(st.session_state.display_content)
            else:
                st.markdown(st.session_state.document_content)

        # Chat interface
        st.subheader("Chat with your document")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for user query
        if prompt := st.chat_input("Ask a question about your document..."):
            # Check if Google API key is available
            if not google_api_key:
                st.error("Google API key is required for generating responses. Please add it in the sidebar settings.")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Show thinking spinner
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Get document content from session state
                        document_content = st.session_state.document_content

                        # Generate response directly
                        response = generate_response(document_content, prompt)

                        # Display response
                        st.markdown(response)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show a welcome message if no document is loaded
        st.info("👈 Please upload a document using the sidebar to start chatting.")


if __name__ == "__main__":
    main()