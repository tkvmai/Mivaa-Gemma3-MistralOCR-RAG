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
import time

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

    # API connection status notifications (show with Dismiss button)
    if "api_notification_dismissed" not in st.session_state:
        st.session_state.api_notification_dismissed = False

    # --- Optimization: Cache Mistral client in session state ---
    if "mistral_client" not in st.session_state or "mistral_status" not in st.session_state:
        mistral_client = None
        mistral_status = False
        if api_key:
            mistral_client = initialize_mistral_client(api_key)
            mistral_status = mistral_client is not None
        st.session_state.mistral_client = mistral_client
        st.session_state.mistral_status = mistral_status
    else:
        mistral_client = st.session_state.mistral_client
        mistral_status = st.session_state.mistral_status

    # --- Optimization: Only check Google API if key changes, cache result ---
    if "google_api_key_checked" not in st.session_state or st.session_state.get("last_google_api_key") != google_api_key:
        google_status = False
        google_api_checked = False
        google_api_message = ""
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            google_status = is_valid
            google_api_checked = True
            google_api_message = message
        st.session_state.google_status = google_status
        st.session_state.google_api_checked = google_api_checked
        st.session_state.google_api_message = google_api_message
        st.session_state.last_google_api_key = google_api_key
    else:
        google_status = st.session_state.google_status
        google_api_message = st.session_state.google_api_message

    notification_msgs = []
    if mistral_status:
        notification_msgs.append(("success", "✅ Mistral API connected successfully"))
    else:
        notification_msgs.append(("error", "❌ Mistral API key not found or invalid."))
    if google_status:
        notification_msgs.append(("success", f"✅ Google API {google_api_message or 'connected successfully'}"))
    else:
        notification_msgs.append(("error", f"❌ Google API: {google_api_message or 'not connected'}"))
    if not mistral_status:
        notification_msgs.append(("warning", "⚠️ Valid Mistral API key required for document processing"))
    if not google_status:
        notification_msgs.append(("warning", "⚠️ Google API key required for chat functionality"))

    if not st.session_state.api_notification_dismissed:
        for level, msg in notification_msgs:
            if level == "success":
                st.success(msg)
            elif level == "error":
                st.error(msg)
            elif level == "warning":
                st.warning(msg)
        if st.button("Dismiss", key="dismiss_api_notification"):
            st.session_state.api_notification_dismissed = True
            st.rerun()

    # --- Optimization: Cache document list in session state ---
    def fetch_docs():
        session = SessionLocal()
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        session.close()
        return docs
    if "docs" not in st.session_state or st.session_state.get("docs_dirty", True):
        st.session_state.docs = fetch_docs()
        st.session_state.docs_dirty = False
    docs = st.session_state.docs

    # UI state for selected documents and menu
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = [doc.id for doc in docs] if docs else []
    if "rename_doc_id" not in st.session_state:
        st.session_state.rename_doc_id = None
    if "rename_value" not in st.session_state:
        st.session_state.rename_value = ""
    if "show_upload" not in st.session_state:
        st.session_state.show_upload = False

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Sources")
        # Add and Discover buttons
        col_add, col_discover = st.columns([1, 1])
        with col_add:
            if st.button("+ Add", key="add_btn"):
                st.session_state.show_upload = not st.session_state.show_upload
        with col_discover:
            st.button("Discover", key="discover_btn", disabled=True)
        # Upload section
        if st.session_state.show_upload:
            st.info("Upload PDFs or Images, or add a URL.")
            upload_tab = st.radio("Upload Type", ["PDF", "Image", "URL"], horizontal=True)
            document_sources = []
            if upload_tab == "PDF":
                uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, key="multi_pdf")
                if uploaded_files and st.button("Process PDFs", key="process_pdfs_btn"):
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
            elif upload_tab == "Image":
                uploaded_images = st.file_uploader("Choose Image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="multi_img")
                if uploaded_images and st.button("Process Images", key="process_imgs_btn"):
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
            elif upload_tab == "URL":
                url = st.text_input("Document URL:", key="url_input")
                if url and st.button("Load Document from URL", key="process_url_btn"):
                    document_sources.append({
                        "type": "document_url",
                        "document_url": url
                    })
            # Process all document sources
            if document_sources:
                session = SessionLocal()
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
                st.session_state.show_upload = False
                st.session_state.docs_dirty = True
                st.rerun()
        # Multiselect for document selection inside a form
        doc_options = []
        doc_id_to_label = {}
        for doc in docs:
            icon = "📄" if (doc.filename and doc.filename.lower().endswith(".pdf")) else ("🖼️" if doc.filetype == "image_url" else "🌐")
            label = f"{icon} {doc.filename or doc.url or f'Document {doc.id}'}"
            doc_options.append(label)
            doc_id_to_label[doc.id] = label
        label_to_doc_id = {v: k for k, v in doc_id_to_label.items()}
        valid_selected_doc_ids = [doc_id for doc_id in st.session_state.selected_doc_ids if doc_id in doc_id_to_label]
        with st.form("doc_select_form"):
            selected_labels = st.multiselect(
                "Select sources:",
                options=doc_options,
                default=[doc_id_to_label[doc_id] for doc_id in valid_selected_doc_ids],
                key="doc_multiselect"
            )
            confirm = st.form_submit_button("Confirm Selection")
        if confirm:
            st.session_state.selected_doc_ids = [label_to_doc_id[label] for label in selected_labels]
        # 3-dot menu for rename/delete
        for doc in docs:
            if st.session_state.rename_doc_id == doc.id:
                new_name = st.text_input("Rename file", value=st.session_state.rename_value, key=f"rename_input_{doc.id}")
                col_rename, col_delete = st.columns(2)
                with col_rename:
                    if st.button("Rename", key=f"rename_btn_{doc.id}"):
                        session = SessionLocal()
                        doc_to_rename = session.query(Document).filter_by(id=doc.id).first()
                        doc_to_rename.filename = new_name
                        session.commit()
                        session.close()
                        st.session_state.rename_doc_id = None
                        st.session_state.docs_dirty = True
                        st.rerun()
                with col_delete:
                    if st.button("Delete", key=f"delete_btn_{doc.id}"):
                        session = SessionLocal()
                        session.query(Document).filter_by(id=doc.id).delete()
                        session.commit()
                        session.close()
                        if doc.id in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.remove(doc.id)
                        st.session_state.rename_doc_id = None
                        st.session_state.docs_dirty = True
                        st.rerun()
            else:
                # Show 3-dot menu button next to each doc
                menu_key = f"menu_{doc.id}"
                if st.button("⋮", key=menu_key):
                    st.session_state.rename_doc_id = doc.id if st.session_state.rename_doc_id != doc.id else None
                    st.session_state.rename_value = doc.filename or ""

    with right_col:
        selected_docs = [d for d in docs if d.id in st.session_state.selected_doc_ids]
        if not selected_docs:
            st.info("Select one or more sources from the left pane to start chatting.")
        else:
            st.subheader(f"Chat with {len(selected_docs)} source(s)")
            # Summary of selected sources
            st.markdown("**Selected sources:** " + ", ".join([d.filename or d.url or f"Document {d.id}" for d in selected_docs]))
            # Placeholder action buttons
            col_note, col_add_note, col_audio, col_mindmap = st.columns([1,1,1,1])
            with col_note:
                st.button("💾 Save to note", key="save_note_btn", disabled=True)
            with col_add_note:
                st.button("📝 Add note", key="add_note_btn", disabled=True)
            with col_audio:
                st.button("🔊 Audio Overview", key="audio_btn", disabled=True)
            with col_mindmap:
                st.button("🗺️ Mind Map", key="mindmap_btn", disabled=True)
            with st.expander("Document Content", expanded=False):
                for d in selected_docs:
                    st.markdown(f"**{d.filename or d.url or f'Document {d.id}'}**\n\n" + d.ocr_text)
            # Chat interface
            if "messages" not in st.session_state or st.session_state.get("last_doc_ids") != set(st.session_state.selected_doc_ids):
                st.session_state.messages = []
                st.session_state.last_doc_ids = set(st.session_state.selected_doc_ids)
            st.subheader("Chat")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Ask a question about your selected sources..."):
                if not google_api_key:
                    st.error("Google API key is required for generating responses.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Combine all selected docs' OCR text
                            context = "\n\n".join([d.ocr_text for d in selected_docs])
                            response = generate_response(context, prompt)
                            st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()