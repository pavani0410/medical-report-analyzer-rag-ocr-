import os
import uuid
from functools import lru_cache
from dotenv import load_dotenv, dotenv_values
from report_ocr import extract_report, ExtractionError, OCRUnavailable
from upload_jobs import UploadJobs

import requests # Import requests for making HTTP calls to Gemini API
import shutil # For removing directories
import time # Import time for sleep
import io # Add this import
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader # FIX: Retain PdfReader only
from PyPDF2.errors import PdfReadError # FIX: Import PdfReadError from the specific errors submodule



GEMINI_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(GEMINI_ENV_PATH)

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
BASE_MODEL_PATH = "models/medical_llm_merged_final" 
ADAPTER_PATH = "models/adapter_after_extra_training" 

OFFLOAD_DIR = "model_offload_cache" 
os.makedirs(OFFLOAD_DIR, exist_ok=True)

USE_LOCAL_TERM_LLM = False

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip().removeprefix("models/")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_TIMEOUT = 60

# === Flask Setup ===
app = Flask(
    __name__,
    static_folder='frontend/dist/assets',
    template_folder='frontend/dist'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.secret_key = os.urandom(24)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# === Global Variables ===
user_vector_stores = {}
pending_reports = {}
upload_jobs = UploadJobs()

# Load the optional language model only when the local term-explanation path uses it.
@lru_cache(maxsize=1)
def get_local_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading final merged model from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map="auto",
        offload_folder=OFFLOAD_DIR
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Creating text generation pipeline on device: {device}")
    gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,  # keep small for speed if you turn local LLM on later
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

# === Embeddings ===
@lru_cache(maxsize=1)
def get_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("[Index] Loading report search model...", flush=True)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Helper: Allowed File Type ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Summarizer (Optional) ===
def summarize_text(text, num_sentences=5):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Could not generate summary."

# === Extract ALL Text from PDF for RAG ===
def extract_full_text_from_pdf_file(file_storage):
    full_text = []
    pdf_bytes = None
    reader = None

    try:
        file_bytes = file_storage.read()
        pdf_bytes = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_bytes)

        if reader.is_encrypted:
            try:
                reader.decrypt('')
            except PdfReadError as e:
                print(f"PDF encryption error (needs password or PyCryptodome): {e}")
                return ""
            except Exception as e:
                print(f"General PDF decryption error: {e}")
                return ""

        for page in reader.pages:
            content = page.extract_text()
            if content:
                full_text.append(content)

        return "\n".join(full_text).strip()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    finally:
        if pdf_bytes is not None:
            pdf_bytes.close()

# === Vector Store Creation ===
def create_vector_store(text):
    if os.getenv('REPORT_SEARCH_BACKEND', 'fast').lower() == 'fast':
        from report_search import ReportSearch
        started = time.perf_counter()
        search = ReportSearch(text)
        print(f'[Index] Report ready in {time.perf_counter() - started:.3f}s (fast local search; no model download).', flush=True)
        return search
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if not text:
        raise ValueError("Cannot create vector store from empty text.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, get_embedding_model())

# === Detect "Explain ... medical term ..." ===
def detect_medical_term(question: str):
    """
    Detects if user is asking to explain a medical term.
    Supports:
      - 'Explain this medical term Abortion'
      - 'Explain the medical term Abortion'
      - 'Explain medical term Abortion'
      - 'Explain the term Abortion'
    """
    q_stripped = question.strip()
    q_lower = q_stripped.lower()

    prefixes = [
        "explain this medical term",
        "explain the medical term",
        "explain medical term",
        "explain the term",
    ]

    for prefix in prefixes:
        if q_lower.startswith(prefix):
            term = q_stripped[len(prefix):].strip(" :?.,")
            return term if term else None
    return None

# === Gemini Formatting Function ===
def get_formatted_gemini_response(raw_medical_text):
    # Read formatter settings on each request so rotating a key does not require
    # restarting the server and reloading OCR models or losing the uploaded report.
    config = dotenv_values(GEMINI_ENV_PATH)
    api_key = (config.get('GEMINI_API_KEY', GEMINI_API_KEY) or '').strip()
    model_name = (config.get('GEMINI_MODEL') or GEMINI_MODEL).strip().removeprefix('models/')
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    if not api_key:
        return "Formatting service is not configured. Set GEMINI_API_KEY in .env and restart the server."
    prompt = f"""The following is a medical analysis or answer generated from a document.
    Please rephrase and summarize the answer clearly and professionally, ensuring it is highly detailed.
    The response must focus on the following three points: the **specific value found in the report**, the **standard limit or normal range**, and a **recommendation based on the result**.

    Structure the response using these exact bold headings, aiming for a total length of 10-12 lines of text:

    **Value:**
    - ... (Extract the numerical value and unit directly from the report.)
    **Standard Limit:**
    - ... (Extract the normal range from the report, if available.)
    **Recommendation:**
    - ... (Provide a clinical recommendation/next step based on the result.)
    **Key Findings:**
    - ... (Briefly state if the result is high/low/normal.)
    **Interpretation:**
    - ... (Explain the clinical meaning of the value.)


    If information for a field is not available in the context, state "Not available in report."
    Use bullet points clearly for each section.

    Medical Content:
    {raw_medical_text}
    """

    headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }

    if model_name.startswith('gemini-3'):
        payload['generationConfig']['thinkingConfig'] = {'thinkingLevel': 'low'}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
        response.raise_for_status()
        result = response.json()

        for candidate in result.get('candidates', []):
            parts = candidate.get('content', {}).get('parts', [])
            text = "\n".join(part['text'] for part in parts if part.get('text') and not part.get('thought'))
            if text.strip():
                return text.strip()
        return "The formatting service returned no usable answer. Please retry."
    except requests.exceptions.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        # Never log the exception, URL, response body or authentication headers.
        app.logger.warning("Formatting request failed (HTTP %s).", status or 'connection error')
        if status == 400:
            try:
                details = exc.response.json().get('error', {}).get('details', [])
                reasons = {detail.get('reason') for detail in details if isinstance(detail, dict)}
            except (ValueError, AttributeError, TypeError):
                reasons = set()
            if reasons.intersection({'API_KEY_INVALID', 'API_KEY_EXPIRED', 'API_KEY_NOT_FOUND',
                                     'API_KEY_SERVICE_BLOCKED', 'API_KEY_HTTP_REFERRER_BLOCKED',
                                     'API_KEY_IP_ADDRESS_BLOCKED'}):
                return "Google rejected the API key. Check the new GEMINI_API_KEY in .env and its API restrictions, then retry."
            return "Google rejected the formatting request (HTTP 400). Check GEMINI_MODEL and the API key settings."
        if status == 404:
            return "Formatting model unavailable. Set GEMINI_MODEL to an available model in .env and restart the server."
        if status in (401, 403):
            return "Formatting service authentication failed. Check GEMINI_API_KEY and its API permissions."
        if status == 429:
            return "Formatting service quota or rate limit reached. Check your Gemini quota and retry later."
        if isinstance(exc, requests.exceptions.Timeout):
            return "The formatting service timed out. Please retry."
        return "Could not reach the formatting service. Check the server connection and retry."
    except (ValueError, KeyError, TypeError, AttributeError):
        app.logger.warning("Formatting service returned an unexpected response.")
        return "The formatting service returned an invalid response. Please retry."

# === Routes ===
@app.route("/")
def serve_react_app():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/<path:filename>')
def serve_root_static_files(filename):
    return send_from_directory(app.template_folder, filename)

@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "Upload a file smaller than 20 MB."}), 413


def process_uploaded_report(user_id, data, filename, progress=None, mode='auto'):
    if progress:
        progress('Reading report...')
    extraction = extract_report(data, filename, progress=progress, mode=mode)
    processing = {'extraction_mode': mode,
                  'extraction_sources': sorted({page['source'] for page in extraction.get('pages', [])}),
                  'answering_method': 'rag'}
    if extraction['requires_review']:
        token = str(uuid.uuid4())
        pending_reports[user_id] = {'token': token, 'created': time.time()}
        # Retain only short-lived review tokens, not medical report content.
        for key, value in list(pending_reports.items()):
            if time.time() - value['created'] > 1800:
                pending_reports.pop(key, None)
        return {"message": "Review and correct the extracted text before asking questions.",
                        "review_token": token, **extraction, **processing}
    if progress:
        progress('Preparing report search...')
    user_vector_stores[user_id] = create_vector_store(extraction['text']).as_retriever(search_type="similarity", k=3)
    return {"message": "Report processed. You can now ask questions.", "requires_review": False, **processing}


@app.route('/upload-status/<job_id>', methods=['GET'])
def upload_status(job_id):
    job = upload_jobs.get(session.get('user_id'), job_id)
    if job is None:
        return jsonify({"error": "Upload status expired or was not found. Upload the report again."}), 404
    return jsonify(job)


@app.route('/upload-status', methods=['GET'])
def current_upload_status():
    user_id = session.get('user_id')
    if user_id and user_id in user_vector_stores:
        return jsonify({'state': 'ready'})
    return jsonify(upload_jobs.latest(user_id) or {'state': 'idle'})


@app.route("/upload", methods=["POST"])
def upload():
    user_id = session.get('user_id') or str(uuid.uuid4())
    session['user_id'] = user_id
    file = request.files.get("file")
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Upload a PDF, PNG, or JPEG report."}), 400
    if upload_jobs.active(user_id):
        job = upload_jobs.latest(user_id)
        return jsonify({"error": "Your previous report is still processing. A refresh does not cancel it.",
                        "job_id": job['job_id'] if job else None}), 409
    mode = request.form.get('extraction_mode', 'auto')
    if mode not in {'auto', 'pdf_text', 'ocr'}:
        return jsonify({"error": "Choose Auto, PDF text only, or Force OCR."}), 400
    if mode == 'pdf_text' and not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "PDF text only requires a PDF. Choose Auto or Force OCR for images."}), 400
    # Never answer from the previous report while a replacement is pending or fails.
    user_vector_stores.pop(user_id, None)
    pending_reports.pop(user_id, None)
    try:
        data = file.read()
        if request.args.get('async') == '1':
            try:
                job_id = upload_jobs.submit(user_id, lambda progress: process_uploaded_report(user_id, data, file.filename, progress, mode=mode))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 409
            return jsonify({"job_id": job_id, "message": "Report accepted. Starting processing..."}), 202
        return jsonify(process_uploaded_report(user_id, data, file.filename, mode=mode))
    except ExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except OCRUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Report processing failed")
        return jsonify({"error": "Report processing failed. Please try again."}), 500


@app.route("/confirm-report", methods=["POST"])
def confirm_report():
    user_id = session.get('user_id')
    pending = pending_reports.get(user_id)
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    if not pending or data.get('review_token') != pending['token'] or time.time() - pending['created'] > 1800:
        return jsonify({"error": "Review expired or replaced. Upload the report again."}), 400
    text = data.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > 200000:
        return jsonify({"error": "Enter reviewed report text (up to 200,000 characters)."}), 400
    try:
        retriever = create_vector_store(text.strip()).as_retriever(search_type="similarity", k=3)
        # A newer upload may have replaced this review while embeddings were created.
        if pending_reports.get(user_id) is not pending:
            return jsonify({"error": "A newer report was uploaded. Review that report instead."}), 409
        user_vector_stores[user_id] = retriever
        pending_reports.pop(user_id, None)
        return jsonify({"message": "Reviewed report ready. You can now ask questions."})
    except Exception:
        app.logger.exception("Reviewed report indexing failed")
        return jsonify({"error": "Could not process the reviewed text. Please retry."}), 500


@app.route("/ask", methods=["POST"])
def ask():
    user_id = session.get('user_id')
    if not user_id or user_id not in user_vector_stores:
        return jsonify({"answer": "**Error:** Please upload a medical report first."})

    retriever = user_vector_stores[user_id]

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "**Error:** Please enter a valid question."})

        print(f"[INFO] User '{user_id}' received question: '{question}'")

        # === BRANCH: medical term explanation ===
        term = detect_medical_term(question)
        if term:
            print(f"[INFO] Detected medical term explanation request for term: '{term}'")

            if USE_LOCAL_TERM_LLM:
                # --- REAL LOCAL LLM PATH (might be slow on your laptop) ---
                local_prompt = f"### Question:\nExplain this medical term {term}\n\n### Answer:\n"
                start_time = time.time()
                local_raw_answer = get_local_llm().invoke(local_prompt)
                print(f"[INFO] Local LLM generation took {time.time() - start_time:.2f} seconds")
                if isinstance(local_raw_answer, list):
                    local_raw_answer = local_raw_answer[0]
                raw_text_for_formatting = str(local_raw_answer)
            else:
                # --- FAST PATH FOR DEMO: ask Gemini directly to explain the term ---
                raw_text_for_formatting = f"Explain this medical term in simple medical language: {term}"

            formatted_answer = get_formatted_gemini_response(raw_text_for_formatting)
            print(f"[INFO] Formatted term explanation from Gemini: {formatted_answer[:200]}...")

            return jsonify({"answer": formatted_answer})

        # === NORMAL FLOW: RAG + GEMINI for report-based questions ===
        docs = retriever.invoke(question)
        if not docs:
            return jsonify({"answer": "I could not find relevant information in the report."})

        context = "\n\n".join(doc.page_content for doc in docs)

        raw_answer = (
            "Use ONLY the following context from the patient's medical report "
            "to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear medical answer based strictly on this context."
        )

        formatted_answer = get_formatted_gemini_response(raw_answer)
        print(f"[INFO] Formatted answer from Gemini: {formatted_answer[:200]}...")

        return jsonify({"answer": formatted_answer})

    except Exception as e:
        print(f"[ERROR] /ask route: {e}")
        return jsonify({"answer": "**Error:** An internal server error occurred while processing your question."})

# === Run App ===
if __name__ == "__main__":
    try:
        app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
    finally:
        upload_jobs.shutdown(wait=False)
        if os.path.exists(OFFLOAD_DIR):
            print(f"Cleaning up offload directory: {OFFLOAD_DIR}")
            shutil.rmtree(OFFLOAD_DIR)
