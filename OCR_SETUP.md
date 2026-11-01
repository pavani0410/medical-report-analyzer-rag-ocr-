# OCR integration

## Current runtime

Report search defaults to `REPORT_SEARCH_BACKEND=fast` and does not load neural
embeddings. `PDF text only` reads embedded PDF text directly, including PDFs with
logos. PDFs containing images require review in that mode because image-only
content is not extracted. Auto still uses OCR when embedded text may be incomplete.

OCR defaults to `OCR_BACKEND=rapid` (RapidOCR + ONNX Runtime), with bundled model
weights. It runs in a separate, persistent local process. A page taking longer than
`OCR_TIMEOUT_SECONDS=90` is stopped, the upload reports an error, and the next job
can proceed. Failed workers are recreated on the next OCR request. A successful
worker reuses the engine across pages and uploads. Every OCR report still requires
text review before questions. Report images are not sent to an external OCR API.

Verified here using synthetic data: direct text-PDF upload/extraction/retrieval took
1.15 seconds; a live Gemini answer preserved the correct value and reference range.
The first ONNX OCR upload took about 58 seconds in one run; a later upload with the
same loaded engine took 1.55 seconds in another run. These are smoke-test timings,
not a promise for other hardware, report sizes, or layouts. Exact test values 98.1,
13.2, and <5.0 survived OCR, review confirmation, retrieval, and the ask route.

Run the app with its installed environment:

    venv\Scripts\python.exe app.py

Checks for the real processing flow:

    venv\Scripts\python.exe -u -m tests.smoke_text
    venv\Scripts\python.exe -u -m tests.smoke_text --live
    venv\Scripts\python.exe -u -m tests.smoke_upload

The `--live` check sends only synthetic extracted text to the configured Gemini API.
The other checks do not invoke an external generation service.

The PaddleOCR-specific settings below apply only when `OCR_BACKEND=paddle` is set.
Install that optional legacy backend with `requirements-ocr-paddle.txt`.

Install the optional local OCR runtime in the application's virtual environment:

    venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
    cd frontend
    npm run build

Start the Flask app as before. The optional Paddle backend downloads pretrained
PaddleOCR models on its first use. Report images are processed locally by OCR; the existing
question-answering flow still sends retrieved report text to Gemini.

If OCR dependencies are unavailable, ordinary text-only PDFs still use the existing
PyPDF2 dependency. Scans return an actionable setup error.

PDF, PNG, JPG and JPEG are accepted (20 MB; PDFs up to 20 pages; images up to
20 megapixels). Text-only PDF pages use embedded text. Pages containing images,
or insufficient embedded text, are rendered with PyMuPDF and OCRed. OCR uses
PP-OCRv5 mobile detection and English mobile recognition. Models load lazily,
CPU by default; set OCR_DEVICE=gpu:0 for an installed GPU-compatible runtime.

OCR uploads return page text, boxes and confidence scores. Neighboring cells are
grouped into rows with | separators. This is geometric row grouping, not a trained
table-structure or clinical-field extractor; merged cells and complex columns
need review. Every OCR upload requires confirmation before indexing or questions.
The browser displays the original file and editable extraction. Regions below
0.90 confidence are listed, but high confidence is not proof of accuracy. Empty
pages are rejected rather than silently omitted. Review tokens expire in 30 minutes.
Reports and review state are process-local; restarting loses them. A single Flask
worker is required with the existing in-memory retrieval storage.

## Fine-tuning later

Set OCR_DETECTION_MODEL_DIR and/or OCR_RECOGNITION_MODEL_DIR to directories
containing exported PaddleOCR inference models. These environment variables are
loaded from .env at startup. Restart after changing model paths.

Before training, collect de-identified, consented reports representing multiple
labs, scanners and layouts. Label text boxes/transcriptions and separately record
correct test/value/unit/reference-range tuples. Hold out patients and preferably
lab templates from training. Measure exact numeric and tuple matches in addition
to character error rate. Never auto-correct decimal points using expected ranges.
Fine-tune recognition for transcription errors, detection for missed text;
row-assignment errors require table extraction work rather than OCR recognition.
No model has been fine-tuned and no accuracy claim is made by this integration.

Official API and training entry point:
https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html

## Verification

    venv\Scripts\python.exe -m unittest discover -s tests -v

Tests isolate OCR from model downloads and the large local language model. For a
real runtime smoke test (downloads models on first run):

    venv\Scripts\python.exe -m tests.smoke_ocr

This synthetic report checks exact decimals and an inequality. Evaluate units and
row associations on representative de-identified reports before making accuracy claims.


## CPU speed defaults

The default loads only mobile detection and English recognition models. Automatic
orientation models are disabled: upload an upright image (JPEG EXIF orientation
is still applied). Set OCR_ORIENTATION=true to enable orientation models if needed;
this adds downloads and processing. Host connectivity probes are skipped by default,
but uncached model files still need downloading. Weights and the engine are reused.
Flask's debug reloader is disabled to avoid duplicate process/model loading.

OCR_CPU_THREADS defaults to at most 4. OCR_DETECTION_MODEL_NAME and
OCR_RECOGNITION_MODEL_NAME can select other models; exported fine-tuned directory
settings still work. Detection is limited to a 1600-pixel longest side internally;
recognition receives the original image. Console timings distinguish model loading
from per-page inference. No latency or accuracy claim has been benchmarked yet.

## Windows CPU compatibility

oneDNN/MKLDNN is disabled by default on Windows after reproducing
ConvertPirAttribute2RuntimeAttribute failures with the installed Paddle runtime.
The ordinary Paddle CPU backend is used instead. OCR_ENABLE_MKLDNN=true can opt
back in after upgrading and verifying a compatible runtime. Inference failures
now print their original traceback in the server console; HTTP errors remain
concise. Restart the Flask process after changing this setting.

## Background uploads and progress

The browser uses POST /upload?async=1, which returns 202 with a job_id immediately.
It polls GET /upload-status/<job_id> for queued, running, done or failed status.
Status is tied to the uploading browser session; another session receives 404.
The same /upload route without the async query still supports synchronous clients.

Only one upload runs at a time so CPU OCR models are reused. The queue is bounded
to four active/queued jobs and one active job per session. Completed results expire
after 30 minutes. Like report retrieval, this state is in memory: use a single
Flask worker, and restarting discards pending work. The browser keeps polling after
five minutes and displays a long-running notice. Refreshing reconnects to the
session's most recent upload via GET /upload-status; a 409 duplicate upload response
also provides the existing job ID for reconnection. Refresh does not cancel native
OCR work. If the runtime is genuinely stuck, stop and restart Flask to clear it.
The original local preview is unavailable after a refresh; open the original file
separately to compare the recovered text before confirming it.

The UI displays model/library preparation, page recognition, and report indexing
stages with elapsed time. Model-load timings include importing the OCR runtime.
An unused LangChain prompt and eager search-model loading were removed from startup;
the default report search now uses lightweight local retrieval without embeddings.
The installed PaddleX compatibility shim still imports NLP dependencies on the first
OCR use, so first-use preparation can be noticeably slower than later uploads.

## Report search and startup speed

The default `REPORT_SEARCH_BACKEND=fast` uses Python-only BM25 retrieval. It does
not import PyTorch, Transformers, sentence-transformers, LangChain, or FAISS for
report indexing. Reports up to 12,000 characters are passed in full to the existing
Gemini answer flow so values, units and reference ranges stay together. Longer
reports retrieve up to three overlapping passages. This removes embedding-model
loading, but lexical search on long reports is less flexible with synonyms than
semantic embeddings. No clinical retrieval-quality equivalence is claimed.

To restore the previous embedding search, set `REPORT_SEARCH_BACKEND=semantic`
in `.env` and restart Flask. With the default backend, text PDFs require no search
model download. OCR remains local and still requires review before questions.

OCR now disables PaddleX eager training-repository initialization and uses complete
model directories already in `~/.paddlex/official_models` directly. Explicit
`OCR_DETECTION_MODEL_DIR` and `OCR_RECOGNITION_MODEL_DIR` still take priority.
The console separates library-import time from total model preparation and page
recognition time. The first OCR call in a new process still imports the native
runtime; keep the Flask process running to reuse its cached engine. Restarting
on every upload repeats that cost. These changes do not reduce image resolution
or remove the OCR review step.

Verify the actual Flask upload and review response with synthetic data:

    venv\Scripts\python.exe -u -m tests.smoke_upload

This uses real OCR and HTTP route handling, never an external LLM or patient data.

## Gemini formatter configuration

Set GEMINI_API_KEY in .env. The default formatter is gemini-3.6-flash;
GEMINI_MODEL can select another model available to the configured project.
The formatter rereads GEMINI_API_KEY and GEMINI_MODEL from .env on each request,
so key rotations do not require restarting or discarding the uploaded report.
The retired gemini-2.0-flash endpoint is no longer used.
The key is sent through x-goog-api-key, never a URL query parameter. Errors return
sanitized setup, authentication, quota or connection messages, and do not echo
provider exception text, request headers or response bodies. Google generation
still sends retrieved report text to the configured Gemini service.

HTTP 400 now distinguishes rejected API keys from rejected requests instead of
reporting both as network failures. Provider response bodies and keys remain hidden.
