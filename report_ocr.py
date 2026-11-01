"""Local OCR extraction; lazy model loading preserves the text-PDF path."""
import io
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

MAX_PAGES = 20
MAX_PIXELS = 20_000_000
_lock = threading.Lock()

class ExtractionError(ValueError):
    pass

class OCRUnavailable(RuntimeError):
    pass

@lru_cache(maxsize=1)
def get_engine():
    started = time.perf_counter()
    print('[OCR] Loading OCR libraries and models...', flush=True)
    try:
        if os.getenv('OCR_BACKEND', 'rapid').lower() == 'rapid':
            from rapidocr_onnxruntime import RapidOCR
            class RapidEngine:
                def __init__(self):
                    self.engine = RapidOCR(use_angle_cls=os.getenv('OCR_ORIENTATION', 'false').lower() == 'true', text_score=0.0)

                def predict(self, image):
                    results, _ = self.engine(image)
                    texts, scores, boxes = [], [], []
                    for points, text, score in results or []:
                        texts.append(text)
                        scores.append(score)
                        boxes.append([min(p[0] for p in points), min(p[1] for p in points),
                                      max(p[0] for p in points), max(p[1] for p in points)])
                    return [{'rec_texts': texts, 'rec_scores': scores, 'rec_boxes': boxes}]
            engine = RapidEngine()
            print(f'[OCR] ONNX engine ready in {time.perf_counter() - started:.1f}s.', flush=True)
            return engine
        # Set before importing PaddleX: its host checks happen during import.
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        # Only inference is used here; avoid initializing PaddleX training repos.
        os.environ.setdefault('PADDLE_PDX_EAGER_INIT', 'False')
        from paddleocr import PaddleOCR
        print(f'[OCR] Libraries imported in {time.perf_counter() - started:.1f}s.', flush=True)
        orientation = os.getenv('OCR_ORIENTATION', 'false').lower() == 'true'
        options = dict(
            text_detection_model_name=os.getenv('OCR_DETECTION_MODEL_NAME', 'PP-OCRv5_mobile_det'),
            text_recognition_model_name=os.getenv('OCR_RECOGNITION_MODEL_NAME', 'en_PP-OCRv5_mobile_rec'),
            device=os.getenv('OCR_DEVICE', 'cpu'),
            # Windows Paddle oneDNN currently fails on a PIR attribute in these models.
            enable_mkldnn=os.getenv('OCR_ENABLE_MKLDNN', 'false' if os.name == 'nt' else 'true').lower() == 'true',
            cpu_threads=int(os.getenv('OCR_CPU_THREADS', str(min(4, os.cpu_count() or 1)))),
            use_doc_orientation_classify=orientation, use_doc_unwarping=False,
            use_textline_orientation=orientation, text_rec_score_thresh=0.0,
            text_det_limit_side_len=1600, text_det_limit_type='max',
        )
        for env, parameter in [('OCR_DETECTION_MODEL_DIR', 'text_detection_model_dir'),
                               ('OCR_RECOGNITION_MODEL_DIR', 'text_recognition_model_dir')]:
            if os.getenv(env):
                options[parameter] = os.environ[env]
            else:
                name_parameter = parameter.replace('_dir', '_name')
                cached = Path.home() / '.paddlex' / 'official_models' / options[name_parameter]
                # A partial download must still go through the normal downloader.
                if (cached / 'inference.yml').is_file() and (cached / 'inference.pdiparams').is_file():
                    options[parameter] = str(cached)
        print('[OCR] Loading lightweight model weights...', flush=True)
        engine = PaddleOCR(**options)
        print(f'[OCR] Models ready in {time.perf_counter() - started:.1f}s; cached for subsequent uploads.', flush=True)
        return engine
    except Exception as exc:
        logging.getLogger(__name__).exception('OCR model initialization failed')
        raise OCRUnavailable('OCR could not start. Install requirements-ocr.txt and allow the initial model download. See OCR_SETUP.md.') from exc

def arrange_rows(lines):
    """Group neighboring cells without inferring missing values or column meanings."""
    rows = []
    for line in sorted(lines, key=lambda item: (item['box'][1], item['box'][0])):
        box = line['box']
        center = (box[1] + box[3]) / 2
        height = max(1, box[3] - box[1])
        match = next((row for row in reversed(rows)
                      if abs(row['center'] - center) <= min(row['height'], height) * .45), None)
        if match is None:
            match = {'center': center, 'height': height, 'cells': []}
            rows.append(match)
        match['cells'].append(line)
    return '\n'.join(' | '.join(cell['text'] for cell in sorted(row['cells'], key=lambda c: c['box'][0])) for row in rows)

def recognize(image, page_number, progress=None):
    from ocr_worker import worker
    if progress:
        progress(f'Reading page {page_number} with local OCR (90-second default time limit)...')
    return worker.recognize(image, page_number)


def _recognize_in_process(image, page_number, progress=None):
    import numpy as np
    if progress:
        progress(f'Preparing OCR for page {page_number} (cached after the first scan)...')
    with _lock:
        engine = get_engine()
        if progress:
            progress(f'Reading page {page_number}...')
        try:
            started = time.perf_counter()
            print(f'[OCR] Reading page {page_number}...', flush=True)
            results = list(engine.predict(np.asarray(image)[:, :, ::-1].copy()))
            print(f'[OCR] Page {page_number} recognition took {time.perf_counter() - started:.1f}s.', flush=True)
        except Exception as exc:
            logging.getLogger(__name__).exception('OCR inference failed on page %s', page_number)
            raise OCRUnavailable('OCR inference failed. See the server console for the underlying error.') from exc
    lines = []
    for result in results:
        for text, score, box in zip(result['rec_texts'], result['rec_scores'], result['rec_boxes']):
            if text.strip():
                lines.append({'text': text, 'confidence': float(score),
                              'box': [int(value) for value in box], 'page': page_number})
    return {'page': page_number, 'source': 'ocr', 'text': arrange_rows(lines), 'lines': lines}

def native_pdf_fallback(data, text_only=False):
    """Retain embedded-text support when the optional PDF renderer is absent."""
    from PyPDF2 import PdfReader

    def contains_image(resources):
        if not resources:
            return False
        resources = resources.get_object() if hasattr(resources, 'get_object') else resources
        objects = resources.get('/XObject', {})
        objects = objects.get_object() if hasattr(objects, 'get_object') else objects
        # Any XObject may contain raster content (including nested forms).
        return bool(objects)

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(''):
            raise ExtractionError('Upload an unlocked PDF without a password.')
        if not 0 < len(reader.pages) <= MAX_PAGES:
            raise ExtractionError(f'Upload a report with 1 to {MAX_PAGES} pages.')
        pages = []
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip() if '/Contents' in page else ''
            images = contains_image(page.get('/Resources'))
            if not text or (not text_only and (len(text) < 40 or images)):
                raise OCRUnavailable('Scanned PDF extraction requires PyMuPDF and PaddleOCR. Install requirements-ocr.txt.')
            pages.append({'page': number, 'source': 'pdf_text', 'text': text, 'lines': [], 'needs_review': images})
        return {'text': '\n\n'.join(f"[Page {p['page']}]\n{p['text']}" for p in pages),
                'pages': pages, 'requires_review': any(p['needs_review'] for p in pages), 'low_confidence': []}
    except (ExtractionError, OCRUnavailable):
        raise
    except Exception as exc:
        raise ExtractionError('This PDF could not be read. Upload a valid, unlocked report.') from exc


def extract_report(data, filename, progress=None, mode='auto'):
    if mode not in {'auto', 'pdf_text', 'ocr'}:
        raise ExtractionError('Choose Auto, PDF text only, or Force OCR.')
    if mode == 'pdf_text' and filename.rsplit('.', 1)[-1].lower() != 'pdf':
        raise ExtractionError('PDF text only requires a PDF. Choose Auto or Force OCR for images.')
    # Ordinary PDFs do not need Pillow, NumPy, a renderer, or an OCR runtime.
    if filename.rsplit('.', 1)[-1].lower() == 'pdf' and mode != 'ocr':
        try:
            return native_pdf_fallback(data, text_only=mode == 'pdf_text')
        except OCRUnavailable:
            if mode == 'pdf_text':
                raise ExtractionError('This PDF has a page without embedded text. Choose Auto or Force OCR for scanned reports.') from None
    from PIL import Image, ImageOps, UnidentifiedImageError
    extension = filename.rsplit('.', 1)[-1].lower()
    pages = []
    if extension == 'pdf':
        try:
            import fitz
        except ImportError:
            if mode == 'ocr':
                raise OCRUnavailable('Force OCR requires PyMuPDF and PaddleOCR. Install requirements-ocr.txt.')
            return native_pdf_fallback(data, text_only=mode == 'pdf_text')
        try:
            with fitz.open(stream=data, filetype='pdf') as document:
                if document.needs_pass:
                    raise ExtractionError('Upload an unlocked PDF without a password.')
                if not 0 < len(document) <= MAX_PAGES:
                    raise ExtractionError(f'Upload a report with 1 to {MAX_PAGES} pages.')
                for index, page in enumerate(document):
                    text = page.get_text('text').strip()
                    # A text header must not hide a scanned table on the same page.
                    if mode == 'pdf_text' and not text:
                        raise ExtractionError('This page has no embedded text. Choose Auto or Force OCR for scanned reports.')
                    if mode == 'pdf_text' or (mode != 'ocr' and len(text) >= 40 and not page.get_images()):
                        pages.append({'page': index + 1, 'source': 'pdf_text', 'text': text, 'lines': [], 'needs_review': bool(page.get_images())})
                    else:
                        scale = min(2.5, (MAX_PIXELS / max(1, page.rect.width * page.rect.height)) ** .5)
                        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
                        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                        pages.append(recognize(image, index + 1, progress=progress))
        except (ExtractionError, OCRUnavailable):
            raise
        except Exception as exc:
            raise ExtractionError('This PDF could not be read. Upload a valid, unlocked report.') from exc
    elif extension in {'png', 'jpg', 'jpeg'}:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ExtractionError('Use an image under 20 megapixels.')
                pages.append(recognize(ImageOps.exif_transpose(image).convert('RGB'), 1, progress=progress))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ExtractionError('This image could not be read. Upload a valid PNG or JPEG.') from exc
    else:
        raise ExtractionError('Upload a PDF, PNG, or JPEG report.')
    if not any(page['text'].strip() for page in pages):
        raise ExtractionError('No readable text was found. Try a clearer scan or photo.')
    if any(not page['text'].strip() for page in pages):
        raise ExtractionError('A page has no readable text. Remove blank pages or use a clearer scan.')
    return {'text': '\n\n'.join(f"[Page {p['page']}]\n{p['text']}" for p in pages),
            'pages': pages, 'requires_review': any(p['source'] == 'ocr' or p.get('needs_review') for p in pages),
            'low_confidence': [line for p in pages for line in p['lines'] if line['confidence'] < .9]}
