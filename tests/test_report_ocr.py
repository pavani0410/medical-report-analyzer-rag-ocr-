import ast
import io
import importlib.util
import time
import unittest
import uuid
import threading
from upload_jobs import UploadJobs
from pathlib import Path
from unittest.mock import patch, Mock

from flask import Flask, request, jsonify, session
from PIL import Image
import report_ocr as ocr

class ExtractionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF runtime not installed')
    def test_text_only_pdf_with_logo_does_not_load_ocr_and_requires_review(self):
        import fitz
        image = io.BytesIO()
        Image.new('RGB', (20, 20), 'white').save(image, format='PNG')
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((30, 30), 'Glucose 98.1 mg/dL reference range 70 to 100')
            page.insert_image(fitz.Rect(30, 60, 50, 80), stream=image.getvalue())
            data = doc.tobytes()
        with patch.object(ocr, 'recognize', side_effect=AssertionError('must not load OCR')):
            result = ocr.extract_report(data, 'logo.pdf', mode='pdf_text')
        self.assertIn('98.1', result['text'])
        self.assertTrue(result['requires_review'])
        self.assertEqual(result['pages'][0]['source'], 'pdf_text')

    def test_text_only_rejects_images_and_invalid_modes(self):
        for mode in ('pdf_text', 'invalid'):
            with self.subTest(mode=mode), self.assertRaises(ocr.ExtractionError):
                ocr.extract_report(b'image', 'report.png', mode=mode)

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF runtime not installed')
    def test_force_ocr_reads_even_embedded_pdf_text(self):
        import fitz
        with fitz.open() as doc:
            doc.new_page().insert_text((30, 30), 'Glucose 98 mg/dL reference range 70 to 100 synthetic report')
            data = doc.tobytes()
        with patch.object(ocr, 'recognize', return_value={'page': 1, 'source': 'ocr', 'text': 'Glucose 98', 'lines': []}) as recognize:
            result = ocr.extract_report(data, 'lab.pdf', mode='ocr')
        recognize.assert_called_once()
        self.assertTrue(result['requires_review'])

    def test_rows_keep_decimals_and_inequalities(self):
        cells = [
            {'text': '<5.0', 'box': [150, 10, 190, 30]},
            {'text': 'CRP', 'box': [10, 11, 60, 31]},
            {'text': 'mg/L', 'box': [230, 12, 270, 30]},
            {'text': 'HbA1c', 'box': [10, 60, 80, 80]},
        ]
        self.assertEqual(ocr.arrange_rows(cells), 'CRP | <5.0 | mg/L\nHbA1c')

    def test_image_review_and_low_confidence_are_preserved(self):
        data = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(data, format='PNG')
        engine = Mock()
        engine.predict.return_value = [{'rec_texts': ['8.1', 'mg/L'], 'rec_scores': [.7, .99],
                                         'rec_boxes': [[10, 10, 30, 25], [50, 10, 80, 25]]}]
        with patch.object(ocr, 'get_engine', return_value=engine):
            page = ocr._recognize_in_process(Image.open(io.BytesIO(data.getvalue())).convert('RGB'), 1)
            result = {'requires_review': True, 'pages': [page], 'low_confidence': [line for line in page['lines'] if line['confidence'] < .9]}
        self.assertTrue(result['requires_review'])
        self.assertEqual(result['pages'][0]['text'], '8.1 | mg/L')
        self.assertEqual(len(result['low_confidence']), 1)
        self.assertEqual(result['low_confidence'][0]['text'], '8.1')

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ocr.ExtractionError):
            ocr.extract_report(b'not an image', 'report.jpg')

    def test_empty_ocr_is_rejected(self):
        data = io.BytesIO()
        Image.new('RGB', (10, 10)).save(data, format='PNG')
        with patch.object(ocr, 'recognize', return_value={'page': 1, 'source': 'ocr', 'text': '', 'lines': []}):
            with self.assertRaises(ocr.ExtractionError):
                ocr.extract_report(data.getvalue(), 'blank.png')

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF runtime not installed')
    def test_mixed_pdf_uses_ocr_only_where_needed(self):
        import fitz
        doc = fitz.open()
        doc.new_page().insert_text((30, 30), 'Hemoglobin 13.2 g/dL reference range 12.0 to 16.0')
        page = doc.new_page()
        image = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(image, format='PNG')
        page.insert_text((30, 30), 'A long header which must not hide the scanned results below')
        page.insert_image(fitz.Rect(30, 60, 130, 160), stream=image.getvalue())
        data = doc.tobytes()
        doc.close()
        with patch.object(ocr, 'recognize', return_value={'page': 2, 'source': 'ocr', 'text': 'Glucose | 98 | mg/dL', 'lines': []}) as recognize:
            result = ocr.extract_report(data, 'mixed.pdf')
        self.assertEqual([p['source'] for p in result['pages']], ['pdf_text', 'ocr'])
        self.assertTrue(result['requires_review'])
        recognize.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF runtime not installed')
    def test_page_limit(self):
        import fitz
        doc = fitz.open()
        for _ in range(ocr.MAX_PAGES + 1):
            doc.new_page()
        data = doc.tobytes()
        doc.close()
        with self.assertRaises(ocr.ExtractionError):
            ocr.extract_report(data, 'long.pdf')

class NativeFallbackTests(unittest.TestCase):
    def test_embedded_text_without_optional_renderer(self):
        from PyPDF2 import PdfWriter
        from PyPDF2.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[0]
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 700 Td (Glucose 98.1 mg/dL reference range 70 to 100 - synthetic report) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        data = io.BytesIO()
        writer.write(data)
        result = ocr.native_pdf_fallback(data.getvalue())
        self.assertIn('98.1', result['text'])
        self.assertFalse(result['requires_review'])

    def test_blank_page_requires_ocr_setup(self):
        from PyPDF2 import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaises(ocr.OCRUnavailable):
            ocr.native_pdf_fallback(data.getvalue())

    def test_invalid_pdf_is_rejected(self):
        with self.assertRaises(ocr.ExtractionError):
            ocr.native_pdf_fallback(b'not a PDF')


class RouteTests(unittest.TestCase):
    def test_selected_mode_is_passed_to_extraction(self):
        self.ns['extract_report'].return_value = {'text': 'Glucose 98', 'requires_review': False}
        response = self.client.post('/upload', data={'file': (io.BytesIO(b'pdf'), 'lab.pdf'), 'extraction_mode': 'ocr'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.ns['extract_report'].call_args.kwargs['mode'], 'ocr')
        self.assertEqual(response.get_json()['extraction_mode'], 'ocr')

    def test_invalid_mode_is_rejected_before_extraction(self):
        response = self.client.post('/upload', data={'file': (io.BytesIO(b'pdf'), 'lab.pdf'), 'extraction_mode': 'bad'})
        self.assertEqual(response.status_code, 400)
        self.ns['extract_report'].assert_not_called()

    def setUp(self):
        # Compile the actual route definitions without loading the multi-GB LLM.
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        names = {'allowed_file', 'upload_too_large', 'upload', 'confirm_report', 'ask', 'process_uploaded_report', 'upload_status', 'current_upload_status'}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        app = Flask(__name__)
        app.secret_key = 'test-only'
        app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
        self.ns = dict(app=app, request=request, jsonify=jsonify, session=session, uuid=uuid, time=time,
                       ALLOWED_EXTENSIONS={'pdf', 'png', 'jpg', 'jpeg'}, user_vector_stores={}, pending_reports={},
                       extract_report=Mock(), create_vector_store=Mock(), upload_jobs=UploadJobs(),
                       ExtractionError=ocr.ExtractionError, OCRUnavailable=ocr.OCRUnavailable)
        exec(compile(tree, 'app.py', 'exec'), self.ns)
        self.client = app.test_client()
        self.addCleanup(self.ns['upload_jobs'].shutdown, wait=True)

    def upload(self):
        return self.client.post('/upload', data={'file': (io.BytesIO(b'image'), 'lab.png')})

    def test_review_gates_questions_and_indexes_corrected_text(self):
        self.ns['extract_report'].return_value = {'text': 'Glucose | 981', 'requires_review': True, 'pages': [], 'low_confidence': []}
        response = self.upload().get_json()
        self.ns['create_vector_store'].assert_not_called()
        answer = self.client.post('/ask', json={'question': 'Glucose?'}).get_json()['answer']
        self.assertIn('upload', answer)
        result = self.client.post('/confirm-report', json={'review_token': response['review_token'], 'text': 'Glucose | 98.1'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.client.get('/upload-status').get_json()['state'], 'ready')
        self.ns['create_vector_store'].assert_called_once_with('Glucose | 98.1')
        repeated = self.client.post('/confirm-report', json={'review_token': response['review_token'], 'text': 'same'})
        self.assertEqual(repeated.status_code, 400)

    def test_failed_replacement_clears_old_report(self):
        with self.client.session_transaction() as sess:
            sess['user_id'] = 'existing'
        self.ns['user_vector_stores']['existing'] = Mock()
        self.ns['extract_report'].side_effect = ocr.ExtractionError('bad image')
        self.assertEqual(self.upload().status_code, 400)
        self.assertNotIn('existing', self.ns['user_vector_stores'])

    def test_missing_runtime_is_actionable(self):
        self.ns['extract_report'].side_effect = ocr.OCRUnavailable('Install requirements-ocr.txt')
        response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertIn('requirements-ocr.txt', response.get_json()['error'])

    def test_expired_review_rejected(self):
        with self.client.session_transaction() as sess:
            sess['user_id'] = 'existing'
        self.ns['pending_reports']['existing'] = {'token': 'old', 'created': time.time() - 1900}
        response = self.client.post('/confirm-report', json={'review_token': 'old', 'text': '98.1'})
        self.assertEqual(response.status_code, 400)

    def test_replaced_review_token_is_rejected(self):
        self.ns['extract_report'].return_value = {'text': 'CRP 5.0', 'requires_review': True, 'pages': [], 'low_confidence': []}
        first = self.upload().get_json()
        second = self.upload().get_json()
        self.assertNotEqual(first['review_token'], second['review_token'])
        response = self.client.post('/confirm-report', json={'review_token': first['review_token'], 'text': 'CRP 5.0'})
        self.assertEqual(response.status_code, 400)
        self.ns['create_vector_store'].assert_not_called()

    def test_other_session_cannot_confirm_review(self):
        self.ns['extract_report'].return_value = {'text': 'CRP 5.0', 'requires_review': True, 'pages': [], 'low_confidence': []}
        review = self.upload().get_json()
        other_client = self.ns['app'].test_client()
        response = other_client.post('/confirm-report', json={'review_token': review['review_token'], 'text': 'CRP 5.0'})
        self.assertEqual(response.status_code, 400)
        self.ns['create_vector_store'].assert_not_called()

    def test_background_upload_returns_before_ocr_and_keeps_status_private(self):
        entered = threading.Event()
        release = threading.Event()
        def extraction(*args, **kwargs):
            kwargs['progress']('Reading page 1...')
            entered.set()
            release.wait(5)
            return {'text': 'Glucose | 98.1', 'requires_review': True, 'pages': [], 'low_confidence': []}
        self.ns['extract_report'].side_effect = extraction
        try:
            response = self.client.post('/upload?async=1', data={'file': (io.BytesIO(b'image'), 'lab.png')})
            self.assertEqual(response.status_code, 202)
            self.assertTrue(entered.wait(2))
            job_id = response.get_json()['job_id']
            other = self.ns['app'].test_client()
            self.assertEqual(other.get('/upload-status/' + job_id).status_code, 404)
            self.assertEqual(other.get('/upload-status').get_json()['state'], 'idle')
            recovered = self.client.get('/upload-status').get_json()
            self.assertEqual(recovered['job_id'], job_id)
            self.assertEqual(recovered['state'], 'running')
            self.assertNotIn('owner', recovered)
            status = self.client.get('/upload-status/' + job_id).get_json()
            self.assertEqual(status['state'], 'running')
            self.assertEqual(status['message'], 'Reading page 1...')
            conflict = self.upload()
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.get_json()['job_id'], job_id)
            self.ns['create_vector_store'].assert_not_called()
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.client.get('/upload-status/' + job_id).get_json()
            if status['state'] == 'done':
                break
            time.sleep(.01)
        self.assertEqual(status['state'], 'done')
        self.assertTrue(status['result']['requires_review'])
        self.assertIn('98.1', status['result']['text'])
        recovered = self.client.get('/upload-status').get_json()
        self.assertEqual(recovered['state'], 'done')
        self.assertEqual(recovered['result']['review_token'], status['result']['review_token'])

    def test_background_failure_reaches_browser(self):
        self.ns['extract_report'].side_effect = ocr.OCRUnavailable('OCR runtime test failure')
        response = self.client.post('/upload?async=1', data={'file': (io.BytesIO(b'image'), 'lab.png')})
        self.assertEqual(response.status_code, 202)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.client.get('/upload-status/' + response.get_json()['job_id']).get_json()
            if status['state'] == 'failed':
                break
            time.sleep(.01)
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['error_code'], 503)
        self.assertEqual(status['error'], 'OCR runtime test failure')

    def test_text_pdf_indexes_immediately(self):
        self.ns['extract_report'].return_value = {'text': 'Glucose 98', 'requires_review': False}
        self.assertEqual(self.upload().status_code, 200)
        self.ns['create_vector_store'].assert_called_once_with('Glucose 98')

if __name__ == '__main__':
    unittest.main()
