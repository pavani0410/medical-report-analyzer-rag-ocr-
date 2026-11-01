"""Exercise real PDF extraction, async upload, retrieval and the ask route."""
import io
import os
import time
import sys
from unittest.mock import patch

from PyPDF2 import PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject, DecodedStreamObject


def synthetic_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    page = writer.pages[0]
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 20 700 Td (SYNTHETIC REPORT - Glucose 98.1 mg/dL reference 70 to 100. CRP less than 5.0 mg/L.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


def main():
    started = time.perf_counter()
    import app
    client = app.app.test_client()
    with patch.object(app, 'get_embedding_model', side_effect=AssertionError('must not load embeddings')):
        response = client.post('/upload?async=1', data={'file': (io.BytesIO(synthetic_pdf()), 'synthetic.pdf'), 'extraction_mode': 'pdf_text'})
        assert response.status_code == 202, response.get_json()
        job_id = response.get_json()['job_id']
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = client.get('/upload-status/' + job_id).get_json()
            if status['state'] in ('done', 'failed'):
                break
            time.sleep(.05)
        assert status['state'] == 'done', status
        assert not status['result']['requires_review'], status
        def answer(context):
            assert '98.1' in context and '70 to 100' in context and 'Glucose' in context
            return 'Verified retrieved report: Glucose 98.1 mg/dL; reference 70 to 100.'
        with patch.object(app, 'get_formatted_gemini_response', side_effect=answer):
            response = client.post('/ask', json={'question': 'What is my glucose?'})
            assert '98.1' in response.get_json()['answer'], response.get_json()
        print(f'PASS: real PDF upload, extraction, retrieval and ask route in {time.perf_counter() - started:.2f}s.', flush=True)
        if os.getenv('VERIFY_LIVE_GEMINI') == '1' or '--live' in sys.argv:
            response = client.post('/ask', json={'question': 'What is my glucose?'})
            result = response.get_json()['answer']
            print('Live synthetic answer:', result, flush=True)
            assert '98.1' in result, 'Live generation failed; see sanitized response above.'
    app.upload_jobs.shutdown(wait=True)


if __name__ == '__main__':
    main()
