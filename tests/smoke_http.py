"""Verify the running local server with synthetic reports and live generation."""
import time
import requests
from tests.smoke_text import synthetic_pdf
from tests.smoke_ocr import synthetic_report


def main():
    client = requests.Session()
    root = 'http://127.0.0.1:5000'
    assert client.get(root, timeout=10).status_code == 200
    for filename, data, mode in [('synthetic.pdf', synthetic_pdf(), 'pdf_text'),
                                 ('synthetic.png', synthetic_report(), 'auto')]:
        started = time.perf_counter()
        response = client.post(root + '/upload?async=1', files={'file': (filename, data)},
                               data={'extraction_mode': mode}, timeout=10)
        assert response.status_code == 202, response.text
        job_id = response.json()['job_id']
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = client.get(root + '/upload-status/' + job_id, timeout=10).json()
            if status['state'] in ('done', 'failed'):
                break
            time.sleep(.5)
        assert status['state'] == 'done', status
        result = status['result']
        if filename.endswith('.png'):
            assert result['requires_review']
            assert all(value in result['text'] for value in ('98.1', '13.2', '<5.0'))
            confirmation = client.post(root + '/confirm-report', json={'review_token': result['review_token'], 'text': result['text']}, timeout=15)
            assert confirmation.status_code == 200, confirmation.text
        print(f'{filename}: processed through HTTP in {time.perf_counter() - started:.2f}s.', flush=True)
        answer = client.post(root + '/ask', json={'question': 'What is my glucose value and report reference range?'}, timeout=75)
        assert answer.status_code == 200 and '98.1' in answer.json()['answer'], answer.text
        print(f'{filename}: live Gemini answer verified.', flush=True)
    print('PASS: running server processed text PDF and OCR image, confirmed review, and answered both via Gemini.', flush=True)


if __name__ == '__main__':
    main()
