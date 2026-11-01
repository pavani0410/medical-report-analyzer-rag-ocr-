"""Real background upload + polling test. Uses synthetic data and no external LLM."""
import faulthandler
import io
import os
import time
from unittest.mock import patch

faulthandler.enable()
if os.getenv('OCR_TRACE') == '1':
    faulthandler.dump_traceback_later(60, repeat=True)
started = time.perf_counter()
print('[Smoke] Importing Flask app...', flush=True)
import app
from tests.smoke_ocr import synthetic_report
print(f'[Smoke] App imported in {time.perf_counter() - started:.1f}s', flush=True)
assert app.get_embedding_model.cache_info().misses == 0
client = app.app.test_client()
assert client.get('/').status_code == 200
print('[Smoke] Uploading synthetic report...', flush=True)
request_started = time.perf_counter()
response = client.post('/upload?async=1', data={'file': (io.BytesIO(synthetic_report()), 'synthetic.png')})
assert response.status_code == 202, response.get_json()
job_id = response.get_json()['job_id']
print(f'[Smoke] Upload accepted in {time.perf_counter() - request_started:.1f}s', flush=True)
deadline = time.monotonic() + 300
last_message = None
while time.monotonic() < deadline:
    response = client.get('/upload-status/' + job_id)
    assert response.status_code == 200, response.get_json()
    status = response.get_json()
    if status.get('message') != last_message:
        last_message = status.get('message')
        print('[Smoke]', last_message, flush=True)
    if status['state'] == 'failed':
        raise AssertionError(status)
    if status['state'] == 'done':
        break
    time.sleep(1)
else:
    raise AssertionError('OCR did not complete in five minutes')
result = status['result']
assert result['requires_review']
for value in ('98.1', '13.2', '<5.0'):
    assert value in result['text'], (value, result['text'])
assert app.get_embedding_model.cache_info().misses == 0
print(result['text'], flush=True)
confirmation = client.post('/confirm-report', json={'review_token': result['review_token'], 'text': result['text']})
assert confirmation.status_code == 200, confirmation.get_json()
def grounded_answer(context):
    assert '98.1' in context and '70 - 100' in context, context
    return 'Glucose: 98.1 mg/dL. Report reference: 70 - 100.'
with patch.object(app, 'get_formatted_gemini_response', side_effect=grounded_answer):
    answer = client.post('/ask', json={'question': 'What is my glucose?'})
    assert '98.1' in answer.get_json()['answer'], answer.get_json()
print('[Smoke] PASS: confirmed OCR text is searchable through /ask.', flush=True)

warm_started = time.perf_counter()
warm = app.extract_report(synthetic_report(), 'synthetic.png')
assert warm['requires_review'] and all(value in warm['text'] for value in ('98.1', '13.2', '<5.0'))
print(f'[Smoke] Cached OCR upload completed in {time.perf_counter() - warm_started:.2f}s.', flush=True)
faulthandler.cancel_dump_traceback_later()
print(f'[Smoke] PASS: real background upload returned review text in {time.perf_counter() - started:.1f}s; no search model loaded.', flush=True)
app.upload_jobs.shutdown(wait=True)
