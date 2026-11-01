"""Persistent OCR subprocess. A hung native import can be killed safely."""
import atexit
import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

PREFIX = 'OCR_RESULT:'


class OCRWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.responses = None

    def _start(self):
        responses = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.responses = responses
        process = self.process

        def read():
            try:
                for line in process.stdout:
                    if line.startswith(PREFIX.encode()):
                        responses.put(json.loads(line[len(PREFIX):]))
            finally:
                responses.put({'error': 'OCR worker stopped unexpectedly.'})

        threading.Thread(target=read, daemon=True, name='ocr-responses').start()

    def _stop(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()

    def close(self):
        with self.lock:
            self._stop()

    def recognize(self, image, page_number):
        from report_ocr import OCRUnavailable
        data = io.BytesIO()
        image.save(data, format='PNG')
        payload = json.dumps({'image': base64.b64encode(data.getvalue()).decode('ascii'), 'page': page_number}).encode() + b'\n'
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._stop()
                self._start()
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                result = self.responses.get(timeout=max(1, float(os.getenv('OCR_TIMEOUT_SECONDS', '90'))))
            except queue.Empty:
                self._stop()
                raise OCRUnavailable('OCR exceeded its time limit and was stopped. The processor is free for another report. Try PDF text only for a searchable PDF or a clearer image.') from None
            except (BrokenPipeError, OSError):
                self._stop()
                raise OCRUnavailable('OCR worker could not run. Check the installed OCR dependencies.') from None
            if 'error' in result:
                self._stop()
                raise OCRUnavailable(result['error'])
            return result['page']


worker = OCRWorker()
# Shutdown must interrupt an in-flight request rather than wait on its lock.
atexit.register(worker._stop)


def main():
    # Read requests before importing native libraries so pipe writes cannot wait
    # for a model import. Only this disposable process loads the OCR runtime.
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            from PIL import Image
            from report_ocr import _recognize_in_process
            with Image.open(io.BytesIO(base64.b64decode(request['image']))) as image:
                page = _recognize_in_process(image.convert('RGB'), request['page'])
            result = {'page': page}
        except Exception:
            # Do not send native exceptions or document contents to HTTP logs.
            result = {'error': 'OCR could not read this page. Check the OCR installation or try a clearer scan.'}
        sys.stdout.write(PREFIX + json.dumps(result) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
