import io
import os
import queue
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from ocr_worker import OCRWorker
from report_ocr import OCRUnavailable


class WorkerTests(unittest.TestCase):
    def test_timeout_kills_worker_and_next_request_starts_fresh(self):
        worker = OCRWorker()
        stalled = Mock()
        stalled.poll.return_value = None
        stalled.stdin = io.BytesIO()
        stalled.stdout = io.BytesIO()
        worker.process = stalled
        worker.responses = queue.Queue()
        with patch.dict(os.environ, {'OCR_TIMEOUT_SECONDS': '1'}):
            with self.assertRaisesRegex(OCRUnavailable, 'time limit'):
                worker.recognize(Image.new('RGB', (10, 10)), 1)
        stalled.kill.assert_called_once()
        self.assertIsNone(worker.process)
        fresh = Mock()
        fresh.poll.return_value = None
        fresh.stdin = io.BytesIO()
        fresh.stdout = io.BytesIO()
        page = {'page': 1, 'source': 'ocr', 'text': 'Glucose 98.1', 'lines': []}
        def start():
            worker.process = fresh
            worker.responses = queue.Queue()
            worker.responses.put({'page': page})
        with patch.object(worker, '_start', side_effect=start) as restart:
            self.assertEqual(worker.recognize(Image.new('RGB', (10, 10)), 1), page)
            restart.assert_called_once()
        worker.close()

