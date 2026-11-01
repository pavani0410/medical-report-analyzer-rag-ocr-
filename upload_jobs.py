"""Bounded, session-owned background uploads for the single-worker Flask app."""
import logging
import threading
import time
import uuid
import queue
from report_ocr import ExtractionError, OCRUnavailable


class UploadJobs:
    def __init__(self, capacity=4, ttl=1800):
        self.capacity = capacity
        self.ttl = ttl
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._thread = None
        self._closing = False

    def submit(self, owner, task):
        with self._lock:
            if self._closing:
                raise ValueError('The report processor is shutting down. Restart the server.')
            now = time.monotonic()
            self._jobs = {key: value for key, value in self._jobs.items()
                          if (value['state'] in ('queued', 'running') or now - value['updated'] < self.ttl)
                          and not (value['owner'] == owner and value['state'] in ('done', 'failed'))}
            active = [job for job in self._jobs.values() if job['state'] in ('queued', 'running')]
            if any(job['owner'] == owner for job in active):
                raise ValueError('Your report is still processing. Wait for it to finish before uploading another.')
            if len(active) >= self.capacity or len(self._jobs) >= 32:
                raise ValueError('The report processor is busy. Please retry shortly.')
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = dict(owner=owner, state='queued', message='Waiting for the report processor...',
                                     updated=now, started=now)
            if self._thread is None:
                # A stuck native OCR call must not keep the Flask process alive on Ctrl+C.
                self._thread = threading.Thread(target=self._work, name='report-upload', daemon=True)
                self._thread.start()
            self._queue.put((job_id, task))
        return job_id

    def _work(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._run(*item)
            finally:
                item = None  # Release the completed task and original upload bytes.
                self._queue.task_done()

    def shutdown(self, wait=True):
        with self._lock:
            if not self._closing:
                self._closing = True
                self._queue.put(None)
        if wait and self._thread:
            self._thread.join()

    def _run(self, job_id, task):
        def progress(message):
            with self._lock:
                self._jobs[job_id].update(message=message, updated=time.monotonic())
        with self._lock:
            self._jobs[job_id]['state'] = 'running'
        try:
            result = task(progress)
            updates = dict(state='done', result=result, message='Report ready.')
        except ExtractionError as exc:
            updates = dict(state='failed', error=str(exc), error_code=400)
        except OCRUnavailable as exc:
            updates = dict(state='failed', error=str(exc), error_code=503)
        except Exception:
            logging.getLogger(__name__).exception('Background report processing failed')
            updates = dict(state='failed', error='Report processing failed. Check the server console for details.', error_code=500)
        with self._lock:
            self._jobs[job_id].update(**updates, updated=time.monotonic())

    def get(self, owner, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job['owner'] != owner:
                return None
            if job['state'] in ('done', 'failed') and time.monotonic() - job['updated'] >= self.ttl:
                del self._jobs[job_id]
                return None
            return {key: value for key, value in job.items() if key not in ('owner', 'started', 'updated')}

    def active(self, owner):
        with self._lock:
            return any(job['owner'] == owner and job['state'] in ('queued', 'running') for job in self._jobs.values())

    def latest(self, owner):
        if not owner:
            return None
        with self._lock:
            now = time.monotonic()
            for job_id, job in reversed(list(self._jobs.items())):
                if job['owner'] != owner:
                    continue
                if job['state'] in ('done', 'failed') and now - job['updated'] >= self.ttl:
                    continue
                return {'job_id': job_id, **{key: value for key, value in job.items()
                        if key not in ('owner', 'started', 'updated')}}
        return None
