export async function waitForUpload(jobId, onProgress, signal) {
    const started = Date.now();
    let failures = 0;
    while (!signal?.aborted) {
        let response;
        try {
            response = await fetch(`/upload-status/${encodeURIComponent(jobId)}`, {
                signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
            });
        } catch (error) {
            if (signal?.aborted) throw error;
            if (++failures >= 3) throw new Error('Connection to the report processor was lost. Refresh to reconnect to the existing upload.');
        }
        if (response) {
            failures = 0;
            const status = await response.json();
            if (!response.ok) throw new Error(status.error || 'Could not check upload status.');
            if (status.state === 'failed') throw new Error(status.error || 'Report processing failed.');
            if (status.state === 'done') return status.result;
            const elapsed = Math.round((Date.now() - started) / 1000);
            onProgress(`${status.message} (${elapsed}s elapsed)${elapsed >= 300 ? ' — Still running. Refreshing will reconnect; it will not cancel processing.' : ''}`);
        }
        await new Promise((resolve, reject) => {
            const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
            const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 1500);
            signal?.addEventListener('abort', abort, { once: true });
            if (signal?.aborted) abort();
        });
    }
    throw new DOMException('Aborted', 'AbortError');
}
