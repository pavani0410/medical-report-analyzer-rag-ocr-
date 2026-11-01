import test from 'node:test';
import assert from 'node:assert/strict';
import { waitForUpload } from './uploadProgress.js';

test('returns the completed result and encodes the job ID', async () => {
    const original = globalThis.fetch;
    try {
        globalThis.fetch = async url => {
            assert.equal(url, '/upload-status/job%2Fid');
            return { ok: true, json: async () => ({ state: 'done', result: { requires_review: true, text: 'CRP <5.0' } }) };
        };
        assert.deepEqual(await waitForUpload('job/id', () => {}), { requires_review: true, text: 'CRP <5.0' });
    } finally { globalThis.fetch = original; }
});

test('surfaces worker failures rather than treating them as a ready report', async () => {
    const original = globalThis.fetch;
    try {
        globalThis.fetch = async () => ({ ok: true, json: async () => ({ state: 'failed', error: 'OCR could not start' }) });
        await assert.rejects(waitForUpload('job', () => {}), /OCR could not start/);
    } finally { globalThis.fetch = original; }
});

test('expired or other-session jobs surface the server error', async () => {
    const original = globalThis.fetch;
    try {
        globalThis.fetch = async () => ({ ok: false, json: async () => ({ error: 'Upload status expired' }) });
        await assert.rejects(waitForUpload('job', () => {}), /Upload status expired/);
    } finally { globalThis.fetch = original; }
});

test('unmount cancellation stops polling without contacting the server', async () => {
    const original = globalThis.fetch;
    try {
        globalThis.fetch = () => { assert.fail('must not fetch after cancellation'); };
        const controller = new AbortController();
        controller.abort();
        await assert.rejects(waitForUpload('job', () => {}, controller.signal), { name: 'AbortError' });
    } finally { globalThis.fetch = original; }
});
