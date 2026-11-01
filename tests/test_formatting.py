import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import requests


class FormattingTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_formatted_gemini_response']
        self.http = SimpleNamespace(post=Mock(), exceptions=requests.exceptions)
        self.config = {}
        self.ns = dict(dotenv_values=lambda path: self.config, GEMINI_ENV_PATH='test.env', requests=self.http, app=SimpleNamespace(logger=logging.getLogger('formatting-test')),
                       GEMINI_TIMEOUT=60, GEMINI_API_KEY='test-secret-should-never-appear', GEMINI_MODEL='gemini-3.6-flash',
                       GEMINI_API_URL='https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent')
        exec(compile(tree, 'app.py', 'exec'), self.ns)
        self.format = self.ns['get_formatted_gemini_response']

    def test_success_uses_header_auth_and_skips_thought_parts(self):
        self.http.post.return_value.json.return_value = {'candidates': [{'content': {'parts': [
            {'text': 'internal thought', 'thought': True}, {'text': 'Value: 98.1 mg/dL'}, {'text': 'Range: 70-100'}]}}]}
        result = self.format('Synthetic glucose 98.1 mg/dL')
        self.assertEqual(result, 'Value: 98.1 mg/dL\nRange: 70-100')
        args, kwargs = self.http.post.call_args
        self.assertNotIn(self.ns['GEMINI_API_KEY'], args[0])
        self.assertEqual(kwargs['headers']['x-goog-api-key'], self.ns['GEMINI_API_KEY'])
        self.assertEqual(kwargs['timeout'], 60)

    def test_404_and_logs_do_not_expose_credentials(self):
        response = requests.Response()
        response.status_code = 404
        self.http.post.side_effect = requests.exceptions.HTTPError(
            'secret-containing URL: ' + self.ns['GEMINI_API_KEY'], response=response)
        with self.assertLogs('formatting-test', level='WARNING') as captured:
            result = self.format('synthetic text')
        self.assertIn('GEMINI_MODEL', result)
        self.assertNotIn(self.ns['GEMINI_API_KEY'], result + ''.join(captured.output))

    def test_timeout_is_actionable_without_exception_text(self):
        self.http.post.side_effect = requests.exceptions.Timeout(self.ns['GEMINI_API_KEY'])
        result = self.format('synthetic text')
        self.assertIn('timed out', result)
        self.assertNotIn(self.ns['GEMINI_API_KEY'], result)

    def test_missing_key_does_not_send_request(self):
        self.ns['GEMINI_API_KEY'] = ''
        self.assertIn('GEMINI_API_KEY', self.format('synthetic'))
        self.http.post.assert_not_called()

    def test_rotated_key_is_read_without_restart(self):
        self.http.post.return_value.json.return_value = {'candidates': [{'content': {'parts': [{'text': 'OK'}]}}]}
        self.format('synthetic')
        self.config['GEMINI_API_KEY'] = 'new-secret'
        self.format('synthetic')
        self.assertEqual(self.http.post.call_args.kwargs['headers']['x-goog-api-key'], 'new-secret')
        self.config['GEMINI_API_KEY'] = ''
        self.assertIn('not configured', self.format('synthetic'))
        self.assertEqual(self.http.post.call_count, 2)

    def test_google_400_invalid_key_is_not_reported_as_network_failure(self):
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error":{"details":[{"reason":"API_KEY_INVALID"}]}}'
        self.http.post.side_effect = requests.exceptions.HTTPError(response=response)
        result = self.format('synthetic')
        self.assertIn('rejected the API key', result)
        self.assertNotIn('Could not reach', result)

    def test_other_400_is_reported_as_rejected_request(self):
        response = requests.Response()
        response.status_code = 400
        self.http.post.side_effect = requests.exceptions.HTTPError(response=response)
        self.assertIn('HTTP 400', self.format('synthetic'))

    def test_quota_error_is_actionable(self):
        response = requests.Response()
        response.status_code = 429
        self.http.post.side_effect = requests.exceptions.HTTPError(response=response)
        self.assertIn('quota', self.format('synthetic'))

if __name__ == '__main__':
    unittest.main()
