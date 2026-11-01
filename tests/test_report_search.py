import ast
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from report_search import ReportSearch


class SearchTests(unittest.TestCase):
    def test_short_report_keeps_all_values_and_ranges_for_paraphrases(self):
        text = 'Glucose | 98.1 | mg/dL | 70-100\nHbA1c | 5.2 | % | <5.7'
        docs = ReportSearch(text).as_retriever(k=3).invoke('How is my blood sugar?')
        self.assertEqual([doc.page_content for doc in docs], [text])

    def test_long_report_retrieves_rare_test_and_preserves_units(self):
        text = ('Routine results normal.\n' * 800 +
                '\nTroponin | 0.02 | ng/mL | <0.04\n' + 'Other results normal.\n' * 800)
        docs = ReportSearch(text).as_retriever(k=3).invoke('Troponin')
        self.assertTrue(docs)
        self.assertTrue(any('Troponin | 0.02 | ng/mL | <0.04' in doc.page_content for doc in docs))
        self.assertLessEqual(len(docs), 3)
        self.assertEqual(ReportSearch(text).invoke('unrelatedzzzz'), [])

    def test_empty_report_rejected(self):
        with self.assertRaises(ValueError):
            ReportSearch('   ')

    def test_default_app_path_does_not_load_neural_dependencies(self):
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_vector_store']
        namespace = {'os': os, 'time': time}
        exec(compile(tree, 'app.py', 'exec'), namespace)
        with patch.dict(os.environ, {'REPORT_SEARCH_BACKEND': 'fast'}), patch.dict(
                'sys.modules', {'torch': None, 'transformers': None, 'langchain_community': None, 'faiss': None}):
            store = namespace['create_vector_store']('CRP | <5.0 | mg/L')
            self.assertIn('<5.0', store.as_retriever().invoke('CRP')[0].page_content)
