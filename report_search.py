"""Small, in-memory report retrieval without model imports or downloads."""
import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    page_content: str


def tokens(text):
    return re.findall(r"[a-z]+\d*[a-z]*|\d+(?:\.\d+)?", text.lower())


class ReportSearch:
    """BM25 retrieval. Short reports use full context to preserve table details."""

    def __init__(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Cannot search an empty report.')
        self.text = text
        self.passages = []
        # Overlap keeps a table heading or value near chunk boundaries available.
        for start in range(0, len(text), 1000):
            self.passages.append(Passage(text[max(0, start - 200):start + 1200]))
        self.counts = [Counter(tokens(p.page_content)) for p in self.passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) or 1
        self.frequencies = Counter(term for count in self.counts for term in count)
        self.k = 3

    def as_retriever(self, search_type='similarity', k=3):
        self.k = k
        return self

    def invoke(self, question):
        # Lab reports are usually short; full context avoids losing units/ranges
        # and answers paraphrased questions without requiring lexical overlap.
        if len(self.text) <= 12000:
            return [Passage(self.text)]
        query = set(tokens(question))
        scores = []
        total = len(self.passages)
        for index, counts in enumerate(self.counts):
            score = 0
            for term in query:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.frequencies[term]
                weight = math.log(1 + (total - df + .5) / (df + .5))
                score += weight * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * self.lengths[index] / self.average))
            scores.append((score, index))
        ranked = sorted(scores, key=lambda item: (-item[0], item[1]))
        selected = sorted(index for score, index in ranked[:self.k] if score > 0)
        return [self.passages[index] for index in selected]
