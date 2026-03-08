# Models/sts_scorer.py
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer

class STSScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        # self.model = SentenceTransformer(model_name)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def compute_similarity(self, s1: str, s2: str) -> float:
        """
        Return cosine similarity between two sentences.
        """
        emb1 = self.model.encode([s1])
        emb2 = self.model.encode([s2])
        
        sim = self.model.similarity(emb1, emb2)[0][0].item()
        return float(sim)

    def distance(self, s1: str, s2: str) -> float:
        """
        Turn similarity into a distance to MINIMIZE in GA:
        smaller distance = captions more similar.
        """
        sim = self.compute_similarity(s1, s2)
        dist = 1 - sim
        return 1.0 - sim  # in [0,2] roughly



"""
if __name__ == "__main__":
    scorer = STSScorer()

    s1 = "children are playing together."
    s2 = "two kids are playing."

    sim = scorer.compute_similarity(s1, s2)
    dist = scorer.distance(s1, s2)

    print("Sentence 1:", s1)
    print("Sentence 2:", s2)
    print("Similarity:", sim)
    print("Distance:", dist)
    
    """