from elasticsearch import Elasticsearch

from core.config import settings
import numpy as np


class ElasticClient:
    def __init__(self, index_name: str, embedding_dim: int = settings.EMBEDDING_DIM):
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.es = Elasticsearch(settings.ELASTICSEARCH_URL)

    def _create_index(self):
        if not self.es.indices.exists(index=self.index_name):
            mappings = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "content": {"type": "text"},
                    "doc_id": {"type": "keyword"},
                }
            }

            self.es.indices.create(index=self.index_name, mappings=mappings)
        else:
            print(f"Index '{self.index_name}' already exists.")

    def _index_document(self, doc_id: str, content: str, embedding: np.ndarray):
        doc = {"doc_id": doc_id, "content": content, "embedding": embedding.tolist()}
        self.es.index(index=self.index_name, document=doc)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector,'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()},
                    },
                }
            },
        }
        response = self.es.search(index=self.index_name, query=query)

        return [
            (hit.get("_source", {}).get("content", ""), hit.get("_score", 0.0))
            for hit in response.get("hits", {}).get("hits", [])
        ]
