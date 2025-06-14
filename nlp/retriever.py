from storage.elastic import ElasticClient
import numpy as np


class TelecomBillRetriever:
    def __init__(self, elastic_client: ElasticClient):
        self.elastic_client = elastic_client

    def retrieve_similar(self, query_embedding: np.ndarray, k: int = 3):
        return self.elastic_client.search(query_embedding=query_embedding, k=k)
