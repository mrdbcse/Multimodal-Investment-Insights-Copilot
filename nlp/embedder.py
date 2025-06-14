import openai
import numpy as np
from core.config import settings


class TelecomBillEmbedder:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = "text-embedding-ada-002"

    def embed(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(input=text, model=self.model)
        print("response:", response)
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
