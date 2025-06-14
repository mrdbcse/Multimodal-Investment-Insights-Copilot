from fastapi import FastAPI, Form

from core.config import settings
from nlp.embedder import TelecomBillEmbedder
from nlp.retriever import TelecomBillRetriever
from storage.elastic import ElasticClient

app = FastAPI()
elastic_client = ElasticClient(
    index_name="telecom_bill_docs", embedding_dim=settings.EMBEDDING_DIM
)
embedder = TelecomBillEmbedder()
retriever = TelecomBillRetriever(elastic_client=elastic_client)


@app.get("/")
def root():
    return {"message": "Server is running!"}


@app.get("/health")
def health():
    return {"status": "Health is OK"}


@app.post("/upload/telecom_bill/")
def upload_telecom_bill(doc_id: str = Form(...), content: str = Form(...)):
    embedding = embedder.embed(content)
    elastic_client.index_document(doc_id, content, embedding)
    return {"message": "Telecom bill document embedded and indexed!"}


@app.post("/search/telecom_bill/")
def search_telecom_bill(query: str = Form(...)):
    query_emb = embedder.embed(query)
    results = retriever.retrieve_similar(query_emb, k=3)
    return {"top_matches": results}
