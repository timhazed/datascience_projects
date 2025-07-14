import os
import json
import uuid
import logging
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt


class QdrantManager:
    def __init__(self, config: dict, collection_name: str, logger: logging.Logger):
        self.config = config
        self.collection = collection_name
        self.embedding_model = config["embedding_model"]
        self.embedding_dimension = config["embedding_dimension"]
        self.logger = logger
        self.batch_size = config.get("qdrant_batch_size", 200)

        self.connect_to_qdrant()
        self.create_collection()
        self.model = SentenceTransformer(self.embedding_model)

    def connect_to_qdrant(self):
        url = self.config["qdrant_url"]
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.logger.info("Connected to Qdrant successfully")

    def create_collection(self):
        try:
            collections = self.client.get_collections()
            if collections is None or self.collection not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
                self.logger.info(f"Created Qdrant collection: {self.collection}")
        except Exception as e:
            self.logger.error(f"Error creating collection: {str(e)}")
            raise

    def generate_embedding(self, text: str):
        try:
            return self.model.encode(text)
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise

    def _batch_points(self, points, size):
        for i in range(0, len(points), size):
            yield points[i:i + size]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _safe_upsert(self, batch):
        return self.client.upsert(
            collection_name=self.collection,
            points=batch,
            wait=True
        )

    def add_documents(self, document: list) -> dict:
        metrics = {
            "chunks_processed": 0,
            "chunks_inserted": 0,
            "doc_id": None,
            "source_url": None,
            "collection": self.collection,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "error": None
        }

        try:
            self.logger.info(f"Processing document chunks for Qdrant collection: {self.collection}")
            points = []

            for chunk in document:
                metrics["chunks_processed"] += 1
                if metrics["doc_id"] is None:
                    metrics["doc_id"] = chunk.get("doc_id")
                    metrics["source_url"] = chunk.get("source_url")

                content = chunk.get("text", "")
                if not content:
                    self.logger.warning(f"Skipping chunk with no text: {chunk.get('chunk_id')}")
                    continue

                try:
                    embedding = self.generate_embedding(content)
                except Exception as e:
                    self.logger.warning(f"Skipping chunk due to embedding error: {e}")
                    continue

                metadata = {k: v for k, v in chunk.items() if k != "text"}

                points.append(models.PointStruct(
                    id=chunk.get("chunk_id", str(uuid.uuid4())),
                    vector=embedding,
                    payload={"content": content, "metadata": metadata}
                ))

            for batch in self._batch_points(points, self.batch_size):
                try:
                    self._safe_upsert(batch)
                    metrics["chunks_inserted"] += len(batch)
                    self.logger.info(f"Inserted batch of {len(batch)} chunks into Qdrant")
                except Exception as e:
                    # Log failed batch with metadata
                    doc_id = metrics["doc_id"]
                    chunk_ids = [point.id for point in batch]
                    self.logger.error(
                        f"Upsert failed for document {doc_id} - chunks {chunk_ids}: {str(e)}"
                    )
                    raise  # Re-raise to stop processing on failure

            metrics["end_time"] = datetime.now(timezone.utc).isoformat()
            return metrics

        except Exception as e:
            metrics["end_time"] = datetime.now(timezone.utc).isoformat()
            metrics["error"] = {
                "message": str(e),
                "type": type(e).__name__
            }
            self.logger.error(f"Fatal error adding document to Qdrant: {str(e)}")
            return metrics