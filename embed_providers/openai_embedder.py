import logging

import openai
from openai import OpenAI
from embed_providers import EmbeddingProvider

class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def validate(self):
        try:
            available_models = self.client.models.list()
            model_ids = [model.id for model in available_models.data]

            if self.model not in model_ids:
                raise ValueError(f"❌ Model '{self.model}' does not exist in your OpenAI account.")
        except openai.AuthenticationError as e:
            raise ValueError("❌ OpenAI API key is invalid. Aborting.") from e
        except Exception as e:
            raise RuntimeError(f"❌ Failed to validate OpenAI key: {e}") from e

    def embed_documents(self, docs):
        results = []
        for doc in docs:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=doc["content"]
                )
                embedding = response.data[0].embedding

                if not self.is_valid_embedding(embedding):
                    logging.warning(f"⚠️ Skipping invalid embedding for {doc['filename']}")
                    continue

                results.append({
                    "filename": doc["filename"],
                    "content": doc["content"],
                    "embedding": embedding
                })
            except Exception as e:
                logging.error(f"OpenAI error on {doc['filename']}: {e}")
        return results