from abc import ABC, abstractmethod
from typing import List, Dict

class EmbeddingProvider(ABC):
    @abstractmethod
    def validate(self) -> None:
        """
        Validate credentials and model configuration.

        Raises:
            ValueError or RuntimeError if the setup is invalid (e.g. wrong API key, missing project)
        """
        pass
    
    @abstractmethod
    def embed_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a list of documents.

        Args:
            docs (List[Dict]): A list of dictionaries, each containing at least:
                - "filename" (str): The file path or identifier of the document
                - "content" (str): The raw text content of the document

        Returns:
            List[Dict]: A list of dictionaries with:
                - "filename": original filename
                - "content": original content
                - "embedding": list of float values representing the embedding vector
        """
        pass