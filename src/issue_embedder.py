"""Issue text embedding generation using OpenAI's API.

This module provides functionality to generate and compare embeddings for GitHub issue text
using OpenAI's embedding models. It handles the preparation of text, API calls, retries,
and similarity computations.

Key Features:
- Text preparation combining issue titles and descriptions
- OpenAI API integration with retry logic
- Embedding generation with configurable models
- Cosine similarity calculations
- Similar issue detection with customizable thresholds

Dependencies:
    - openai: For API access and embedding generation
    - numpy: For vector operations and similarity calculations
    - os: For environment variable access
    - typing: For type hints
    - time: For retry delays

The module uses OpenAI's text-embedding-3-large model by default and implements
exponential backoff for API retries.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI
from openai import AzureOpenAI
import time

class IssueEmbedder:
    """Generates and manages embeddings for GitHub issue text using OpenAI's API.
    
    This class provides a comprehensive interface for generating and comparing
    embeddings of GitHub issue text. It handles API authentication, text preparation,
    embedding generation with retries, and similarity calculations.
    
    Attributes:
        api_key (str): OpenAI API key for authentication
        client (OpenAI): OpenAI client instance
        model (str): Name of the embedding model to use
        max_retries (int): Maximum number of retry attempts for API calls
        retry_delay (int): Initial delay in seconds between retries
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedder with API credentials and default settings.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, reads from OPENAI_API_KEY
                                    environment variable.
        """
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

        if self.azure_endpoint and self.azure_api_version and self.azure_deployment:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
                azure_deployment=self.azure_deployment
            )
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        self.model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
    
    def _prepare_text(self, title: str, body: str = "") -> str:
        """Prepare issue text for embedding by combining title and body.
        
        Args:
            title (str): Issue title
            body (str): Issue description/body text
            
        Returns:
            str: Formatted text combining title and body with appropriate labels
        """
        text = f"Title: {title}"
        if body:
            text += f"\nDescription: {body}"
        return text.strip()
    
    def get_embedding(self, title: str, body: str = "") -> np.ndarray:
        """Generate embedding vector for issue text with retry logic.
        
        Args:
            title (str): Issue title
            body (str): Issue description/body text
            
        Returns:
            np.ndarray: Embedding vector as float32 numpy array
            
        Raises:
            Exception: If embedding generation fails after all retries
            
        The method implements exponential backoff for retries with configurable
        initial delay and maximum attempts.
        """
        text = self._prepare_text(title, body)
        retries = 0
        
        while retries < self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise
                
                # Exponential backoff
                wait_time = self.retry_delay * (2 ** (retries - 1))
                print(f"Error getting embedding, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score between 0 and 1
            
        The cosine similarity is calculated as the dot product of the normalized vectors,
        providing a measure of their directional similarity.
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def find_similar_issues(
        self,
        embedding: np.ndarray,
        candidates: List[Tuple[int, np.ndarray]],
        threshold: float = 0.8,
        max_results: int = 5
    ) -> List[Tuple[int, float]]:
        """Find similar issues based on embedding similarity scores.
        
        Args:
            embedding (np.ndarray): Target embedding to compare against
            candidates (List[Tuple[int, np.ndarray]]): List of (issue_number, embedding) pairs
            threshold (float, optional): Minimum similarity score (0-1). Defaults to 0.8
            max_results (int, optional): Maximum number of results. Defaults to 5
            
        Returns:
            List[Tuple[int, float]]: List of (issue_number, similarity_score) pairs,
                                    sorted by similarity in descending order
                                    
        The method computes similarities with all candidates, filters by threshold,
        sorts by score, and returns the top matches up to max_results.
        """
        similarities = [
            (num, self.compute_similarity(embedding, cand_emb))
            for num, cand_emb in candidates
        ]
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and limit results
        return [
            (num, score) for num, score in similarities
            if score >= threshold
        ][:max_results] 