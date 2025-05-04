import os
import logging
import json
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
from .openai_rate_limiter import rate_limited_completion

class IssueLabelClassifier:
    """Classifies issues for automatic labeling using embeddings and content analysis."""
    
    def __init__(
        self, 
        openai_api_key: str, 
        embedding_model: str = "text-embedding-3-large",
        label_set: List[str] = None,
        confidence_threshold: float = 0.70
    ):
        """Initialize the issue labeler.

        Args:
            openai_api_key (str): OpenAI API key for generating embeddings
            embedding_model (str): The OpenAI embedding model to use
            label_set (List[str]): The set of labels to choose from
            confidence_threshold (float): Minimum confidence to apply a label
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.label_set = label_set or []
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Create embeddings for labels if set is provided
        if self.label_set:
            self.label_embeddings = self._create_label_embeddings()
            self.logger.info(f"Created embeddings for {len(self.label_set)} labels")
            
    def _create_label_embeddings(self) -> Dict[str, List[float]]:
        """Create embeddings for each label with descriptions."""
        label_embeddings = {}
        
        for label in self.label_set:
            # Create enhanced prompt for the label to capture its meaning
            label_prompt = f"Issues that should be labeled with '{label}' are about:"
            embedding = self._get_embedding(label_prompt)
            label_embeddings[label] = embedding
            
        return label_embeddings
    
    @rate_limited_completion
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a text using OpenAI API with rate limiting."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def classify_issue(self, issue_title: str, issue_body: str) -> List[str]:
        """Classify the issue and return appropriate labels.
        
        Args:
            issue_title (str): The title of the issue
            issue_body (str): The body/content of the issue
            
        Returns:
            List[str]: List of appropriate labels for the issue
        """
        if not self.label_set:
            self.logger.warning("No label set provided. Cannot classify issue.")
            return []
            
        # Combine title and body for embedding
        issue_text = f"Title: {issue_title}\n\nBody: {issue_body}"
        issue_embedding = self._get_embedding(issue_text)
        
        # Calculate similarity with each label
        label_scores = {}
        for label, label_embedding in self.label_embeddings.items():
            similarity = self._calculate_similarity(issue_embedding, label_embedding)
            label_scores[label] = similarity
            self.logger.debug(f"Label '{label}' score: {similarity}")
        
        # Select labels above the threshold
        selected_labels = [
            label for label, score in label_scores.items() 
            if score >= self.confidence_threshold
        ]
        
        self.logger.info(f"Classified issue with {len(selected_labels)} labels: {selected_labels}")
        return selected_labels
        
    def classify_with_llm(self, issue_title: str, issue_body: str) -> List[str]:
        """Use a large language model to classify the issue.
        This method is an alternative to embedding-based classification.
        
        Args:
            issue_title (str): The title of the issue
            issue_body (str): The body/content of the issue
            
        Returns:
            List[str]: List of appropriate labels for the issue
        """
        if not self.label_set:
            self.logger.warning("No label set provided. Cannot classify issue.")
            return []
        
        try:
            # Prepare prompt with available labels
            prompt = f"""
            Your task is to classify a GitHub issue into one or more appropriate labels.
            
            Available labels: {', '.join(self.label_set)}
            
            Issue Title: {issue_title}
            
            Issue Body:
            {issue_body}
            
            Return only the appropriate labels as a JSON array of strings.
            Only include labels from the available list that are relevant to this issue.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use a capable model for classification
                messages=[
                    {"role": "system", "content": "You are a GitHub issue classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            response_text = response.choices[0].message.content.strip()
            try:
                response_json = json.loads(response_text)
                if "labels" in response_json:
                    labels = response_json["labels"]
                else:
                    # If the model didn't use the "labels" key, try to use the whole response
                    if isinstance(response_json, list):
                        labels = response_json
                    else:
                        labels = list(response_json.values())[0] if response_json else []
            except:
                # Fallback: try to extract a list from the text if JSON parsing fails
                self.logger.warning(f"Failed to parse JSON response: {response_text}")
                labels = []
                
            # Validate that all returned labels are in the allowed set
            valid_labels = [label for label in labels if label in self.label_set]
            
            self.logger.info(f"LLM classified issue with {len(valid_labels)} labels: {valid_labels}")
            return valid_labels
            
        except Exception as e:
            self.logger.error(f"Error using LLM for classification: {e}")
            return []