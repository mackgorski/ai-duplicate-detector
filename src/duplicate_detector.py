"""Main duplicate detector implementation using embeddings and GPT verification.

This module provides a comprehensive solution for detecting duplicate and related issues in a GitHub repository
using a combination of embedding-based similarity detection and GPT-powered verification.

Key Features:
- Two-stage duplicate detection (embeddings + GPT verification)
- Cluster analysis for groups of similar issues
- Related issue detection and cross-referencing
- Rate limiting and error handling
- Detailed logging of decision process

The duplicate detection process follows these steps:
1. Calculate embeddings for issue content
2. Find similar issues using embedding similarity
3. Verify potential duplicates using GPT
4. Analyze clusters of similar issues
5. Identify related but non-duplicate issues

Classes:
    DuplicateDetector: Main class handling duplicate detection logic

Constants:
    SIMILARITY_CLUSTERING_THRESHOLD (float): Threshold for clustering similar issues (0.01)
    RELATED_ISSUE_THRESHOLD (float): Threshold for identifying related issues (0.82)

Dependencies:
    - openai: For GPT API and embeddings
    - numpy: For vector operations
    - tiktoken: For token counting
    - logging: For detailed logging
"""

import os
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import json
import tiktoken
import logging

import os
from openai import OpenAI
from openai import AzureOpenAI
import numpy as np

from embedding_store import EmbeddingStore
from issue_embedder import IssueEmbedder
from openai_rate_limiter import OpenAIRateLimiter

# Constants - with environment variable fallbacks
SIMILARITY_CLUSTERING_THRESHOLD = 0.01  # Issues within 0.01 similarity score are considered a cluster
# Get RELATED_ISSUE_THRESHOLD from environment or use default
RELATED_ISSUE_THRESHOLD = float(os.environ.get("RELATED_ISSUE_THRESHOLD", 0.82))

class DuplicateDetector:
    """Detects duplicate issues using embeddings and GPT verification.
    
    This class implements a sophisticated duplicate detection system that combines
    embedding-based similarity detection with GPT-powered verification. It can identify
    exact duplicates, similar issues that should be consolidated, and related but
    distinct issues.

    Attributes:
        api_key (str): OpenAI API key for embeddings and GPT
        store (EmbeddingStore): Storage for issue embeddings
        embedder (IssueEmbedder): Handles embedding generation
        gpt_client (OpenAI): OpenAI client for GPT API calls
        embedding_high_threshold (float): Threshold for high-confidence duplicates
        embedding_low_threshold (float): Threshold for requiring GPT verification
        gpt_confidence_threshold (float): Minimum GPT confidence to confirm duplicate
        max_candidates (int): Maximum number of candidates to check
        rate_limiter (OpenAIRateLimiter): Handles API rate limiting
        gpt_tokenizer (Tokenizer): Tokenizer for GPT-3.5-turbo
        last_candidates (List[Dict]): Stores candidates from last check
        last_related_candidates (List[Dict]): Stores related issues from last check
        _last_verification_result (Optional[Tuple]): Stores last GPT verification result
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the duplicate detector.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, reads from environment.
        """

        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

        self.store = EmbeddingStore()
        self.embedder = IssueEmbedder(api_key=self.api_key)
        # Support Azure OpenAI configuration
        if self.azure_endpoint and self.azure_api_version and self.azure_deployment:
            self.gpt_client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
                azure_deployment=self.azure_deployment
            )
        else:
            self.gpt_client = OpenAI(api_key=self.api_key)
        
        # Thresholds for duplicate detection - read from environment variables or use defaults
        self.embedding_high_threshold = 0.95  # Above this, use embedding only
        self.embedding_low_threshold = float(os.environ.get("DUPLICATE_THRESHOLD", 0.85))
        self.gpt_confidence_threshold = 0.7   # Lower threshold from 0.8 to 0.7 to be more aggressive with duplicates
        
        # Maximum number of candidates to check - read from environment or use default
        self.max_candidates = int(os.environ.get("MAX_ISSUES_TO_PROCESS", 20))
        self.rate_limiter = OpenAIRateLimiter()
        self.gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
        
        # Store last run results for logging
        self.last_candidates = []
        self.last_related_candidates = []
        self._last_verification_result = None  # To store the result of the last GPT verification
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the GPT tokenizer.
        
        Args:
            text (str): Text to count tokens for

        Returns:
            int: Number of tokens in the text
        """
        return len(self.gpt_tokenizer.encode(text))
    
    def _construct_gpt_prompt(
        self,
        issue1_title: str,
        issue1_body: str,
        issue2_title: str,
        issue2_body: str
    ) -> str:
        """Construct the prompt for GPT verification.
        
        Creates a detailed prompt that instructs GPT to analyze two issues and determine
        if they are duplicates. The prompt includes specific guidelines for analysis
        and formatting requirements for the response.

        Args:
            issue1_title (str): Title of first issue
            issue1_body (str): Body of first issue
            issue2_title (str): Title of second issue
            issue2_body (str): Body of second issue

        Returns:
            str: Formatted prompt for GPT analysis
        """
        return f"""Compare these two issues and determine if they are duplicates.

Issue 1:
Title: {issue1_title}
Description: {issue1_body}

Issue 2:
Title: {issue2_title}
Description: {issue2_body}

Perform a detailed analysis and classify the relationship between these issues:

Classification Types:
- EXACT: Issues describe exactly the same problem/requirement with no meaningful differences
- SIMILAR: Issues have significant overlap but with some minor differences in scope/approach
- DISTINCT: Issues are fundamentally different despite potential surface similarities

IMPORTANT GUIDELINES:
- Both EXACT and SIMILAR issues should be marked as duplicates (is_duplicate = true)
- Only DISTINCT issues should be marked as non-duplicates (is_duplicate = false)
- Be more aggressive in marking issues as duplicates - in issue tracking, it's better to consolidate similar issues
- Minor differences in implementation details or extra requirements should not prevent marking as duplicates

Analysis Points:
1. Core Problem Match:
   - What specific problem or requirement do they address?
   - Are the fundamental goals the same?
   - Do they solve the same user need?

2. Technical Approach:
   - Do they propose similar technical solutions?
   - Are the implementation details aligned?
   - Do they use the same technologies/methods?

3. Scope Match:
   - Do they cover the same scope of work?
   - Are there differences in the breadth of the solution?
   - Do they target the same user scenarios?

4. Key Differences:
   - What meaningful differences exist between the issues?
   - Are there unique aspects in either issue?
   - Do these differences justify separate tracking?

Respond in JSON format:
{{
    "is_duplicate": boolean,
    "confidence": float between 0 and 1,
    "classification": "EXACT | SIMILAR | DISTINCT",
    "explanation": "Brief but informative explanation of the decision",
    "core_matches": [
        "List specific matching points about the core problem",
        "Be detailed and specific"
    ],
    "technical_matches": [
        "List specific matching technical aspects",
        "Include implementation details"
    ],
    "scope_matches": [
        "List specific matching scope items",
        "Include user scenarios and requirements"
    ],
    "key_differences": [
        "List any meaningful differences",
        "Include scope, approach, or requirement differences"
    ]
}}

Important:
- Provide specific, detailed points in each list
- Don't leave sections empty - if no matches, explain why
- Focus on meaningful comparisons, not surface-level text matching"""
    
    def _construct_cluster_analysis_prompt(
        self,
        main_issue_title: str,
        main_issue_body: str,
        candidate_issues: List[Dict]
    ) -> str:
        """Construct prompt for analyzing a cluster of similar issues.
        
        Creates a prompt that instructs GPT to analyze a group of similar issues
        and determine their relationships, including which issues are duplicates
        and which should be kept as the primary issue.

        Args:
            main_issue_title (str): Title of the main issue being checked
            main_issue_body (str): Body of the main issue being checked
            candidate_issues (List[Dict]): List of similar issues to analyze

        Returns:
            str: Formatted prompt for cluster analysis
        """
        candidates_text = ""
        for i, issue in enumerate(candidate_issues, 1):
            candidates_text += f"""
Candidate {i} (Issue #{issue['number']}):
Title: {issue['title']}
Description: {issue['body'][:1000]}...  # Truncate long descriptions
"""

        return f"""Analyze this cluster of similar issues and determine their relationship.

Main Issue:
Title: {main_issue_title}
Description: {main_issue_body}

Potential Duplicate Candidates:
{candidates_text}

Task:
1. Determine if any of these issues are true duplicates of the main issue
2. Determine if some or all issues should be consolidated into one
3. Identify which issue should be kept open (usually the most comprehensive or oldest)
4. Identify any issues that are related but not duplicates

Respond in JSON format:
{{
    "analysis": "Detailed analysis of the issue cluster",
    "duplicates": [list of issue numbers that are true duplicates],
    "issue_to_keep": issue number of the issue that should remain open,
    "related_issues": [list of issue numbers that are related but not duplicates],
    "consolidation_summary": "Summary of how the issues should be consolidated",
    "keep_reason": "Reason why the selected issue should be kept open",
    "unique_aspects": {{
        "issue_number": ["List of unique aspects from this issue that should be preserved"]
    }}
}}

Important:
- Be thorough in your analysis
- Provide specific reasons for your decisions
- Focus on content similarity, not just wording
- Consider issue comprehensiveness and completeness when deciding which to keep"""