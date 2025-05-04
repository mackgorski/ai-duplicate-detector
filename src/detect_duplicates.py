"""GitHub Issue Duplicate Detector using embeddings and GPT verification.

This script provides functionality to detect and handle duplicate GitHub issues using a combination of 
embedding-based similarity detection and GPT-powered verification. It can also identify and cross-reference 
related but non-duplicate issues.

Key Features:
- Embedding-based initial similarity detection
- GPT-powered verification of potential duplicates
- Handling of issue priorities (Epic > Task > Sub-task)
- Cross-referencing of related issues
- Comprehensive rate limiting for API calls
- Robust error handling and logging
- Fallback mechanisms for API failures

Environment Variables Required:
    GITHUB_TOKEN: GitHub API authentication token
    OPENAI_API_KEY: OpenAI API key for embeddings and GPT
    GITHUB_REPOSITORY: Repository in format "owner/repo"
    ISSUE_NUMBER: Number of the issue to check

Dependencies:
    - github (PyGithub)
    - requests
    - openai (imported via duplicate_detector)
    - numpy (imported via duplicate_detector)

The script uses a two-stage approach for duplicate detection:
1. Initial embedding-based similarity check
2. GPT verification for medium-confidence matches

For issues that are related but not duplicates, it adds cross-reference comments to help
track relationships between issues.
"""

import os
import sys
import json
import time
import logging
import traceback
import re  # Add re module for regex in rate limiter
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from github import Github
from github.Issue import Issue
from github.Repository import Repository
from github.GithubException import GithubException
import requests
from requests.exceptions import RequestException

# Import the duplicate detector
from duplicate_detector import DuplicateDetector, RELATED_ISSUE_THRESHOLD
from .issue_labeler import IssueLabelClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('duplicate_detection.log')
    ]
)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1
CACHE_DIR = Path(".github/cache")
# Define only relevant paths - remove progress/cache file references
# PROGRESS_FILE = CACHE_DIR / "duplicate_detection_progress.json"
# ISSUES_CACHE_FILE = CACHE_DIR / "issues_cache.json"

def log_rate_limit_status(response, context: str):
    """Logs GitHub API rate limit status from response headers.
    
    Args:
        response: The response object from a GitHub API call
        context (str): Description of the API call context for logging
        
    The function extracts and logs:
        - Rate limit quota
        - Remaining calls
        - Used calls
        - Reset time
        - Resource type
    """
    try:
        reset_timestamp = int(response.headers.get('x-ratelimit-reset', 0))
        reset_time_str = datetime.utcfromtimestamp(reset_timestamp).isoformat() + 'Z' if reset_timestamp else 'N/A'
        headers_to_log = {
            'limit': response.headers.get('x-ratelimit-limit'),
            'remaining': response.headers.get('x-ratelimit-remaining'),
            'used': response.headers.get('x-ratelimit-used'),
            'reset': reset_time_str,
            'resource': response.headers.get('x-ratelimit-resource')
        }
        logging.info(f"GitHub API Rate Limit Status after {context}: {headers_to_log}")
    except Exception as e:
        logging.warning(f"Could not log rate limit status after {context}: {str(e)}")

class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls with separate tracking for embeddings and GPT.
    
    This class manages rate limits for both embedding and GPT API calls to prevent exceeding
    OpenAI's rate limits. It tracks:
    - Requests per minute
    - Tokens per minute
    - Tokens per day
    - Requests per day (for GPT)
    
    The class implements exponential backoff for retries and maintains usage statistics.
    
    Attributes:
        embedding_rpm (int): Max embedding requests per minute
        embedding_tpm (int): Max embedding tokens per minute
        embedding_tpd (int): Max embedding tokens per day
        gpt_rpm (int): Max GPT requests per minute
        gpt_tpm (int): Max GPT tokens per minute
        gpt_tpd (int): Max GPT tokens per day
        gpt_rpd (int): Max GPT requests per day
    """
    
    def __init__(self):
        """Initialize rate limiters for both models."""
        # text-embedding-3-large limits
        self.embedding_rpm = 3000  # requests per minute
        self.embedding_tpm = 1_000_000  # tokens per minute
        self.embedding_tpd = 3_000_000  # tokens per day
        
        # gpt-3.5-turbo-1106 limits
        self.gpt_rpm = 500  # requests per minute
        self.gpt_tpm = 200_000  # tokens per minute
        self.gpt_tpd = 2_000_000  # tokens per day
        self.gpt_rpd = 10_000  # requests per day
        
        # Tracking for embeddings
        self.embedding_requests = []  # [(timestamp, tokens)]
        self.embedding_daily_tokens = []  # [(timestamp, tokens)]
        
        # Tracking for GPT
        self.gpt_requests = []  # [(timestamp, tokens)]
        self.gpt_daily_tokens = []  # [(timestamp, tokens)]
        self.gpt_daily_requests = []  # [timestamp]
        
        self.last_retry_wait = 1  # Initial retry wait in seconds
        
        # Stats for logging
        self.embedding_api_calls = 0
        self.gpt_api_calls = 0
        self.embedding_tokens = 0
        self.gpt_tokens = 0
        self.rate_limit_retries = 0
    
    def _clean_old_records(self, records: List[Tuple[float, int]], window_seconds: int) -> List[Tuple[float, int]]:
        """Remove records older than the window.
        
        Args:
            records: List of (timestamp, tokens) tuples
            window_seconds: Time window in seconds
            
        Returns:
            List of records within the time window
        """
        now = time.time()
        return [(t, tokens) for t, tokens in records if now - t < window_seconds]
    
    def _clean_old_timestamps(self, timestamps: List[float], window_seconds: int) -> List[float]:
        """Remove timestamps older than the window.
        
        Args:
            timestamps: List of timestamps
            window_seconds: Time window in seconds
            
        Returns:
            List of timestamps within the time window
        """
        now = time.time()
        return [t for t in timestamps if now - t < window_seconds]
    
    def _sum_tokens(self, records: List[Tuple[float, int]]) -> int:
        """Sum the tokens from records.
        
        Args:
            records: List of (timestamp, tokens) tuples
            
        Returns:
            Total token count
        """
        return sum(tokens for _, tokens in records)
    
    def wait_if_needed_embedding(self, tokens: int):
        """Wait if approaching embedding rate limits.
        
        Args:
            tokens: Number of tokens in the upcoming request
            
        This method implements a waiting strategy to prevent hitting rate limits:
        - Waits if close to requests per minute limit
        - Waits if close to tokens per minute limit
        - Waits if close to daily token limit
        """
        now = time.time()
        
        # Clean old records
        self.embedding_requests = self._clean_old_records(self.embedding_requests, 60)  # 1 minute window
        self.embedding_daily_tokens = self._clean_old_records(self.embedding_daily_tokens, 86400)  # 24 hour window
        
        # Check request rate
        if len(self.embedding_requests) >= self.embedding_rpm:
            oldest = self.embedding_requests[0][0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logging.info(f"Approaching embedding RPM limit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.embedding_requests = []
        
        # Check token rate (per minute)
        minute_tokens = self._sum_tokens(self.embedding_requests)
        if minute_tokens + tokens > self.embedding_tpm:
            logging.info(f"Approaching embedding TPM limit. Waiting for next minute window...")
            time.sleep(60)
            self.embedding_requests = []
        
        # Check daily token rate
        daily_tokens = self._sum_tokens(self.embedding_daily_tokens)
        if daily_tokens + tokens > self.embedding_tpd:
            logging.warning(f"Daily embedding token limit reached. Waiting for reset...")
            time.sleep(3600)  # Wait an hour and try again
            self.embedding_daily_tokens = []
    
    def wait_if_needed_gpt(self, tokens: int):
        """Wait if approaching GPT rate limits.
        
        Args:
            tokens: Number of tokens in the upcoming request
            
        This method implements a waiting strategy to prevent hitting rate limits:
        - Waits if close to requests per minute limit
        - Waits if close to tokens per minute limit
        - Waits if close to daily request limit
        - Waits if close to daily token limit
        """
        now = time.time()
        
        # Clean old records
        self.gpt_requests = self._clean_old_records(self.gpt_requests, 60)  # 1 minute window
        self.gpt_daily_tokens = self._clean_old_records(self.gpt_daily_tokens, 86400)  # 24 hour window
        self.gpt_daily_requests = self._clean_old_timestamps(self.gpt_daily_requests, 86400)  # 24 hour window
        
        # Check request rate (per minute)
        if len(self.gpt_requests) >= self.gpt_rpm:
            oldest = self.gpt_requests[0][0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logging.info(f"Approaching GPT RPM limit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.gpt_requests = []
        
        # Check token rate (per minute)
        minute_tokens = self._sum_tokens(self.gpt_requests)
        if minute_tokens + tokens > self.gpt_tpm:
            logging.info(f"Approaching GPT TPM limit. Waiting for next minute window...")
            time.sleep(60)
            self.gpt_requests = []
        
        # Check daily request rate
        if len(self.gpt_daily_requests) >= self.gpt_rpd:
            oldest = self.gpt_daily_requests[0]
            wait_time = 86400 - (now - oldest)
            logging.warning(f"Daily GPT request limit reached. Waiting {wait_time/3600:.1f} hours...")
            time.sleep(wait_time)
            self.gpt_daily_requests = []
        
        # Check daily token rate
        daily_tokens = self._sum_tokens(self.gpt_daily_tokens)
        if daily_tokens + tokens > self.gpt_tpd:
            logging.warning(f"Daily GPT token limit reached. Waiting for reset...")
            time.sleep(3600)  # Wait an hour and try again
            self.gpt_daily_tokens = []
    
    def record_embedding_usage(self, tokens: int):
        """Record embedding API usage.
        
        Args:
            tokens: Number of tokens used in the request
        """
        now = time.time()
        self.embedding_requests.append((now, tokens))
        self.embedding_daily_tokens.append((now, tokens))
        self.embedding_api_calls += 1
        self.embedding_tokens += tokens
    
    def record_gpt_usage(self, tokens: int):
        """Record GPT API usage.
        
        Args:
            tokens: Number of tokens used in the request
        """
        now = time.time()
        self.gpt_requests.append((now, tokens))
        self.gpt_daily_tokens.append((now, tokens))
        self.gpt_daily_requests.append(now)
        self.gpt_api_calls += 1
        self.gpt_tokens += tokens
    
    def handle_rate_limit_error(self, error_message: str) -> float:
        """Handle rate limit error with exponential backoff.
        
        Args:
            error_message: Error message from the API
            
        Returns:
            float: Wait time in seconds before next retry
            
        This method implements exponential backoff with a maximum wait time of 5 minutes.
        It also extracts wait time from error messages when available.
        """
        wait_match = re.search(r"wait (\d+) seconds", error_message)
        wait_time = int(wait_match.group(1)) if wait_match else self.last_retry_wait
        
        wait_time += 2  # Add buffer
        logging.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        
        self.last_retry_wait = min(wait_time * 2, 300)  # Cap at 5 minutes
        self.rate_limit_retries += 1
        return wait_time
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics for logging.
        
        Returns:
            Dict containing:
            - embedding_api_calls: Total embedding API calls
            - embedding_tokens: Total embedding tokens used
            - gpt_api_calls: Total GPT API calls
            - gpt_tokens: Total GPT tokens used
            - rate_limit_retries: Number of rate limit retries
        """
        return {
            "embedding_api_calls": self.embedding_api_calls,
            "embedding_tokens": self.embedding_tokens,
            "gpt_api_calls": self.gpt_api_calls,
            "gpt_tokens": self.gpt_tokens,
            "rate_limit_retries": self.rate_limit_retries
        }

def retry_with_backoff(func, max_retries=MAX_RETRIES, initial_delay=RETRY_DELAY):
    """Decorator to retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        Decorated function that implements retry logic
        
    The decorator implements:
    - Exponential backoff between retries
    - Special handling for rate limit errors
    - Detailed logging of retry attempts
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, GithubException) as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    (hasattr(e, 'status_code') and e.status_code == 429) or 
                    (hasattr(e, 'status') and e.status == 429) or
                    "rate limit" in str(e).lower()
                )
                
                if is_rate_limit:
                    # For rate limit errors, get the retry time from headers if available
                    retry_after = None
                    if hasattr(e, 'response') and e.response and 'Retry-After' in e.response.headers:
                        retry_after = int(e.response.headers['Retry-After'])
                    
                    if retry_after:
                        logging.warning(f"Rate limit hit, waiting {retry_after}s as specified in header")
                        time.sleep(retry_after + 1)  # Add 1s buffer
                    else:
                        logging.warning(f"Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                else:
                    logging.warning(f"Request failed with {str(e)}, retrying in {delay}s ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
    
    return wrapper

@retry_with_backoff
def get_github_api_headers(token: str) -> Dict[str, str]:
    """Generate headers for GitHub API requests.
    
    Args:
        token: GitHub API token
        
    Returns:
        Dict containing authorization and content-type headers
    """
    return {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

def get_repo_info(repository: str) -> tuple[str, str]:
    """Extract owner and repository name from repository string.
    
    Args:
        repository: Repository string in format "owner/repo"
        
    Returns:
        Tuple of (owner, repo)
        
    Raises:
        EnvironmentError: If repository string is not provided
    """
    if not repository:
        logging.error("GITHUB_REPOSITORY environment variable not set")
        raise EnvironmentError("GITHUB_REPOSITORY environment variable not set")
    return tuple(repository.split('/'))

@retry_with_backoff
def get_issue_sub_issues(owner: str, repo: str, issue_number: int, headers: Dict[str, str]) -> List[int]:
    """Get list of sub-issues for a given issue with retries.
    
    Args:
        owner: Repository owner
        repo: Repository name
        issue_number: Issue number to check
        headers: GitHub API headers
        
    Returns:
        List of sub-issue numbers
        
    This function:
    - Fetches the issue timeline
    - Extracts cross-referenced issues
    - Handles rate limits and retries
    - Logs rate limit status
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    log_rate_limit_status(response, f"getting sub-issues for #{issue_number}") # Log headers
    
    sub_issues = []
    for event in response.json():
        if event.get('event') == 'cross-referenced':
            source = event.get('source', {})
            if source.get('type') == 'issue' and source.get('issue', {}).get('number'):
                sub_issues.append(source['issue']['number'])
    
    logging.debug(f"Found {len(sub_issues)} sub-issues for issue #{issue_number}")
    return sub_issues

@retry_with_backoff
def transfer_sub_issues(owner: str, repo: str, from_issue: int, to_issue: int, headers: Dict[str, str]):
    """Transfer sub-issues from one issue to another with retries.
    
    Args:
        owner: Repository owner
        repo: Repository name
        from_issue: Source issue number
        to_issue: Target issue number
        headers: GitHub API headers
        
    This function:
    - Gets sub-issues from source issue
    - Removes them from source issue
    - Adds them to target issue
    - Handles rate limits and retries
    - Logs detailed progress
    """
    try:
        # get_issue_sub_issues already logs its rate limit status
        sub_issues = get_issue_sub_issues(owner, repo, from_issue, headers)
        if not sub_issues:
            logging.info(f"No sub-issues to transfer from #{from_issue} to #{to_issue}")
            return
        
        logging.info(f"Transferring {len(sub_issues)} sub-issues from #{from_issue} to #{to_issue}")
        
        success_count = 0
        for sub_issue in sub_issues:
            try:
                # Remove from old parent
                url_del = f"https://api.github.com/repos/{owner}/{repo}/issues/{from_issue}/timeline" # Renamed url variable
                response_del = requests.delete(url_del, headers=headers, timeout=30) # Renamed response variable
                response_del.raise_for_status()
                log_rate_limit_status(response_del, f"removing sub-issue #{sub_issue} from #{from_issue}") # Log headers
                
                # Add to new parent
                url_add = f"https://api.github.com/repos/{owner}/{repo}/issues/{to_issue}/timeline" # Renamed url variable
                data = {"event": "cross-referenced", "source": {"type": "issue", "issue": sub_issue}}
                response_add = requests.post(url_add, headers=headers, json=data, timeout=30) # Renamed response variable
                response_add.raise_for_status()
                log_rate_limit_status(response_add, f"adding sub-issue #{sub_issue} to #{to_issue}") # Log headers
                
                success_count += 1
            except Exception as e:
                logging.error(f"Failed to transfer sub-issue #{sub_issue}: {str(e)}")
        
        logging.info(f"Successfully transferred {success_count}/{len(sub_issues)} sub-issues")
    except Exception as e:
        logging.error(f"Failed to transfer sub-issues from #{from_issue} to #{to_issue}: {str(e)}")
        raise

@retry_with_backoff
def add_issue_comment(owner: str, repo: str, issue_number: int, body: str, headers: Dict[str, str]) -> bool:
    """Add a comment to an issue with retries.
    
    Args:
        owner: Repository owner
        repo: Repository name
        issue_number: Issue number to comment on
        body: Comment text
        headers: GitHub API headers
        
    Returns:
        bool: True if comment was added successfully
        
    This function:
    - Posts comment via GitHub API
    - Handles rate limits and retries
    - Logs rate limit status and errors
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        data = {"body": body}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        log_rate_limit_status(response, f"adding comment to #{issue_number}") # Log headers
        logging.info(f"Added comment to issue #{issue_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to add comment to issue #{issue_number}: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response: {e.response.status_code} - {e.response.text[:200]}")
        return False

@retry_with_backoff
def add_issue_label(owner: str, repo: str, issue_number: int, label: str, headers: Dict[str, str]) -> bool:
    """Add a label to an issue with retries.
    
    Args:
        owner: Repository owner
        repo: Repository name
        issue_number: Issue number to label
        label: Label to add
        headers: GitHub API headers
        
    Returns:
        bool: True if label was added successfully
        
    This function:
    - Adds label via GitHub API
    - Handles rate limits and retries
    - Logs rate limit status and errors
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/labels"
        data = {"labels": [label]}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        log_rate_limit_status(response, f"adding label '{label}' to #{issue_number}") # Log headers
        logging.info(f"Added label '{label}' to issue #{issue_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to add label to issue #{issue_number}: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response: {e.response.status_code} - {e.response.text[:200]}")
        return False

@retry_with_backoff
def close_issue(owner: str, repo: str, issue_number: int, headers: Dict[str, str]) -> bool:
    """Close an issue with retries.
    
    Args:
        owner: Repository owner
        repo: Repository name
        issue_number: Issue number to close
        headers: GitHub API headers
        
    Returns:
        bool: True if issue was closed successfully
        
    This function:
    - Closes issue via GitHub API
    - Sets state_reason to "not_planned"
    - Handles rate limits and retries
    - Logs rate limit status and errors
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        data = {"state": "closed", "state_reason": "not_planned"}
        response = requests.patch(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        log_rate_limit_status(response, f"closing issue #{issue_number}") # Log headers
        logging.info(f"Closed issue #{issue_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to close issue #{issue_number}: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response: {e.response.status_code} - {e.response.text[:200]}")
        return False

def create_duplicate_comment(
    duplicate_number: int,
    explanation: str,
    similarity_score: float,
    analysis_details: Optional[dict] = None
) -> str:
    """Create a well-formatted comment for a duplicate issue.
    
    Args:
        duplicate_number: Number of the original issue
        explanation: Explanation of why issues are duplicates
        similarity_score: Similarity score between issues
        analysis_details: Optional dict with detailed GPT analysis
        
    Returns:
        str: Formatted markdown comment
        
    The comment includes:
    - Duplicate notice header
    - Analysis summary table
    - Core problem matches
    - Technical approach matches
    - Scope matches
    - Key differences
    - Detailed explanation
    - Link to original issue
    """
    
    # Create a direct link to the original issue
    issue_link = f"##{duplicate_number}"
    
    # Case 1: Embedding-only detection (no GPT analysis)
    if analysis_details is None or not any(analysis_details.get(key, []) for key in ['core_matches', 'technical_matches', 'scope_matches', 'key_differences']):
        return f"""🔄 **Duplicate Issue Notice**

This issue appears to be a duplicate of #{duplicate_number}. Here's the analysis:

### Analysis Summary
| Metric | Value |
|--------|-------|
| Detection Method | Embedding-based similarity detection |
| Similarity Score | {similarity_score:.2f} |

### Explanation
This duplicate was detected based on high content similarity between the issues. The similarity score exceeds our high-confidence threshold, so no additional GPT analysis was needed.

---
*This issue will be closed as a duplicate. Please continue the discussion in the original issue: {issue_link}*"""

    # Case 2: Full GPT analysis
    core_matches = analysis_details.get('core_matches', [])
    technical_matches = analysis_details.get('technical_matches', [])
    scope_matches = analysis_details.get('scope_matches', [])
    key_differences = analysis_details.get('key_differences', [])
    
    sections = []
    
    sections.append(f"""### Analysis Summary
| Metric | Value |
|--------|-------|
| Detection Method | Full GPT Analysis |
| Classification | {analysis_details.get('classification', 'SIMILAR')} |
| Similarity Score | {similarity_score:.2f} |
| Confidence | {analysis_details.get('confidence', 0.0):.2f} |
""")
    if core_matches:
        sections.append(f"""### 🎯 Core Problem Matches
{chr(10).join(f"- {match}" for match in core_matches)}""")
        
    if technical_matches:
        sections.append(f"""### 🔧 Technical Approach Matches
{chr(10).join(f"- {match}" for match in technical_matches)}""")
        
    if scope_matches:
        sections.append(f"""### 🔍 Scope Matches
{chr(10).join(f"- {match}" for match in scope_matches)}""")
        
    if key_differences:
        sections.append(f"""### 🔍 Key Differences
{chr(10).join(f"- {diff}" for diff in key_differences)}""")
    
    if explanation:
        sections.append(f"""### 📝 Explanation
{explanation}""")
    
    sections.append("---")
    sections.append(f"*This issue will be closed as a duplicate. Please continue the discussion in the original issue: {issue_link}*")
    
    return f"""## 🔄 **Duplicate Issue Notice**

This issue appears to be a duplicate of #{duplicate_number}. Here's the analysis:

{chr(10).join(chr(10) + section for section in sections)}"""

def move_sub_issues(repo: Repository, from_issue: Issue, to_issue: Issue) -> bool:
    """Move sub-issues from one issue to another using PyGithub API with error handling.
    
    Args:
        repo: PyGithub Repository object
        from_issue: Source Issue object
        to_issue: Target Issue object
        
    Returns:
        bool: True if sub-issues were moved successfully
        
    This function:
    - Extracts repository owner and name
    - Safely extracts GitHub token
    - Transfers sub-issues between issues
    - Handles errors and logging
    """
    try:
        # Get the owner and repo name from the repo object
        owner, repo_name = repo.full_name.split('/')
        
        # Get auth token from the repo connection - safely extract token
        token = None
        try:
            # Try different paths to get the token depending on PyGithub version
            if hasattr(repo, '_Github') and hasattr(repo._Github, '_Github__requester'):
                token = repo._Github._Github__requester._Requester__auth.token
            elif hasattr(repo, '_github_object') and hasattr(repo._github_object, '_requester'):
                token = repo._github_object._requester._Requester__auth.token
            elif hasattr(repo, 'get_pulls'):  # Just check if it's a real repo object
                # As a backup, use the environment token
                token = os.environ.get("GITHUB_TOKEN")
        except Exception as e:
            logging.warning(f"Could not extract token from Repository object: {str(e)}")
            # Fall back to environment token
            token = os.environ.get("GITHUB_TOKEN")
            
        if not token:
            logging.error("Failed to get GitHub token for sub-issue transfer")
            return False
            
        headers = get_github_api_headers(token)
        
        # Transfer the sub-issues
        transfer_sub_issues(owner, repo_name, from_issue.number, to_issue.number, headers)
        return True
    except Exception as e:
        logging.error(f"Failed to move sub-issues from #{from_issue.number} to #{to_issue.number}: {str(e)}")
        return False

def handle_duplicate_issue(
    gh: Github,
    repo: Repository,
    issue: Issue,
    duplicate_number: int,
    explanation: str,
    similarity_score: float,
    analysis_details: Optional[dict] = None
) -> bool:
    """Handle a duplicate issue with robust error handling and logging."""
    issue_number = issue.number
    original_issue = None
    
    try:
        # Get the original issue
        try:
            original_issue = repo.get_issue(duplicate_number)
            if not original_issue:
                logging.error(f"Original issue #{duplicate_number} not found")
                return False
        except GithubException as e:
            logging.error(f"Failed to get original issue #{duplicate_number}: {str(e)}")
            return False
        
        # Skip if the issue is already closed
        if issue.state == "closed":
            logging.info(f"Issue #{issue_number} is already closed, skipping")
            return True
        
        # Get issue types and priorities
        # Priority order: Epic > Task > Sub-task
        def get_issue_priority(issue_obj):
            """Get the priority level of an issue based on its type field."""
            # Check issue type field
            if hasattr(issue_obj, 'type') and issue_obj.type:
                type_name = issue_obj.type.name.lower() if hasattr(issue_obj.type, 'name') else ''
                if "epic" in type_name:
                    return 3  # Highest priority
                elif "task" in type_name and "sub" not in type_name:
                    return 2  # Medium priority
                elif "sub-task" in type_name or "subtask" in type_name:
                    return 1  # Lowest priority
            
            # Fallback to title check if type field is not available
            title = issue_obj.title.lower() if hasattr(issue_obj, 'title') else ''
            if "epic" in title or "epic:" in title:
                return 3
            elif "task" in title or "task:" in title:
                return 2
            elif "subtask" in title or "sub-task" in title or "sub task" in title:
                return 1
            
            # Default priority for regular issues
            return 2  # Regular issues default to task level
        
        # Get priorities
        current_issue_priority = get_issue_priority(issue)
        original_issue_priority = get_issue_priority(original_issue)
        
        # Check if we need to swap which issue is considered the duplicate
        if current_issue_priority > original_issue_priority:
            logging.info(f"Issue priority conflict detected: Issue #{issue_number} (priority {current_issue_priority}) " +
                        f"has higher priority than Issue #{duplicate_number} (priority {original_issue_priority})")
            logging.info(f"Swapping duplicate relationship - keeping #{issue_number} and marking #{duplicate_number} as duplicate instead")
            
            # Swap the issues - the original issue becomes the duplicate instead
            swap_comment = f"""🔄 **Duplicate Resolution Notice**

Issue #{duplicate_number} has been identified as a duplicate of this issue.
Since this issue has higher priority (Epic > Task > Sub-task), we're keeping this issue open and closing the duplicate.
"""
            try:
                # Add a comment to the current issue explaining what happened
                issue.create_comment(swap_comment)
                
                # Handle the original issue as the duplicate instead
                return handle_duplicate_issue(gh, repo, original_issue, issue_number, explanation, similarity_score, analysis_details)
            except Exception as e:
                logging.error(f"Failed to swap duplicate relationship: {str(e)}")
                # Continue with normal flow if swapping fails
            
        # Create comment for the duplicate issue
        comment = create_duplicate_comment(
            duplicate_number,
            explanation,
            similarity_score,
            analysis_details
        )
        
        # Add comment to the duplicate issue (the one being closed)
        success = False
        try:
            issue.create_comment(comment)
            logging.info(f"Added duplicate comment to issue #{issue_number}")
            success = True
        except Exception as e:
            logging.error(f"Failed to add comment to issue #{issue_number} via PyGithub: {str(e)}")
            
            # Fallback to REST API if PyGithub fails
            if not success:
                # Safely extract token
                token = None
                try:
                    # Try different paths to get the token depending on PyGithub version
                    if hasattr(gh, '_Github__requester') and hasattr(gh._Github__requester, '_Requester__auth'):
                        token = gh._Github__requester._Requester__auth.token
                    else:
                        # As a backup, use the environment token
                        token = os.environ.get("GITHUB_TOKEN")
                except Exception as e:
                    logging.warning(f"Could not extract token from Github object: {str(e)}")
                    # Fall back to environment token
                    token = os.environ.get("GITHUB_TOKEN")
                
                if not token:
                    logging.error("Failed to get GitHub token for adding comment")
                    return False
                
                headers = get_github_api_headers(token)
                owner, repo_name = repo.full_name.split('/')
                success = add_issue_comment(owner, repo_name, issue_number, comment, headers)
        
        if not success:
            logging.error(f"Failed to add comment to issue #{issue_number} - skipping further actions")
            return False
            
        # Create a cross-reference comment on the original issue
        cross_ref_comment = f"""♻️ **Cross Reference**

Issue #{issue_number} was marked as a duplicate of this issue.

| Metric | Value |
|--------|-------|
| Detection Method | {"Full GPT analysis" if analysis_details else "High-confidence embedding match"} |
| {("Confidence" if analysis_details else "Embedding Score")} | {(analysis_details.get('confidence', similarity_score) if analysis_details else similarity_score):.3f} |
"""
        try:
            original_issue.create_comment(cross_ref_comment)
            logging.info(f"Added cross-reference comment to original issue #{duplicate_number}")
        except Exception as e:
            logging.warning(f"Failed to add cross-reference comment to original issue #{duplicate_number}: {str(e)}")
            # This is not critical, so we continue even if it fails
        
        # Add duplicate label to the issue
        label_success = False
        try:
            issue.add_to_labels("duplicate")
            logging.info(f"Added duplicate label to issue #{issue_number}")
            label_success = True
        except Exception as e:
            logging.error(f"Failed to add label to issue #{issue_number} via PyGithub: {str(e)}")
            
            # Fallback to REST API if PyGithub fails
            if not label_success:
                # Safely extract token (reusing previous token if possible)
                if not token:
                    try:
                        if hasattr(gh, '_Github__requester') and hasattr(gh._Github__requester, '_Requester__auth'):
                            token = gh._Github__requester._Requester__auth.token
                        else:
                            token = os.environ.get("GITHUB_TOKEN")
                    except Exception:
                        token = os.environ.get("GITHUB_TOKEN")
                
                if not token:
                    logging.error("Failed to get GitHub token for adding label")
                    return False
                
                headers = get_github_api_headers(token)
                owner, repo_name = repo.full_name.split('/')
                label_success = add_issue_label(owner, repo_name, issue_number, "duplicate", headers)
        
        if not label_success:
            logging.error(f"Failed to add label to issue #{issue_number}")
            return False
        
        # Move sub-issues if possible
        sub_issues_moved = move_sub_issues(repo, issue, original_issue)
        if not sub_issues_moved:
            logging.warning(f"Failed to move sub-issues from #{issue_number} to #{duplicate_number}")
        
        # Close the issue
        close_success = False
        try:
            issue.edit(state="closed", state_reason="not_planned")
            logging.info(f"Closed issue #{issue_number}")
            close_success = True
        except Exception as e:
            logging.error(f"Failed to close issue #{issue_number} via PyGithub: {str(e)}")
            
            # Fallback to REST API if PyGithub fails
            if not close_success:
                # Safely extract token (reusing previous token if possible)
                if not token:
                    try:
                        if hasattr(gh, '_Github__requester') and hasattr(gh._Github__requester, '_Requester__auth'):
                            token = gh._Github__requester._Requester__auth.token
                        else:
                            token = os.environ.get("GITHUB_TOKEN")
                    except Exception:
                        token = os.environ.get("GITHUB_TOKEN")
                
                if not token:
                    logging.error("Failed to get GitHub token for closing issue")
                    return False
                
                headers = get_github_api_headers(token)
                owner, repo_name = repo.full_name.split('/')
                close_success = close_issue(owner, repo_name, issue_number, headers)
        
        if not close_success:
            logging.error(f"Failed to close issue #{issue_number}")
            return False
            
        logging.info(f"Successfully handled duplicate issue #{issue_number} -> #{duplicate_number}")
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error handling duplicate issue #{issue_number}: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def log_verification_decision(result, issue_number, candidate):
    """Log detailed information about why an issue was not marked as a duplicate."""
    if not result:
        return
    
    # Only log for issues that passed the similarity threshold but weren't marked as duplicates
    candidate_number = candidate.get('number')
    similarity = candidate.get('similarity', 0)
    
    logging.info(f"===== VERIFICATION DECISION FOR #{issue_number} vs #{candidate_number} (similarity: {similarity:.4f}) =====")
    
    if isinstance(result, tuple) and len(result) == 4:
        is_duplicate, confidence, explanation, analysis = result
        
        if not is_duplicate:
            logging.info(f"Decision: NOT a duplicate. Confidence: {confidence:.4f}")
            logging.info(f"Explanation: {explanation}")
            
            # Log key differences if available
            if analysis and 'key_differences' in analysis and analysis['key_differences']:
                logging.info("Key differences that prevented duplicate classification:")
                for i, diff in enumerate(analysis['key_differences'][:5], 1):  # Show up to 5 differences
                    logging.info(f"  {i}. {diff}")
        
        elif is_duplicate and confidence < 0.7:  # Using the threshold from DuplicateDetector
            logging.info(f"Decision: Potential duplicate but confidence too low. Confidence: {confidence:.4f}")
            logging.info(f"Explanation: {explanation}")
    
    logging.info(f"=====")

def check_issue(
    gh: Github,
    repo: Repository,
    detector: DuplicateDetector,
    issue: Issue
) -> Optional[Tuple[int, str, float, dict]]:
    """Check if an issue is a duplicate with better error handling."""
    issue_number = issue.number
    
    # Skip issues that are already closed or labeled as duplicate
    if issue.state == "closed":
        logging.info(f"Skipping closed issue #{issue_number}")
        return None
        
    for label in issue.labels:
        if label.name.lower() == "duplicate":
            logging.info(f"Skipping already labeled duplicate issue #{issue_number}")
            return None
    
    try:
        logging.info(f"Checking issue #{issue_number}: {issue.title}")
            
        # Check for duplicates
        title = issue.title
        body = issue.body or ""  # Handle None body
        updated_at = issue.updated_at.isoformat() if issue.updated_at else ""
        
        # Skip issues with empty title or very short content
        if not title:
            logging.warning(f"Skipping issue #{issue_number} with empty title")
            return None
            
        if len(title) < 5 and len(body) < 20:
            logging.warning(f"Skipping issue #{issue_number} with very short content")
            return None
        
        # Use comprehensive settings for finding the best possible duplicate
        similarity_threshold = 0.80  # Lower threshold to catch more potential duplicates  
        max_candidates = 50  # Higher limit to find the best match
        
        # Check for duplicates with comprehensive settings
        # The DuplicateDetector already has internal rate limiting and retry logic
        result = detector.check_duplicate(
            issue_number, 
            title, 
            body, 
            updated_at,
            similarity_threshold=similarity_threshold,
            max_candidates=max_candidates
        )
        
        # Extract candidate information from the result for detailed logging
        if hasattr(detector, 'last_candidates') and detector.last_candidates:
            logging.info(f"===== POTENTIAL DUPLICATE CANDIDATES FOR #{issue_number} =====")
            for idx, candidate in enumerate(detector.last_candidates[:10], 1):  # Log up to top 10
                candidate_number = candidate.get('number', 'unknown')
                similarity = candidate.get('similarity', 0)
                logging.info(f"Candidate #{idx}: Issue #{candidate_number} - Similarity: {similarity:.4f}")
            
            if len(detector.last_candidates) > 10:
                logging.info(f"...and {len(detector.last_candidates) - 10} more candidates")
            logging.info(f"=====")
            
            # If we found no duplicates but had candidates above threshold, log the verification decision
            if not result or (isinstance(result, tuple) and result[0] <= 0):
                # Get the highest similarity candidate
                best_candidate = detector.last_candidates[0] if detector.last_candidates else None
                if best_candidate and best_candidate.get('similarity', 0) >= similarity_threshold:
                    # Extract GPT verification result from detector if available
                    gpt_result = None
                    if hasattr(detector, '_last_verification_result'):
                        gpt_result = detector._last_verification_result
                    log_verification_decision(gpt_result, issue_number, best_candidate)
        
        # Log related issues separately
        if hasattr(detector, 'last_related_candidates') and detector.last_related_candidates:
            logging.info(f"===== RELATED ISSUES FOR #{issue_number} =====")
            for idx, related in enumerate(detector.last_related_candidates[:10], 1):
                related_number = related.get('number', 'unknown')
                related_similarity = related.get('similarity', 0)
                logging.info(f"Related #{idx}: Issue #{related_number} - Similarity: {related_similarity:.4f}")
            logging.info(f"=====")
        else:
            logging.info(f"No related issues found for #{issue_number}")
        
        if result:
            duplicate_number, similarity, explanation, analysis = result
            logging.info(f"FINAL DECISION: Issue #{issue_number} is a duplicate of #{duplicate_number} (similarity: {similarity:.2f})")
            
            # Log analysis details if available
            if analysis:
                classification = analysis.get('classification', 'UNKNOWN')
                confidence = analysis.get('confidence', 0.0)
                logging.info(f"Analysis: Classification={classification}, Confidence={confidence:.2f}")
                
                # Log key matches and differences (truncated for readability)
                for key in ['core_matches', 'technical_matches', 'scope_matches', 'key_differences']:
                    if key in analysis and analysis[key]:
                        items = analysis[key]
                        sample = items[0] if items else "None"
                        logging.info(f"  {key.replace('_', ' ').title()}: {sample[:100]}... ({len(items)} items)")
            
            return (duplicate_number, explanation, similarity, analysis)
        else:
            logging.info(f"FINAL DECISION: No duplicates found for issue #{issue_number}")
            
            # If we have candidates above threshold but no duplicate was found, explain why
            if hasattr(detector, 'last_candidates') and detector.last_candidates:
                best_candidate = detector.last_candidates[0]
                similarity = best_candidate.get('similarity', 0)
                candidate_number = best_candidate.get('number')
                
                if similarity >= detector.embedding_low_threshold:
                    if similarity >= detector.embedding_high_threshold:
                        # This shouldn't normally happen - high similarity issues should be auto-marked as duplicates
                        logging.info(f"NOTE: Issue #{candidate_number} had very high similarity ({similarity:.4f}) but was not marked as duplicate")
                        logging.info(f"This may be due to a processing error or a manual override in the logic")
                    else:
                        # Medium similarity requires GPT verification
                        logging.info(f"VERIFICATION DETAIL: Issue #{candidate_number} had good similarity ({similarity:.4f}) but GPT verification rejected it")
                        
                        if hasattr(detector, '_last_verification_result') and detector._last_verification_result:
                            is_duplicate, confidence, gpt_explanation, analysis_details = detector._last_verification_result
                            if not is_duplicate:
                                logging.info(f"GPT determined these are distinct issues (confidence: {confidence:.4f})")
                            else:
                                logging.info(f"GPT found a potential duplicate but confidence ({confidence:.4f}) was below threshold ({detector.gpt_confidence_threshold:.4f})")
                            
                            logging.info(f"Explanation: {gpt_explanation}")
                            
                            # Log key differences if available
                            if analysis_details and 'key_differences' in analysis_details and analysis_details['key_differences']:
                                logging.info("Key differences identified by GPT:")
                                for i, diff in enumerate(analysis_details['key_differences'][:3], 1):  # Show up to 3 differences
                                    logging.info(f"  {i}. {diff}")
            
            # Check if we have issues that are related but not duplicates
            if hasattr(detector, 'last_related_candidates') and detector.last_related_candidates:
                # Return a special result to indicate related issues
                related_info = {
                    'related_issues': [{'number': c.get('number'), 'similarity': c.get('similarity', 0.0)} 
                                      for c in detector.last_related_candidates[:5]]
                }
                if related_info['related_issues']:
                    logging.info(f"Found {len(related_info['related_issues'])} related issues for #{issue_number}")
                    return (-1, "Issues are related but not duplicates", 0.0, related_info)
            
            return None
                
    except Exception as e:
        logging.error(f"Unexpected error processing issue #{issue_number}: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def create_related_issue_comment(related_issues, similarity_scores):
    """Create a comment for issues that are related but not duplicates."""
    issues_list = []
    for issue_num, score in zip(related_issues, similarity_scores):
        issues_list.append(f"- #{issue_num} (similarity: {score:.2f})")
    
    comment = f"""## 🔍 **Related Issues**

This issue might be related to:

{chr(10).join(issues_list)}

These issues have similar content but were not classified as duplicates. You may want to check them for context or related solutions.
"""
    return comment

def add_labels_to_issue(github_token, repo_owner, repo_name, issue_number, labels):
    """Add labels to a GitHub issue."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/labels"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.post(url, headers=headers, json=labels)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error adding labels to issue #{issue_number}: {str(e)}")
        if hasattr(e.response, 'text'):
            logging.error(f"Response: {e.response.text}")
        return None

def handle_related_issues(
    gh: Github,
    repo: Repository,
    issue: Issue,
    related_issues_info: Dict
) -> bool:
    """Add cross-reference comments to related issues."""
    issue_number = issue.number
    
    try:
        # Extract related issues with their similarity scores
        related_issues = []
        similarity_scores = []
        
        if 'related_issues' in related_issues_info:
            related_list = related_issues_info['related_issues']
            # Handle both formats: list of dicts or list of numbers
            for related in related_list:
                if isinstance(related, dict):
                    related_issues.append(related.get('number'))
                    similarity_scores.append(related.get('similarity', 0.0))
                else:
                    related_issues.append(related)
                    similarity_scores.append(0.8)  # Default similarity if not provided
        
        if not related_issues:
            logging.info(f"No related issues to reference for issue #{issue_number}")
            return True
        
        logging.info(f"Adding cross-references for {len(related_issues)} related issues to #{issue_number}")
        logging.info(f"Related issues: {related_issues}")
        logging.info(f"Similarity scores: {similarity_scores}")
        
        # Create and add comment to current issue
        comment = create_related_issue_comment(related_issues, similarity_scores)
        
        # Log the exact comment content for debugging
        logging.info(f"Generated comment for issue #{issue_number}:")
        logging.info(f"{comment}")
        
        success = False
        try:
            issue.create_comment(comment)
            logging.info(f"Added related issues comment to #{issue_number}")
            success = True
        except Exception as e:
            logging.error(f"Failed to add related issues comment to #{issue_number}: {str(e)}")
            
            # Fallback to REST API if PyGithub fails
            if not success:
                token = os.environ.get("GITHUB_TOKEN")
                if not token:
                    logging.error("Failed to get GitHub token for adding comment")
                    return False
                
                headers = get_github_api_headers(token)
                owner, repo_name = repo.full_name.split('/')
                success = add_issue_comment(owner, repo_name, issue_number, comment, headers)
        
        # Add cross-reference comments to each related issue
        for i, related_num in enumerate(related_issues):
            try:
                # Get the related issue
                related_issue = repo.get_issue(int(related_num))
                
                # Create cross-reference comment
                related_comment = f"""## 🔍 **Related Issue**

Issue #{issue_number} might be related to this issue (similarity: {similarity_scores[i]:.2f}).
"""
                
                # Log the exact comment content for debugging
                logging.info(f"Generated comment for related issue #{related_num}:")
                logging.info(f"{related_comment}")
                
                try:
                    related_issue.create_comment(related_comment)
                    logging.info(f"Added cross-reference comment to related issue #{related_num}")
                except Exception as e:
                    logging.warning(f"Failed to add cross-reference to related issue #{related_num}: {str(e)}")
                    # Non-critical, continue with other related issues
            except Exception as e:
                logging.warning(f"Could not get or comment on related issue #{related_num}: {str(e)}")
                # Continue with other related issues
        
        return success
    except Exception as e:
        logging.error(f"Error handling related issues for #{issue_number}: {str(e)}")
        return False

def main():
    """Main function with improved error handling and logging."""
    start_time = time.time()
    
    try:
        # Check environment variables
        token = os.environ.get("GITHUB_TOKEN")
        openai_key = os.environ.get("OPENAI_API_KEY")
        repository = os.environ.get("GITHUB_REPOSITORY")
        
        if not all([token, openai_key, repository]):
            missing = []
            if not token: missing.append("GITHUB_TOKEN")
            if not openai_key: missing.append("OPENAI_API_KEY")
            if not repository: missing.append("GITHUB_REPOSITORY")
            logging.critical(f"Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)
        
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize duplicate detector
        try:
            detector = DuplicateDetector()
            logging.info("Initialized duplicate detector")
            
            # Log decision criteria for better understanding of how duplicates are determined
            logging.info(f"===== DUPLICATE DETECTION CRITERIA =====")
            logging.info(f"Embedding similarity threshold: {detector.embedding_low_threshold:.2f}")
            logging.info(f"High confidence embedding threshold: {detector.embedding_high_threshold:.2f}")
            logging.info(f"GPT confidence threshold: {detector.gpt_confidence_threshold:.2f}")
            logging.info(f"Related issue threshold: {RELATED_ISSUE_THRESHOLD:.2f}")
            logging.info(f"Decision logic:")
            logging.info(f"  1. Similarity > {detector.embedding_high_threshold}: Mark as duplicate without GPT verification")
            logging.info(f"  2. Similarity > {detector.embedding_low_threshold}: Use GPT verification")
            logging.info(f"     - If GPT confidence > {detector.gpt_confidence_threshold}: Mark as duplicate")
            logging.info(f"     - Otherwise: Not a duplicate")
            logging.info(f"  3. {detector.embedding_low_threshold} > Similarity > {RELATED_ISSUE_THRESHOLD}: Related issues, add cross-references")
            logging.info(f"  4. Similarity < {RELATED_ISSUE_THRESHOLD}: Not related")
            logging.info(f"=====")
        except Exception as e:
            logging.critical(f"Failed to initialize DuplicateDetector: {str(e)}")
            sys.exit(1)
        
        # Check if we're running for a specific issue
        issue_number = os.environ.get("ISSUE_NUMBER")
        if not issue_number:
            logging.critical("ISSUE_NUMBER environment variable is required. This script expects to be run from a GitHub workflow triggered by an issue event.")
            logging.info("Make sure your workflow passes 'github.event.issue.number' to this script.")
            sys.exit(1)
            
        logging.info(f"Running for specific issue #{issue_number}")
        
        try:
            # Initialize GitHub client
            gh = Github(token)
            repo = gh.get_repo(repository)
            
            # Get the issue
            try:
                issue = repo.get_issue(int(issue_number))
                logging.info(f"Successfully retrieved issue #{issue_number}: {issue.title}")
            except GithubException as e:
                logging.error(f"Failed to get issue #{issue_number}: {str(e)}")
                sys.exit(1)
            
            # Check if the issue is already closed or has duplicate label
            if issue.state == "closed":
                logging.info(f"Issue #{issue_number} is already closed, skipping duplicate detection")
                sys.exit(0)
            
            for label in issue.labels:
                if label.name.lower() == "duplicate":
                    logging.info(f"Issue #{issue_number} is already labeled as duplicate, skipping")
                    sys.exit(0)
            
            # Check for duplicates
            result = check_issue(gh, repo, detector, issue)
            
            if result:
                duplicate_number, explanation, similarity, analysis = result
                
                # Check if this is a duplicate or just related issues
                if duplicate_number > 0:
                    # It's a duplicate - handle it
                    logging.info(f"Found duplicate issue: #{duplicate_number}")
                    success = handle_duplicate_issue(
                        gh, repo, issue, duplicate_number, explanation, similarity, analysis
                    )
                    if success:
                        logging.info(f"Successfully handled duplicate issue #{issue_number}")
                    else:
                        logging.error(f"Failed to handle duplicate issue #{issue_number}")
                        sys.exit(1)
                elif duplicate_number == -1:
                    # It's related issues, not a duplicate (-1 indicates related issues)
                    logging.info(f"Found related issues for #{issue_number}, adding cross-references")
                    success = handle_related_issues(gh, repo, issue, analysis)
                    if success:
                        logging.info(f"Successfully added cross-references for related issues to #{issue_number}")
                    else:
                        logging.warning(f"Failed to add some or all cross-references for related issues")
                
                # Safely log rate limiter stats if available
                try:
                    if hasattr(detector, 'rate_limiter'):
                        if hasattr(detector.rate_limiter, 'get_usage_stats'):
                            stats = detector.rate_limiter.get_usage_stats()
                            logging.info(f"API usage stats: {stats}")
                        else:
                            # Alternative stats reporting if get_usage_stats doesn't exist
                            if hasattr(detector.rate_limiter, 'embedding_api_calls'):
                                embedding_calls = detector.rate_limiter.embedding_api_calls
                                gpt_calls = getattr(detector.rate_limiter, 'gpt_api_calls', 0)
                                logging.info(f"API usage: {embedding_calls} embedding calls, {gpt_calls} GPT calls")
                except Exception as e:
                    logging.warning(f"Could not get API usage stats: {str(e)}")
                    
                sys.exit(0)
            else:
                # No duplicates and no related issues
                logging.info(f"Issue #{issue_number} is not a duplicate")
                sys.exit(0)
                
        except Exception as e:
            logging.critical(f"Error processing issue #{issue_number}: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)
        
        elapsed = time.time() - start_time
        logging.info(f"Duplicate detection completed in {elapsed:.2f} seconds")
            
    except Exception as e:
        logging.critical(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()