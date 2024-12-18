"""
Models package for handling LLM interactions and text preprocessing.
"""

from models.ollama import OllamaService
from models.text_processing import TextPreprocessor
from models.response_handler import ResponseHandler

__all__ = [
    'OllamaService',
    'TextPreprocessor',
    'ResponseHandler',
]

# Version info
__version__ = '1.0.0'

# Default configuration
DEFAULT_MODELS = {
    'summary': 'llama3.2',
    'translation': 'wangshenzhi/gemma2-9b-chinese-chat'
}

# Global settings
MAX_TOKENS = 4700
OLLAMA_URL = "http://localhost:11434/api/generate"