"""
Models package for handling LLM interactions and text preprocessing.
"""


from .ollama import OllamaService
from .text_processing import TextPreprocessor

__all__ = [
    'OllamaService',
    'TextPreprocessor'
]

# Default models
DEFAULT_MODELS = {
    'summary': 'llama3.2',
    'translation': 'wangshenzhi/gemma2-9b-chinese-chat'
}

# Global settings
MAX_TOKENS = 4700
OLLAMA_URL = "http://localhost:11434/api/generate"