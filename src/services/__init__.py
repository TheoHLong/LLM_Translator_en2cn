"""
Services package for document translation and processing.
"""

from .document_parser import DocumentParser
from .translation_service import TranslationService, TranslationConfig
from .summarizer import ContentSummarizer, SummaryConfig

__all__ = [
    'DocumentParser',
    'TranslationService',
    'TranslationConfig',
    'ContentSummarizer',
    'SummaryConfig'
]

# Default configurations
DEFAULT_TRANSLATION_CONFIG = {
    'max_tokens': 4700,
    'model_name': 'wangshenzhi/gemma2-9b-chinese-chat',
    'fallback_engine': 'google'
}

DEFAULT_SUMMARY_CONFIG = {
    'model_name': 'llama3.2',
    'chunk_size': 500,
    'delay': 2
}

# Processing modes
PROCESSING_MODES = {
    'EXTRACT': '信息提取',
    'TRANSLATE': '自然翻译'
}