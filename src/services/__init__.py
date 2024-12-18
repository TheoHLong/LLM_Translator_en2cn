"""
Services package for core application functionalities.
"""

from services.document_parser import DocumentParser
from services.translator import TranslationService, TranslationConfig
from services.summarizer import ContentSummarizer, SummaryConfig

__all__ = [
    'DocumentParser',
    'TranslationService',
    'TranslationConfig',
    'ContentSummarizer',
    'SummaryConfig',
]

# Version info
__version__ = '1.0.0'

# Service configurations
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

# Default prompts
DEFAULT_SUMMARY_PROMPT = """
Summarize the passage in one clear and intuitive paragraph, focusing on the central theme 
and essential details without using introductory phrases.
"""

DEFAULT_TRANSLATION_PROMPT = "忠实且自然地把下面内容翻译成中文，并只输出翻译后的文字："

# Processing modes
PROCESSING_MODES = {
    'EXTRACT': '信息提取',
    'TRANSLATE': '自然翻译'
}