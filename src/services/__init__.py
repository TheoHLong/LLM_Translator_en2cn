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

# Processing modes
PROCESSING_MODES = {
    'EXTRACT': '信息提取',
    'TRANSLATE': '自然翻译'
}