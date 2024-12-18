"""
Utilities package for helper functions and common operations.
"""

from utils.file_handlers import FileHandler, FileConfig
from utils.text_utils import TextUtils, TextConfig

__all__ = [
    'FileHandler',
    'FileConfig',
    'TextUtils',
    'TextConfig',
]

# Version info
__version__ = '1.0.0'

# File configurations
DEFAULT_FILE_CONFIG = {
    'temp_directory': 'temp',
    'allowed_extensions': ('.pdf', '.docx', '.txt', '.srt', '.html'),
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'mime_types': {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'html': 'text/html',
        'srt': 'application/x-subrip'
    }
}

# Text configurations
DEFAULT_TEXT_CONFIG = {
    'max_tokens_per_chunk': 4700,
    'min_chars_per_chunk': 5,
    'default_encoding': 'utf-8',
    'summary_min_length': 200,
    'title_max_words': 20,
    'title_max_chars': 200,
    'title_min_chars': 6
}

# Common utility functions
def get_version():
    """Get the current version of the utilities package."""
    return __version__

def is_development():
    """Check if running in development environment."""
    import os
    return os.getenv('ENVIRONMENT') == 'development'