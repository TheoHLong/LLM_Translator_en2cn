import os
import string
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    DEBUG: bool = False
    TESTING: bool = False
    ENV: str = os.getenv('ENVIRONMENT', 'production')

@dataclass
class OllamaConfig:
    """Configuration settings for Ollama LLM service."""
    BASE_URL: str = "http://localhost:11434/api/generate"
    SUMMARY_MODEL: str = "gemma3:4b"
    TRANSLATION_MODEL: str = "qwen2.5"
    MAX_TOKENS: int = 4700
    TIMEOUT: int = 300  # Increased to 5 minutes
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 1  # seconds between retries

@dataclass
class FileConfig:
    """Configuration settings for file handling."""
    UPLOAD_FOLDER: str = "temp"
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: tuple = ('.pdf', '.docx', '.txt', '.html')
    MIME_TYPES: Dict[str, str] = field(default_factory=lambda: {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'html': 'text/html',
        'vtt': 'text/vtt',
    })

    def __post_init__(self):
        # Create upload folder if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), self.UPLOAD_FOLDER), exist_ok=True)

@dataclass
class TextConfig:
    """Configuration settings for text processing."""
    MAX_TOKENS_PER_CHUNK: int = 4700
    MIN_CHARS_PER_CHUNK: int = 5
    DEFAULT_ENCODING: str = 'utf-8'
    SUMMARY_MIN_LENGTH: int = 200
    DEFAULT_CHUNK_SIZE: int = 500  # Added this to match usage in services
    TITLE_MAX_WORDS: int = 20
    TITLE_MAX_CHARS: int = 200
    TITLE_MIN_CHARS: int = 6
    HTML_ELEMENTS_TO_REMOVE: tuple = (
        'script', 'style', 'meta', 'link', 
        'header', 'footer', 'nav', 'aside', 'iframe'
    )

@dataclass
class TranslationConfig:
    """Configuration settings for translation service."""
    SOURCE_LANG: str = 'en'
    TARGET_LANG: str = 'zh'
    SOURCE_LANG_FULL: str = 'English'
    TARGET_LANG_FULL: str = 'Chinese'
    TARGET_COUNTRY: str = 'China'
    FALLBACK_ENGINE: str = 'google'
    DEFAULT_TRANSLATION_PROMPT: str = "忠实且自然地把下面内容翻译成中文，并只输出翻译后的文字："
    REFLECTION_PROMPT: str = """Analyze this translation and provide specific improvement suggestions focusing on:
1. Accuracy (fixing mistranslations, omissions)
2. Fluency (grammar, natural flow)
3. Style (maintaining tone and cultural context)
4. Terminology (consistency, idioms)"""

@dataclass
class SummaryConfig:
    """Configuration settings for text summarization."""
    MIN_LENGTH_FOR_SUMMARY: int = 300
    DEFAULT_CHUNK_SIZE: int = 500
    PROCESSING_DELAY: int = 2
    DEFAULT_SUMMARY_PROMPT: str = """
    Summarize the passage in one clear and intuitive paragraph, focusing on the central theme 
    and essential details without using introductory phrases.
    """

@dataclass
class WebConfig:
    """Configuration settings for web interface."""
    PORT: int = 80
    HOST: str = '0.0.0.0'
    TITLE: str = "石头网络"
    DEBUG: bool = True
    CDN_ENABLED: bool = False
    RECONNECT_TIMEOUT: int = 5 * 60
    MAX_MSG_SIZE: int = 0
    REMOTE_ACCESS: bool = True

@dataclass
class DocumentParserConfig:
    """Configuration settings for document parsing."""
    # Semantic Scholar API settings
    S2_API_KEY: str = ''  # Removed sensitive key
    S2_API_URL: str = 'https://partner.semanticscholar.org/graph/v1'
    
    # WebDriver settings
    WEBDRIVER_OPTIONS: Tuple[str, ...] = (
        '--headless',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--window-size=1920,1080'
    )
    WEBDRIVER_EXCLUDED_SWITCHES: Tuple[str, ...] = (
        'enable-logging',
        'enable-automation'
    )
    WEBDRIVER_IMPLICIT_WAIT: int = 2
    WEBDRIVER_PAGE_LOAD_WAIT: int = 2

    # HTML cleaning settings
    HTML_ELEMENTS_TO_REMOVE: Tuple[str, ...] = (
        'header', 'meta', 'script', '[document]',
        'noscript', 'head', 'input'
    )
    
    # PDF conversion settings
    PDF_TO_HTML_URL: str = 'https://papertohtml.org/'
    PDF_UPLOAD_WAIT: int = 3
    PDF_CONVERSION_WAIT: int = 90
    PDF_ELEMENTS_TO_REMOVE: Tuple[str, ...] = (
        'header', 'title', 'meta', 'div.app__signup-form',
        'div.text-center', 'div.paper__head div',
        'footer.app__footer', 'script', 'form',
        '.page__description', '.home__icon',
        'ul.paper__meta', 'div.paper__toc.card'
    )

@dataclass
class TextProcessingConfig:
    """Configuration settings for text processing."""
    # Character limits
    MIN_CHARS: int = 6
    MAX_WORDS: int = 20
    MAX_CHARS: int = MAX_WORDS * 10
    
    # Processing settings
    TOLERANCE: float = 1e-06
    DEFAULT_CHUNK_SIZE: int = 4700
    
    # PDF title extraction settings
    TITLE_REGEX_PATTERNS: Tuple[str, ...] = (
        r'^[0-9 \t-]+(abstract|introduction)?\s+$',
        r'^(abstract|unknown|title|untitled):?$',
        r'paper\s+title|technical\s+report|proceedings|preprint|to\s+appear|submission',
        r'(integrated|international).*conference|transactions\s+on|symposium\s+on|downloaded\s+from\s+http'
    )
    
    # PDF text extraction settings
    PDF_ENCODING: str = 'utf-8'
    VALID_CHARS: str = "-_.() %s%s" % (string.ascii_letters, string.digits)

    # Title cleaning patterns
    TITLE_CLEAN_PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        r',': ' ',
        r': ': ' - ',
        r'\.pdf(\.pdf)*$': '',
        r'[ \t][ \t]*': ' '
    })

class Config:
    """Main configuration class that combines all config components."""
    def __init__(self):
        self.base = BaseConfig()
        self.ollama = OllamaConfig()
        self.file = FileConfig()
        self.text = TextConfig()
        self.translation = TranslationConfig()
        self.summary = SummaryConfig()
        self.web = WebConfig()
        self.document_parser = DocumentParserConfig()
        self.text_processing = TextProcessingConfig()
        self.PROCESSING_MODES = {
            'EXTRACT': '信息提取',
            'TRANSLATE': '自然翻译'
        }

    def get_web_server_config(self) -> Dict[str, Any]:
        """Get configuration for web server setup."""
        return {
            'port': self.web.PORT,
            'debug': self.web.DEBUG,
            'cdn': self.web.CDN_ENABLED,
            'reconnect_timeout': self.web.RECONNECT_TIMEOUT,
            'websocket_settings': dict(max_msg_size=self.web.MAX_MSG_SIZE),
            'remote_access': self.web.REMOTE_ACCESS,
        }

    @property
    def upload_path(self) -> Path:
        """Get the full path to the upload directory."""
        return Path(os.getcwd()) / self.file.UPLOAD_FOLDER

# Create global config instance
config = Config()