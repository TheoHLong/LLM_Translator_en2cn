import os
from typing import Dict, Any
from dataclasses import dataclass
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
    SUMMARY_MODEL: str = "llama3.2"
    TRANSLATION_MODEL: str = "wangshenzhi/gemma2-9b-chinese-chat"
    MAX_TOKENS: int = 4700
    TIMEOUT: int = 300  # seconds
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 1  # seconds

@dataclass
class FileConfig:
    """Configuration settings for file handling."""
    UPLOAD_FOLDER: str = "temp"
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: tuple = ('.pdf', '.docx', '.txt', '.html')
    MIME_TYPES: Dict[str, str] = None

    def __post_init__(self):
        self.MIME_TYPES = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'doc': 'application/msword',
            'txt': 'text/plain',
            'html': 'text/html',
        }
        
        # Create upload folder if it doesn't exist
        os.makedirs(os.path.join(os.getcwd(), self.UPLOAD_FOLDER), exist_ok=True)

@dataclass
class TextConfig:
    """Configuration settings for text processing."""
    MAX_TOKENS_PER_CHUNK: int = 4700
    MIN_CHARS_PER_CHUNK: int = 5
    DEFAULT_ENCODING: str = 'utf-8'
    SUMMARY_MIN_LENGTH: int = 200
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
    
    # Translation prompts
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
    
    # Summary prompts
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
    
    # Google Analytics
    GA_TRACKING_ID: str = "G-SMSZ6XKZNN"
    GA_JS_URL: str = "https://www.googletagmanager.com/gtag/js"
    
    # Custom JavaScript
    CUSTOM_JS: str = """
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-SMSZ6XKZNN');
    """

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
        
        # Processing modes
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

    def get_google_analytics_code(self) -> str:
        """Get Google Analytics initialization code."""
        return f"""
        <script async src="{self.web.GA_JS_URL}?id={self.web.GA_TRACKING_ID}"></script>
        <script>
            {self.web.CUSTOM_JS}
        </script>
        """

    @property
    def upload_path(self) -> Path:
        """Get the full path to the upload directory."""
        return Path(os.getcwd()) / self.file.UPLOAD_FOLDER

def load_config() -> Config:
    """Load and return configuration based on environment."""
    return Config()

# Create global config instance
config = load_config()