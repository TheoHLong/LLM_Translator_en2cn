import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from typing import Optional, Dict, Any
from pywebio.input import *
from pywebio.output import *
from pywebio import config, start_server
from pywebio.session import hold, set_env, run_js, info
from pywebio.platform import config as pywebio_config  # Fixed this import
import time

# Import our modules
from config import config as app_config  # Renamed this import
from models.ollama import OllamaService  
from models.text_processing import TextPreprocessor
from services.translation_service import TranslationService, TranslationConfig
from services.summarizer import ContentSummarizer, SummaryConfig
from services.document_parser import DocumentParser
from utils.file_handlers import FileHandler, FileConfig
from utils.text_utils import TextUtils, TextConfig
from services import PROCESSING_MODES
from utils.vtt_parser import extract_text_from_vtt

@pywebio_config(theme="minty", title="Document Translation & Summary System / 文档翻译与总结系统")
def start_app():
    """Main application entry point."""
    app = DocumentTranslatorApp()
    app.start()

class DocumentTranslatorApp:
    """Main application class for document translation and summarization."""
    
    def __init__(self):
        """Initialize application components and configurations."""
        # Initialize configurations
        self.translation_config = TranslationConfig(
            max_tokens=app_config.ollama.MAX_TOKENS,
            model_name=app_config.ollama.TRANSLATION_MODEL,
            fallback_engine=app_config.translation.FALLBACK_ENGINE
        )
        
        self.summary_config = SummaryConfig(
            model_name=app_config.ollama.SUMMARY_MODEL,
            chunk_size=app_config.text.DEFAULT_CHUNK_SIZE,
            delay=app_config.summary.PROCESSING_DELAY
        )
        self.file_config = FileConfig()
        self.text_config = TextConfig()

        # Initialize services
        self.ollama_service = OllamaService()
        self.translator = TranslationService(self.translation_config)
        self.summarizer = ContentSummarizer(self.summary_config)
        self.doc_parser = DocumentParser()
        self.file_handler = FileHandler(self.file_config)
        self.text_utils = TextUtils(self.text_config)

        # Configuration for languages
        self.language_config = {
            'source_lang': 'English',
            'target_lang': 'Chinese',
            'country': 'China'
        }
        
        self.html_language_config = {
            'in_lang': 'en',
            'out_lang': 'zh',
            **self.language_config
        }

    def start(self):
        """Start the web application."""
        # Add custom styles
        put_html('''
            <style>
            .container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 2em;
                padding: 2em;
                background: #f8f9fa;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section {
                background: #ffffff;
                padding: 1.5em;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1.5em;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1em;
                margin: 1em 0;
            }
            .feature-item {
                padding: 1em;
                background: #ffffff;
                border-radius: 8px;
                text-align: center;
            }
            .status {
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }
            .text-black {
                color: #000000 !important;
            }
            .subtitle {
                color: #000000;
                font-size: 1.1em;
                margin: 0.5em 0;
            }
            .success { background-color: #d4edda; color: #000000; }
            .error { background-color: #f8d7da; color: #000000; }
            .warning { background-color: #fff3cd; color: #000000; }
            .processing { background-color: #e2e3e5; color: #000000; }

            /* Add these new styles for making output text black */
            .markdown-body {
                color: #000000 !important;
            }
            .markdown-body p,
            .markdown-body h1,
            .markdown-body h2,
            .markdown-body h3,
            .markdown-body h4,
            .markdown-body h5,
            .markdown-body h6,
            .markdown-body span,
            .markdown-body div {
                color: #000000 !important;
            }
            /* Make output text black */
            .webio-text-output {
                color: #000000 !important;
            }
            pre, code {
                color: #000000 !important;
            }
            /* Force output content to be black */
            [data-scope="content"] * {
                color: #000000 !important;
            }
            </style>
        ''')

        # Header Section
        # put_html('<div class="header">')
        put_html('<h1 class="text-black">Document Translation & Summary System using LLM<br></h1>')
        put_html('''
                 

                 
        ''')
        
        # Feature Grid
        put_html('''
            <div class="feature-grid">
                <div class="feature-item">
                    <h3 class="text-black">📝 Content Extraction<br>内容提取</h3>
                </div>
                <div class="feature-item">
                    <h3 class="text-black">🔄 Real-time Translation<br>实时翻译</h3>
                </div>
            </div>
        ''')

        # Mode Selection Section
        put_html('<div class="section">')
        stone_mode = self._get_processing_mode()
        put_html('</div>')

        # File Upload Section
        put_html('<div class="section">')
        put_markdown('### File Upload / 上传文件')
        file = self._get_file_upload()
        put_html('</div>')

        if not file:
            return

        # Process the file
        self._process_file(file, stone_mode)
        
        put_html('</div>')  # Close container

    def _display_status(self, message: str, status: str = 'info'):
        """Display status message with appropriate styling."""
        style = f'status {status}'
        put_html(f'<div class="{style}">{message}</div>')

    def _get_processing_mode(self) -> str:
        """Get processing mode from user input."""
        return radio(
            "Select Processing Mode / 请选择处理模式：",
            [
                {'label': '📑 Content Extraction / 内容提取', 'value': PROCESSING_MODES['EXTRACT']},
                {'label': '🔄 Document Translation / 文档翻译', 'value': PROCESSING_MODES['TRANSLATE']}
            ],
            inline=True,
            required=True,
            value=PROCESSING_MODES['EXTRACT']
        )

    def _get_file_upload(self) -> Optional[Dict]:
        """Handle file upload from user."""
        return file_upload(
            label='Select File to Process / 选择要处理的文件',
            accept=['.pdf', '.docx', '.txt', '.vtt'],
            max_size='100M',
            multiple=False,
            placeholder='Support PDF, Word, TXT / 支持 PDF、Word、TXT 格式',
            help_text='Maximum file size: 100MB / 最大支持100MB的文件上传'
        )

    def _process_file(self, file: Dict[str, Any], stone_mode: str):
        """Process uploaded file based on its type and mode."""
        try:
            # Save uploaded file
            file_path = self.file_handler.save_file(file)
            self._display_status('File upload successful! / 文件上传成功！', 'success')

            if file['mime_type'] == 'application/pdf':
                self._process_pdf(file_path, stone_mode)
            elif file['mime_type'] == 'text/html':
                self._process_html(file_path, stone_mode)
            elif file['mime_type'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                self._process_docx(file_path, stone_mode)
            elif file['mime_type'] == 'text/plain':
                self._process_text(file_path, stone_mode)
            elif file['mime_type'] == 'text/vtt':
                self._process_vtt(file_path, stone_mode)
            else:
                self._display_status('Unsupported file format / 不支持的文件格式', 'error')

        except Exception as e:
            self._display_status(f"Error processing file / 处理文件时出错: {str(e)}", 'error')
        finally:
            self.file_handler.clean_up_temp_files()

    def _process_pdf(self, file_path: str, stone_mode: str):
        """Process PDF file."""
        # Process PDF content
        status_msg = 'Processing PDF content... (First time takes about 3 minutes) / 正在处理PDF内容...(首次处理约需3分钟)'
        self._display_status(status_msg, 'processing')

        put_loading()
        soup, _ = self.doc_parser.parse_pdf(file_path)  # Ignore metadata

        proc_msg = 'Extracting information... / 正在提取信息...' if stone_mode == PROCESSING_MODES['EXTRACT'] else 'Translating content... / 正在翻译内容...'
        self._display_status(proc_msg, 'processing')

        self.summarizer.process_html_content(
            soup=soup,
            **self.html_language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )

        self._display_status('Processing complete! / 处理完成！', 'success')

    def _process_html(self, file_path: str, stone_mode: str):
        """Process HTML file."""
        self._display_status('Processing HTML content... / 正在处理HTML内容...', 'processing')
        
        put_loading()
        soup = self.doc_parser.parse_html_file(file_path)

        self.summarizer.process_html_content(
            soup=soup,
            **self.html_language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )
        
        self._display_status('Processing complete! / 处理完成！', 'success')

    def _process_docx(self, file_path: str, stone_mode: str):
        """Process DOCX file."""
        self._display_status('Processing Word document... / 正在处理Word文档...', 'processing')
        
        put_loading()
        soup = self.doc_parser.parse_docx(file_path)

        self._display_status('Translating content... / 正在翻译内容...', 'processing')
        self.summarizer.process_html_content(
            soup=soup,
            **self.html_language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )
        
        self._display_status('Processing complete! / 处理完成！', 'success')

    def _process_vtt(self, file_path: str, stone_mode: str):
        """Process VTT file like a text file after extracting content."""
        self._display_status('Processing VTT subtitles... / 正在处理VTT字幕...', 'processing')
        
        put_loading()
        # Extract text from VTT
        text = extract_text_from_vtt(file_path)
        
        # Process just like a text file
        self.summarizer.process_text_content(
            text=text,
            stone_mode=stone_mode,
            translator=self.translator,
            **self.language_config
        )
        
        self._display_status('Processing complete! / 处理完成！', 'success')

    def _process_text(self, file_path: str, stone_mode: str):
        """Process text file."""
        self._display_status('Processing text file... / 正在处理文本文件...', 'processing')
        
        put_loading()
        with open(file_path) as f:
            text = f.read()

        self.summarizer.process_text_content(
            text=text,
            stone_mode=stone_mode,
            translator=self.translator,
            **self.language_config
        )
        
        self._display_status('Processing complete! / 处理完成！', 'success')

def main():
    try:
        # Start Ollama server and initialize models
        ollama = OllamaService()
        ollama.start_server()
        time.sleep(1)

        # Initialize models
        if not ollama.are_models_pulled():
            print("Pulling required models...")
            ollama.pull_models()

        # Start the server
        start_server(
            start_app,
            port=8501,
            debug=True,
            cdn=False,
            reconnect_timeout=5*60,
            websocket_settings=dict(max_msg_size=0),
            remote_access=True
        )
    except Exception as e:
        print(f"Application startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()