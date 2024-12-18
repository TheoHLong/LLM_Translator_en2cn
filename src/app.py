import os
import sys

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import re
from typing import Optional, Dict, Any
from pywebio.input import *
from pywebio.output import *
from pywebio import config
from pywebio.session import hold, set_env, run_js, info
import logging
from pywebio.platform.aiohttp import start_server
import time
from tqdm import tqdm

# Import our modules
from models.ollama import OllamaService  
from models.text_processing import TextPreprocessor
from services.translation_service import TranslationService, TranslationConfig
from services.summarizer import ContentSummarizer, SummaryConfig
from services.document_parser import DocumentParser
from utils.file_handlers import FileHandler, FileConfig
from utils.text_utils import TextUtils, TextConfig

from services import (
    DEFAULT_TRANSLATION_CONFIG,
    DEFAULT_SUMMARY_CONFIG,
    PROCESSING_MODES
)

@config(title="石头网络")
def start_app():
    """Main application entry point."""
    app = DocumentTranslatorApp()
    app.start()

class DocumentTranslatorApp:
    """Main application class for document translation and summarization."""
    
    def __init__(self):
        """Initialize application components and configurations."""
        # Initialize configurations
        self.translation_config = TranslationConfig(**DEFAULT_TRANSLATION_CONFIG)
        self.summary_config = SummaryConfig(**DEFAULT_SUMMARY_CONFIG)
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
            'in_lang': 'en',
            'out_lang': 'zh',
            'source_lang': 'English',
            'target_lang': 'Chinese',
            'country': 'China'
        }

    def start(self):
        """Start the web application."""
        # Display header information
        self._display_header()

        # Get processing mode from user
        stone_mode = self._get_processing_mode()

        # Handle file upload
        file = self._get_file_upload()
        if not file:
            return

        # Process the file
        self._process_file(file, stone_mode)

    def _display_header(self):
        """Display application header and introduction."""
        put_markdown('## EXTRACT: summary of each passage/总结文章段落')
        put_markdown('## TRANSLATE: Translate each passage/翻译文章段落')

    def _get_processing_mode(self) -> str:
        """Get processing mode from user input."""
        return radio("Choose Reading Mode：",
                    [PROCESSING_MODES['EXTRACT'], PROCESSING_MODES['TRANSLATE']],
                    inline=True,
                    required=True,
                    value=PROCESSING_MODES['EXTRACT'],
                    help_text='',
                    )

    def _get_file_upload(self) -> Optional[Dict]:
        """Handle file upload from user."""
        return file_upload(
            label='Upload File',
            accept=['.pdf', '.docx', '.txt'],
            max_size='100M',
            multiple=False,
            placeholder='.pdf'
        )

    def _process_file(self, file: Dict[str, Any], stone_mode: str):
        """
        Process uploaded file based on its type and mode.
        
        Args:
            file (dict): Uploaded file information
            stone_mode (str): Processing mode selected by user
        """
        try:
            # Save uploaded file
            file_path = self.file_handler.save_file(file)

            if file['mime_type'] == 'application/pdf':
                self._process_pdf(file_path, stone_mode)
            elif file['mime_type'] == 'text/html':
                self._process_html(file_path, stone_mode)
            elif file['mime_type'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                self._process_docx(file_path, stone_mode)
            elif file['mime_type'] == 'text/plain':
                self._process_text(file_path, stone_mode)
            else:
                put_warning("File format error.")
        except Exception as e:
            put_error(f"处理文件时出错: {str(e)}")
        finally:
            # Clean up temporary files
            self.file_handler.clean_up_temp_files()

    def _process_pdf(self, file_path: str, stone_mode: str):
        """Process PDF file."""
        # Try to get paper metadata first
        put_info('主要内容：')
        with put_loading(shape='border', color='info'):
            metadata = self.doc_parser.get_document_metadata(file_path)
            if metadata:
                self.summarizer.process_paper_metadata(metadata, self.translator)

        # Process PDF content
        if info.user_agent.is_pc:
            put_info('PDF内容提取中 ...  (新上传的论文约3分钟，请保持网页连接，一段时间后再回来) / (曾上传过的论文约20秒)')
        else:
            put_info('PDF内容提取中 ...  (新上传的论文约3分钟，注意保持连接，浏览器后台运行可能会自动断开连接) / (曾上传过的论文约20秒)')

        with put_loading(shape='border', color='info'):
            soup, metadata = self.doc_parser.parse_pdf(file_path)

        if stone_mode == PROCESSING_MODES['EXTRACT']:
            put_success('信息提取中 ...  ')
        else:
            put_success('实时翻译中 ...  ')

        self.summarizer.process_html_content(
            soup=soup,
            **self.language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )

        put_success("Done")

    def _process_html(self, file_path: str, stone_mode: str):
        """Process HTML file."""
        with put_loading(shape='border', color='info'):
            soup = self.doc_parser.parse_html_file(file_path)

        self.summarizer.process_html_content(
            soup=soup,
            **self.language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )
        put_success("Done")

    def _process_docx(self, file_path: str, stone_mode: str):
        """Process DOCX file."""
        with put_loading(shape='border', color='info'):
            soup = self.doc_parser.parse_docx(file_path)

        put_success('实时翻译中 ...  ')
        self.summarizer.process_html_content(
            soup=soup,
            **self.language_config,
            stone_mode=stone_mode,
            translator=self.translator
        )
        put_success("Done")

    def _process_text(self, file_path: str, stone_mode: str):
        """Process text file."""
        with put_loading(shape='border', color='info'):
            with open(file_path) as f:
                text = f.read()

        self.summarizer.process_text_content(
            text=text,
            stone_mode=stone_mode,
            translator=self.translator,
            **self.language_config
        )
        put_success("翻译完成")

def main():
    try:
        # Start Ollama server and initialize models
        ollama = OllamaService()
        ollama.start_server()
        time.sleep(1)

        # Initialize models
        if not ollama.are_models_pulled():
            print(f"Pulling required models...")
            ollama.pull_models()

        # Start the server
        start_server(
            start_app,  # Changed from app.start to start_app
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