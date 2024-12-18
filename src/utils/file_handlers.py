import os
import glob
from typing import Dict, Any, Optional, Union, BinaryIO
import codecs
import validators
from validators import ValidationFailure
from dataclasses import dataclass
import PyPDF2
import mammoth
from bs4 import BeautifulSoup
import srt
import re
from deep_translator import GoogleTranslator
from langdetect import detect
from models.text_processing import TextPreprocessor

@dataclass
class FileConfig:
    """Configuration settings for file handling."""
    temp_directory: str = 'temp'
    allowed_extensions: tuple = ('.pdf', '.docx', '.txt', '.srt', '.html')
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    mime_types = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'html': 'text/html',
        'srt': 'application/x-subrip'
    }

class FileHandler:
    """Handles file operations including saving, validation, and processing."""

    def __init__(self, config: Optional[FileConfig] = None):
        """
        Initialize FileHandler with configuration.
        
        Args:
            config (FileConfig, optional): Configuration settings
        """
        self.config = config or FileConfig()
        self.text_processor = TextPreprocessor()

    def setup_temp_directory(self) -> str:
        """
        Create temporary directory if it doesn't exist.
        
        Returns:
            str: Path to temporary directory
        """
        dir_temp = os.path.join(os.getcwd(), self.config.temp_directory)
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp)
        return dir_temp

    def save_file(self, file: Dict[str, Any]) -> str:
        """
        Save uploaded file to temporary directory.
        
        Args:
            file (dict): File data with filename and content
            
        Returns:
            str: Path to saved file
        """
        dir_temp = self.setup_temp_directory()

        # Get file extension and sanitize filename
        basename, file_extension = os.path.splitext(file['filename'])
        try:
            basename = self.text_processor.detect_and_translate_to_english(basename)
        except:
            pass
            
        file_path = os.path.join(dir_temp, basename + file_extension)

        with open(file_path, "wb") as saved_file:
            saved_file.write(file['content'])
        return file_path

    def process_subtitles(
        self,
        file_path: str,
        translator: Any,
        stone_mode: str,
        source_lang: str,
        target_lang: str,
        country: str
    ) -> None:
        """
        Process and translate subtitle file.
        
        Args:
            file_path (str): Path to subtitle file
            translator: Translation service instance
            stone_mode (str): Processing mode
            source_lang (str): Source language
            target_lang (str): Target language
            country (str): Target country
        """
        from pywebio.output import put_text, put_loading, put_row
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-16') as f:
                srt_content = f.read()

        subtitle_generator = srt.parse(srt_content)
        subtitles = list(subtitle_generator)

        for subtitle in subtitles:
            content = subtitle.content
            
            # Skip if content is too short or empty
            if len(content.strip()) < 2:
                continue
                
            # Translate content
            with put_loading(shape='border', color='info'):
                if stone_mode == '信息提取':
                    translated_text = translator.fast_translate(content)
                else:
                    translated_text = translator.translate(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        source_text=content,
                        country=country
                    )
                
                time_str = f"{subtitle.start} --> {subtitle.end}"
                put_row([
                    put_text(f"{subtitle.index}\n{time_str}\n{content}"),
                    None,
                    put_text(translated_text)
                ])

        # Clean up
        os.remove(file_path)

    def validate_file(self, file: Dict[str, Any]) -> bool:
        """
        Validate uploaded file.
        
        Args:
            file (dict): File data with filename and content
            
        Returns:
            bool: True if file is valid
        """
        # Check file extension
        _, extension = os.path.splitext(file['filename'])
        if extension.lower() not in self.config.allowed_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        # Check file size
        if len(file['content']) > self.config.max_file_size:
            raise ValueError(f"File size exceeds maximum limit of {self.config.max_file_size // 1024 // 1024}MB")

        # Check mime type
        if file['mime_type'] not in self.config.mime_types.values():
            raise ValueError(f"Unsupported MIME type: {file['mime_type']}")

        return True

    def get_file_type(self, file_path: str) -> str:
        """
        Get file type from extension.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            str: File type
        """
        _, extension = os.path.splitext(file_path)
        extension = extension.lower().lstrip('.')
        return self.config.mime_types.get(extension, 'application/octet-stream')

    def is_url(self, url: str) -> bool:
        """
        Check if string is valid URL.
        
        Args:
            url (str): String to check
            
        Returns:
            bool: True if valid URL
        """
        result = validators.url(url)
        return not isinstance(result, ValidationFailure)

    def txt2text(self, file_path: str, chunk_size: int = 4700) -> list:
        """
        Convert text file to list of text chunks.
        
        Args:
            file_path (str): Path to text file
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            list: List of text chunks
        """
        with open(file_path, "r") as file:
            lines = file.readlines()

        text = ' '.join(lines)
        sentences = self.text_processor.parse_merge(text, chunk_size)
        return sentences

    def read_file_content(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Read file content with proper encoding handling.
        
        Args:
            file_path (str): Path to file
            encoding (str): Initial encoding to try
            
        Returns:
            str: File content
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-16', 'latin-1', 'cp1252']
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Unable to decode file {file_path} with any known encoding")

    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            file_path (str): Path to document
            
        Returns:
            dict: Document metadata
        """
        metadata = {}
        file_type = self.get_file_type(file_path)

        if file_type == 'application/pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'num_pages': len(pdf_reader.pages)
                    })

        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            try:
                result = mammoth.convert_to_html(file_path)
                soup = BeautifulSoup(result.value, 'html.parser')
                metadata['title'] = soup.find('title').text if soup.find('title') else ''
                metadata['content_length'] = len(result.value)
            except:
                pass

        return metadata

    def clean_up_temp_files(self, pattern: str = "*") -> None:
        """
        Clean up temporary files matching pattern.
        
        Args:
            pattern (str): File pattern to match
        """
        dir_temp = self.setup_temp_directory()
        pattern_path = os.path.join(dir_temp, pattern)
        
        for file_path in glob.glob(pattern_path):
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    @staticmethod
    def create_error_message(error: Exception) -> str:
        """
        Create user-friendly error message.
        
        Args:
            error (Exception): The error that occurred
            
        Returns:
            str: Formatted error message
        """
        error_type = type(error).__name__
        error_map = {
            'ValueError': '无效的文件或参数',
            'FileNotFoundError': '文件未找到',
            'PermissionError': '没有足够的权限',
            'UnicodeDecodeError': '文件编码错误',
            'OSError': '系统错误'
        }
        return error_map.get(error_type, '未知错误') + f": {str(error)}"