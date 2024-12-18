import requests
import json
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag
import time
from dataclasses import dataclass
from models.text_processing import TextPreprocessor
from pywebio.output import (
    put_text, put_loading, put_row, 
    put_processbar, put_markdown,
    put_html, set_processbar
)

@dataclass
class SummaryConfig:
    """Configuration settings for summarization."""
    model_name: str = "llama3.2"
    chunk_size: int = 500
    delay: int = 2
    base_url: str = "http://localhost:11434/api/generate"
    default_prompt: str = """
    Summarize the passage in one clear and intuitive paragraph, focusing on the central theme 
    and essential details without using introductory phrases.
    """

class ContentSummarizer:
    """Handles text and document summarization using Ollama LLM."""

    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize summarizer with configuration.
        
        Args:
            config (SummaryConfig, optional): Configuration settings
        """
        self.config = config or SummaryConfig()
        self.text_processor = TextPreprocessor()

    def summarize_text(self, content: str, prompt: Optional[str] = None) -> Optional[str]:
        """
        Summarize a piece of text using Ollama.
        
        Args:
            content (str): Text to summarize
            prompt (str, optional): Custom summarization prompt
            
        Returns:
            str: Summarized text
        """
        if not prompt:
            prompt = self.config.default_prompt

        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.config.model_name,
            "prompt": prompt + content,
            "stream": False
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                response_text = response.text
                data = json.loads(response_text)
                return data['response']
            else:
                print(f"Error in summarization: {response.status_code}")
                return None
        except Exception as e:
            print(f"Summarization error: {e}")
            return None

    def process_html_content(
        self,
        soup: BeautifulSoup,
        in_lang: str,
        out_lang: str,
        stone_mode: str,
        source_lang: str,
        target_lang: str,
        country: str,
        translator: Any  # Reference to TranslationService
    ) -> None:
        """
        Process and summarize HTML content with translation.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            in_lang (str): Input language
            out_lang (str): Output language
            stone_mode (str): Processing mode ('信息提取' or '自然翻译')
            source_lang (str): Source language for translation
            target_lang (str): Target language for translation
            country (str): Target country for localization
            translator: TranslationService instance
        """
        put_processbar('bar')

        delay_sent = 1
        try:
            contents = soup.find_all([
                'p', 'figure', 'figcaption',
                'li', 'h1', 'h2', 'h3'
            ])

            for i, tags in enumerate(tqdm(contents)):
                # Update progress
                set_processbar('bar', (i + 1) / len(contents))
                self._process_content_element(
                    tags,
                    stone_mode,
                    translator,
                    source_lang,
                    target_lang,
                    country,
                    delay_sent
                )
                
        except Exception as e:
            print(f"Error processing HTML content: {e}")
            raise

    def _process_content_element(
        self,
        element: Tag,
        stone_mode: str,
        translator: Any,
        source_lang: str,
        target_lang: str,
        country: str,
        delay: int
    ) -> None:
        """
        Process individual HTML element.
        
        Args:
            element (Tag): HTML element to process
            stone_mode (str): Processing mode
            translator: TranslationService instance
            source_lang (str): Source language
            target_lang (str): Target language
            country (str): Target country
            delay (int): Delay between processing
        """
        import pywebio.output as output

        if element.name == 'figure':
            output.put_html(element.prettify(), sanitize=True)
            return

        text = element.get_text()
        if len(text) <= 10:
            return

        chunks = self.text_processor.parse_merge(text)
        translated_text_merge = " "
        t_text = None

        for chunk in chunks:
            head_types = ["h1", "h2", "h3"]
            
            if (element.name not in head_types) and (stone_mode == '信息提取') and (len(chunk) > 300):
                # Summarize long non-header content
                s_text = self.summarize_text(chunk)
                time.sleep(0.5)
            else:
                s_text = chunk

            if s_text is None:
                s_text = " "

            # Translate the content
            if stone_mode == '信息提取':
                t_text = translator.fast_translate(s_text)
            elif element.name not in head_types:
                t_text = translator.translate(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_text=s_text,
                    country=country
                )

            if t_text is None:
                t_text = " "
            translated_text_merge += t_text
            time.sleep(0.5)

        # Display the processed content
        self._display_processed_content(
            element,
            translated_text_merge,
            head_types
        )

    def _display_processed_content(
        self,
        element: Tag,
        translated_text: str,
        head_types: List[str]
    ) -> None:
        """
        Display processed content using PyWebIO.
        
        Args:
            element (Tag): Original HTML element
            translated_text (str): Translated/summarized text
            head_types (list): List of header element types
        """
        import pywebio.output as output
        from pywebio.session import info

        indent = '  ' * 3

        if element.name in head_types:
            output.put_html(element.prettify(), sanitize=True)
            output.put_markdown('### ' + translated_text)
        elif element.name == 'figcaption':
            output.put_markdown('#### ' + translated_text)
        else:
            if info.user_agent.is_pc:
                output.put_row([
                    output.put_html(element.prettify(), sanitize=True),
                    None,
                    output.put_text(indent + translated_text)
                ])
            else:
                output.put_html(element.prettify(), sanitize=True)
                output.put_text(indent + translated_text)

    def process_text_content(
        self,
        text: str,
        stone_mode: str,
        translator: Any,
        source_lang: str,
        target_lang: str,
        country: str
    ) -> None:
        """
        Process and summarize plain text content.
        
        Args:
            text (str): Text to process
            stone_mode (str): Processing mode
            translator: TranslationService instance
            source_lang (str): Source language
            target_lang (str): Target language
            country (str): Target country
        """
        import pywebio.output as output
        from pywebio.session import info

        # Initialize progress bar
        output.put_processbar('bar')
        
        # Determine chunk size based on mode
        chunk_size = 500 if stone_mode == '信息提取' else 600
        chunks = self.text_processor.parse_merge_txt(text, chunk_size)
        
        for i, chunk in enumerate(tqdm(chunks)):
            # Update progress bar
            output.set_processbar('bar', (i + 1) / len(chunks))
            
            # Process chunk based on mode and length
            if (stone_mode == '信息提取') and (len(chunk) > 200):
                try:
                    processed_text = self.summarize_text(chunk)
                except:
                    processed_text = chunk
            else:
                processed_text = chunk

            # Translate if chunk is long enough
            if len(processed_text) >= 5:
                translated_text = translator.fast_translate(processed_text)
                
                # Display results
                if info.user_agent.is_pc:
                    output.put_row([
                        output.put_text(processed_text),
                        None,
                        output.put_text(translated_text)
                    ])
                else:
                    output.put_text(processed_text)
                    output.put_text(translated_text)

                time.sleep(self.config.delay)

    def process_paper_metadata(
        self,
        metadata: Dict[str, Any],
        translator: Any
    ) -> None:
        """
        Process and display academic paper metadata.
        
        Args:
            metadata (dict): Paper metadata
            translator: TranslationService instance
        """
        import pywebio.output as output

        output.put_info('主要内容：')
        
        with output.put_loading(shape='border', color='info'):
            output.put_markdown('### ' + metadata['title'])
            
            if metadata.get('tldr'):
                text = metadata['tldr']
                chunks = self.text_processor.parse_merge(text)
                for chunk in chunks:
                    if len(chunk) >= 5:
                        translated_text = translator.fast_translate(chunk)
                        output.put_row([
                            output.put_text(chunk),
                            None,
                            output.put_text(translated_text)
                        ])
            
            if metadata.get('doi'):
                output.put_markdown('### doi: ' + metadata['doi'])