import re
import string
import unidecode
from typing import List, Dict, Any, Optional, Tuple
from langdetect import detect
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass
import tiktoken
from tqdm import tqdm

@dataclass
class TextConfig:
    """Configuration settings for text processing."""
    max_tokens_per_chunk: int = 4700
    min_chars_per_chunk: int = 5
    default_encoding: str = 'utf-8'
    summary_min_length: int = 200
    title_max_words: int = 20
    title_max_chars: int = 200
    title_min_chars: int = 6

class TextUtils:
    """Utility class for text processing and manipulation."""

    def __init__(self, config: Optional[TextConfig] = None):
        """
        Initialize TextUtils with configuration.
        
        Args:
            config (TextConfig, optional): Configuration settings
        """
        self.config = config or TextConfig()
        self.char_parsing_state = {
            'INIT_X': 0,
            'INIT_D': 1,
            'INSIDE_WORD': 2
        }

    def parse_merge(self, texts: str, n: int = 4700) -> List[str]:
        """
        Split text into chunks by periods.
        
        Args:
            texts (str): Text to split
            n (int): Maximum chunk size
            
        Returns:
            List[str]: List of text chunks
        """
        sentences = []
        sentence = ""
        for i in texts.split("."):
            if len(sentence) + len(i) > n:
                sentences.append(sentence)
                sentence = ""
            sentence += i + ' '
        sentences.append(sentence)
        return sentences

    def parse_merge_txt(self, texts: str, n: int = 4700) -> List[str]:
        """
        Split text by newlines into chunks.
        
        Args:
            texts (str): Text to split
            n (int): Maximum chunk size
            
        Returns:
            List[str]: List of text chunks
        """
        sentences = []
        sentence = ""
        for i in texts.split("\n"):
            if len(sentence) + len(i) > n:
                sentences.append(sentence)
                sentence = ""
            sentence += i + ' '
        sentences.append(sentence)
        return sentences

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def sanitize_title(self, title: str) -> str:
        """
        Sanitize and format title text.
        
        Args:
            title (str): Title to sanitize
            
        Returns:
            str: Sanitized title
        """
        # Limit length
        words = title.split(' ')
        title = ' '.join(words[0:self.config.title_max_words])
        if len(title) > self.config.title_max_chars:
            title = title[0:self.config.title_max_chars]

        # Handle special characters
        try:
            title = unidecode.unidecode(title.encode('utf-8').decode('utf-8'))
        except UnicodeDecodeError:
            pass

        # Clean up formatting
        title = re.sub(r',', ' ', title)
        title = re.sub(r': ', ' - ', title)
        title = re.sub(r'\.pdf(\.pdf)*$', '', title)
        title = re.sub(r'[ \t][ \t]*', ' ', title)

        # Keep only valid characters
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return ''.join(c for c in title if c in valid_chars)

    def extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from HTML content.
        
        Args:
            soup (BeautifulSoup): HTML content
            
        Returns:
            str: Extracted text
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines and normalize space
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return self.clean_text(text)

    def is_junk_line(self, line: str) -> bool:
        """
        Check if a line should be considered junk.
        
        Args:
            line (str): Line to check
            
        Returns:
            bool: True if line is junk
        """
        too_small = len(line.strip()) < self.config.title_min_chars
        
        is_placeholder = bool(re.search(
            r'^[0-9 \t-]+(abstract|introduction)?\s+$|^(abstract|unknown|title|untitled):?$',
            line.strip().lower()
        ))
        
        is_copyright = bool(re.search(
            r'paper\s+title|technical\s+report|proceedings|preprint|to\s+appear|submission|'
            r'(integrated|international).*conference|transactions\s+on|symposium\s+on|downloaded\s+from\s+http',
            line.lower()
        ))

        stripped_ascii = ''.join([c for c in line.strip() if c in string.ascii_letters])
        ascii_length = len(stripped_ascii)
        stripped_chars = re.sub(r'[ \t\n]', '', line.strip())
        chars_length = len(stripped_chars)
        is_serial_number = ascii_length < chars_length / 2

        return too_small or is_placeholder or is_copyright or is_serial_number

    def count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count tokens in text.
        
        Args:
            text (str): Text to count tokens in
            encoding_name (str): Name of tokenizer encoding
            
        Returns:
            int: Number of tokens
        """
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    def calculate_chunk_size(self, token_count: int, token_limit: int) -> int:
        """
        Calculate optimal chunk size for token splitting.
        
        Args:
            token_count (int): Total number of tokens
            token_limit (int): Maximum tokens per chunk
            
        Returns:
            int: Optimal chunk size
        """
        if token_count <= token_limit:
            return token_count

        num_chunks = (token_count + token_limit - 1) // token_limit
        chunk_size = token_count // num_chunks

        remaining_tokens = token_count % token_limit
        if remaining_tokens > 0:
            chunk_size += remaining_tokens // num_chunks

        return chunk_size

    def detect_language(self, text: str) -> str:
        """
        Detect text language.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code
        """
        try:
            return detect(text)
        except:
            return 'en'

    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of sentences
        """
        # Split on sentence endings
        sentence_endings = r'[.!?]+'
        potential_sentences = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        
        # Recombine sentence parts with their endings
        for i in range(0, len(potential_sentences), 2):
            current = potential_sentences[i]
            if i + 1 < len(potential_sentences):
                current += potential_sentences[i + 1]
            if current.strip():
                sentences.append(current.strip())
                
        return sentences

    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistical information about text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Text statistics
        """
        words = text.split()
        sentences = self.extract_sentences(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'language': self.detect_language(text)
        }

    def process_text_segments(
        self,
        segments: List[str],
        processor: Any,
        show_progress: bool = True
    ) -> List[str]:
        """
        Process list of text segments with given processor.
        
        Args:
            segments (List[str]): Text segments to process
            processor: Function to process each segment
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[str]: Processed segments
        """
        results = []
        iterator = tqdm(segments) if show_progress else segments
        
        for segment in iterator:
            try:
                processed = processor(segment)
                results.append(processed)
            except Exception as e:
                print(f"Error processing segment: {e}")
                results.append(segment)
                
        return results

    def split_text_for_processing(
        self,
        text: str,
        max_chars: Optional[int] = None,
        preserve_format: bool = False
    ) -> List[str]:
        """
        Split text into processable chunks.
        
        Args:
            text (str): Text to split
            max_chars (int, optional): Maximum characters per chunk
            preserve_format (bool): Whether to preserve text formatting
            
        Returns:
            List[str]: Text chunks
        """
        if max_chars is None:
            max_chars = self.config.max_tokens_per_chunk
            
        if preserve_format:
            return self.parse_merge_txt(text, max_chars)
        else:
            return self.parse_merge(text, max_chars)

    def clean_html_content(
        self,
        soup: BeautifulSoup,
        remove_elements: Optional[List[str]] = None
    ) -> BeautifulSoup:
        """
        Clean HTML content by removing unwanted elements.
        
        Args:
            soup (BeautifulSoup): HTML content
            remove_elements (List[str], optional): Elements to remove
            
        Returns:
            BeautifulSoup: Cleaned HTML
        """
        if remove_elements is None:
            remove_elements = [
                'script', 'style', 'meta', 'link', 'header',
                'footer', 'nav', 'aside', 'iframe'
            ]
            
        for element in remove_elements:
            for tag in soup.find_all(element):
                tag.decompose()
                
        return soup