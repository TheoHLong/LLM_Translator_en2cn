from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import json
from deep_translator import GoogleTranslator
from langdetect import detect
import time
from tqdm import tqdm
from models.text_processing import TextPreprocessor

@dataclass
class TranslationConfig:
    """Configuration settings for translation."""
    max_tokens: int = 4700  # Maximum tokens per chunk
    model_name: str = "wangshenzhi/gemma2-9b-chinese-chat"  # Model for translation
    summary_model: str = "llama3.2"  # Model for summarization
    base_url: str = "http://localhost:11434/api/generate"
    fallback_engine: str = "google"
    delay_seconds: float = 0.5  # Delay between API calls

class TranslationService:
    """Handles all translation operations using Ollama LLM and fallback services."""

    # Language code mappings for fallback
    LANG_CODES = {
        'english': 'en',
        'chinese': 'zh',
        'zh-cn': 'zh',
        'en': 'en',
        'zh': 'zh'
    }

    def __init__(self, config: Optional[TranslationConfig] = None):
        """Initialize translation service with configuration."""
        self.config = config or TranslationConfig()
        self.text_processor = TextPreprocessor()

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str = "",
        max_tokens: Optional[int] = None,
        preserve_formatting: bool = True
    ) -> str:
        """
        Translate text using Ollama with multi-step refinement.
        
        Args:
            source_lang (str): Source language
            target_lang (str): Target language 
            source_text (str): Text to translate
            country (str): Target country for localization
            max_tokens (int, optional): Maximum tokens per request
            preserve_formatting (bool): Whether to preserve special characters/formatting
            
        Returns:
            str: Translated text
        """
        if not source_text or len(source_text.strip()) == 0:
            return ""

        try:
            # Preprocess to handle special characters and formatting
            if preserve_formatting:
                preprocessed_text = self.text_processor.preprocess(source_text)
            else:
                preprocessed_text = source_text

            # Use Ollama translation
            translated_text = self._translate_with_ollama(
                source_lang,
                target_lang,
                preprocessed_text,
                country,
                max_tokens or self.config.max_tokens
            )

            # Postprocess to restore special characters and formatting
            if preserve_formatting:
                translated_text = self.text_processor.postprocess(translated_text)

            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            return source_text

    def summarize(self, text: str, language: str = "English", output_lang: str = "Chinese") -> str:
        """
        Process text: Original -> language Summary -> language Rephrase -> output_lang Translation.
        All steps use Ollama for high quality.
        
        Args:
            text (str): Text to summarize
            language (str): Source language of the text
            output_lang (str): Desired output language
            
        Returns:
            str: Processed and translated text
        """
        try:
            # 1. Get summary in original language
            summary_prompt = (
                f"Summarize the following {language} passage in one clear and intuitive paragraph, "
                f"focusing on the central theme and essential details. Respond in {language}. "
                "Do not use any introductory phrases: "
            )
            
            preprocessed_text = self.text_processor.preprocess(text)
            language_summary = self._get_completion(
                prompt=summary_prompt + preprocessed_text,
                system_message=f"You are an expert at summarizing {language} text.",
                model=self.config.summary_model
            )
            
            if not language_summary:
                return text

            # 2. Rephrase summary in original language
            rephrase_prompt = (
                f"Please rephrase the following {language} summary to be more clear, natural, "
                f"and engaging while maintaining all key information. Respond in {language}."
                "Do not use any introductory phrases: "
                f"{language_summary}"
            )
            
            language_rephrased = self._get_completion(
                prompt=rephrase_prompt,
                system_message=f"You are an expert at clear and engaging {language} writing.",
                model=self.config.summary_model
            )

            # Use the rephrased version if successful, otherwise use original summary
            final_text = language_rephrased if language_rephrased else language_summary

            # 3. Translate to output language using Ollama if languages differ
            if language.lower() != output_lang.lower():
                translated_text = self._translate_with_ollama(
                    source_lang=language,
                    target_lang=output_lang,
                    source_text=final_text,
                    country="",  # Not needed for this case
                    max_tokens=self.config.max_tokens
                )
                return self.text_processor.postprocess(translated_text)
            
            return self.text_processor.postprocess(final_text)

        except Exception as e:
            print(f"Processing error: {e}")
            if language.lower() != output_lang.lower():
                # Try direct translation as fallback
                try:
                    return self._translate_with_ollama(
                        source_lang=language,
                        target_lang=output_lang,
                        source_text=text,
                        country="",
                        max_tokens=self.config.max_tokens
                    )
                except:
                    return text
            return text

    def _translate_with_ollama(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        max_tokens: int
    ) -> str:
        """Translate text using Ollama with chunking if needed."""
        num_tokens = self._count_tokens(source_text)
        
        if num_tokens < max_tokens:
            # Single chunk translation
            return self._single_chunk_translation(
                source_lang, target_lang, source_text, country
            )
        else:
            # Multi-chunk translation
            chunks = self._split_into_chunks(source_text, max_tokens)
            translations = self._translate_chunks(
                source_lang, target_lang, chunks, country
            )
            return "".join(translations)

    def _get_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        model: Optional[str] = None
    ) -> Optional[str]:
        """Get completion from Ollama API."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model or self.config.model_name,
            "prompt": f"{system_message}\n{prompt}",
            "stream": False,
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                data=json.dumps(data),
                timeout=30
            )
            response.raise_for_status()
            result = response.json().get('response', '')
            return result if result and len(result.strip()) > 0 else None
        except Exception as e:
            print(f"Completion error: {e}")
            return None

    def _split_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Split text into optimally sized chunks."""
        token_size = self._calculate_chunk_size(
            self._count_tokens(text),
            max_tokens
        )

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=50,  # Small overlap to maintain context
        )

        return splitter.split_text(text)

    def _translate_chunks(
        self,
        source_lang: str,
        target_lang: str,
        chunks: List[str],
        country: str
    ) -> List[str]:
        """Translate multiple chunks while maintaining consistency."""
        translated_chunks = []
        context = ""  # Store previous translations for context

        for i, chunk in enumerate(chunks):
            # Add context from previous translation
            context_prompt = f"Previous translation context: {context}\n\n" if context else ""
            
            translation = self._single_chunk_translation(
                source_lang,
                target_lang,
                chunk,
                country,
                context_prompt
            )
            
            translated_chunks.append(translation)
            
            # Update context with the latest translation
            context = translation if i == len(chunks) - 2 else ""
            
            # Add delay to avoid rate limiting
            time.sleep(self.config.delay_seconds)

        return translated_chunks

    def _single_chunk_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        context: str = ""
    ) -> str:
        """Execute translation process for single chunk."""
        system_message = f"You are an expert linguist and professional translator, specializing in {source_lang} to {target_lang} translation."
        
        # Add country-specific context if provided
        if country:
            system_message += f" Your translations should be appropriate for {target_lang} speakers in {country}."

        prompt = f"""{context}Translate this text from {source_lang} to {target_lang}.
Focus on accuracy and natural expression.

Source text:
{source_text}

Translation:"""

        result = self._get_completion(prompt, system_message)
        return result if result else source_text

    def _count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """Count tokens in text using specified encoding."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    def _calculate_chunk_size(self, token_count: int, token_limit: int) -> int:
        """Calculate optimal chunk size for text splitting."""
        if token_count <= token_limit:
            return token_count

        num_chunks = (token_count + token_limit - 1) // token_limit
        chunk_size = token_count // num_chunks

        # Distribute remaining tokens
        remaining_tokens = token_count % token_limit
        if remaining_tokens > 0:
            chunk_size += remaining_tokens // num_chunks

        return chunk_size

    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            return detect(text)
        except:
            return 'en'