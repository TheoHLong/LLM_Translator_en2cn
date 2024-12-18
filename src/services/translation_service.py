from typing import Optional, List, Dict, Any
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
    max_tokens: int = 4700
    model_name: str = "wangshenzhi/gemma2-9b-chinese-chat"
    base_url: str = "http://localhost:11434/api/generate"
    fallback_engine: str = "google"

class TranslationService:
    """Handles all translation operations using Ollama LLM and fallback services."""

    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize translation service with configuration.
        
        Args:
            config (TranslationConfig, optional): Configuration settings
        """
        self.config = config or TranslationConfig()
        self.text_processor = TextPreprocessor()

    # Need to add this method
    def fast_translate(self, text: str) -> str:
        """Quick translation fallback using Google Translate."""
        try:
            translator = GoogleTranslator(source='auto', target='zh-CN')
            return translator.translate(text)
        except Exception as e:
            print(f"Fast translation error: {e}")
            return text

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str = "",
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Translate text using Ollama with multi-step refinement.
        
        Args:
            source_lang (str): Source language
            target_lang (str): Target language
            source_text (str): Text to translate
            country (str): Target country for localization
            max_tokens (int, optional): Maximum tokens per request
            
        Returns:
            str: Translated text
        """
        try:
            preprocessed_text = self.text_processor.preprocess(source_text)
            translated_text = self._translate_with_ollama(
                source_lang,
                target_lang,
                preprocessed_text,
                country,
                max_tokens or self.config.max_tokens
            )
            return self.text_processor.postprocess(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")

    def _translate_with_ollama(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        max_tokens: int
    ) -> str:
        """
        Translate text using Ollama with chunking if needed.
        
        Args:
            source_lang (str): Source language
            target_lang (str): Target language
            source_text (str): Text to translate
            country (str): Target country
            max_tokens (int): Maximum tokens per request
            
        Returns:
            str: Translated text
        """
        num_tokens = self._count_tokens(source_text)

        if num_tokens < max_tokens:
            return self._single_chunk_translation(
                source_lang, target_lang, source_text, country
            )
        else:
            return self._multi_chunk_translation(
                source_lang, target_lang, source_text, country, max_tokens
            )

    def _single_chunk_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str
    ) -> str:
        """Execute three-step translation process for single chunk."""
        # Initial translation
        translation_1 = self._get_initial_translation(source_lang, target_lang, source_text)
        
        # Get expert reflection
        reflection = self._get_translation_reflection(
            source_lang, target_lang, source_text, translation_1, country
        )


        
        
        # Generate improved translation
        translation_2 = self._get_improved_translation(
            source_lang, target_lang, source_text, translation_1, reflection
        )
        
        return translation_2

    def _multi_chunk_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        max_tokens: int
    ) -> str:
        """Handle translation of text split into multiple chunks."""
        token_size = self._calculate_chunk_size(
            self._count_tokens(source_text),
            max_tokens
        )

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        chunks = splitter.split_text(source_text)
        translated_chunks = []

        for chunk in chunks:
            translation = self._single_chunk_translation(
                source_lang, target_lang, chunk, country
            )
            translated_chunks.append(translation)

        return "".join(translated_chunks)

    def _get_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant."
    ) -> Optional[str]:
        """Get completion from Ollama API."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.config.model_name,
            "prompt": f"{system_message}\n{prompt}",
            "stream": False,
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            print(f"Completion error: {e}")
            return None

    def _get_initial_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str
    ) -> str:
        """Generate initial translation."""
        system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        
        prompt = f"""This is a {source_lang} to {target_lang} translation. Provide only the {target_lang} translation:
{source_lang}: {source_text}

{target_lang}:"""

        translation = self._get_completion(prompt, system_message)
        return translation if translation else self.fast_translate(source_text)

    def _get_translation_reflection(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation: str,
        country: str
    ) -> str:
        """Generate reflection on initial translation."""
        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}."
        
        country_context = f"The final style should match {target_lang} colloquially spoken in {country}. " if country else ""
        
        prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
{country_specific}

Source Text:
{source_text}

Translation:
{translation}

Please analyze the translation and provide specific suggestions for improving:
1. Accuracy (fixing mistranslations, omissions)
2. Fluency (grammar, natural flow)
3. Style (maintaining the source text style and cultural context)
4. Terminology (consistency and appropriate idioms)

Provide only the specific suggestions for improvements."""

        reflection = self._get_completion(prompt, system_message)
        return reflection if reflection else ""

    def _get_improved_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation: str,
        reflection: str
    ) -> str:
        """Generate improved translation based on reflection."""
        system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
        
        prompt = f"""Improve this translation based on the expert suggestions:

Source ({source_lang}):
{source_text}

Initial Translation:
{translation}

Expert Suggestions:
{reflection}

Provide only the improved translation, no explanations."""

        improved = self._get_completion(prompt, system_message)
        return improved if improved else translation

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

        remaining_tokens = token_count % token_limit
        if remaining_tokens > 0:
            chunk_size += remaining_tokens // num_chunks

        return chunk_size

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Detected language code
        """
        try:
            return detect(text)
        except:
            return 'en'

    def detect_and_translate_to_english(self, text: str) -> str:
        """
        Detect language and translate to English if needed.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text in English
        """
        lang_type = self.detect_language(text)
        if lang_type != 'en':
            translator = GoogleTranslator(source='auto', target='en')
            text = translator.translate(text)
        return text

    def detect_and_translate_to_chinese(self, text: str) -> str:
        """
        Detect language and translate to Chinese if needed.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text in Chinese
        """
        lang_des = "zh-CN"
        lang_type = self.detect_language(text)
        if lang_type != lang_des:
            translator = GoogleTranslator(source='auto', target='zh-CN')
            text = translator.translate(text)
        return text