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

    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize translation service with configuration.
        
        Args:
            config (TranslationConfig, optional): Configuration settings
        """
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

            # Main translation with chunking and refinement
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
            try:
                # Fallback to Google Translate
                translator = GoogleTranslator(source='auto', target='zh-CN')
                result = translator.translate(source_text)
                return result if result else source_text
            except Exception as e:
                print(f"Fallback translation error: {e}")
                return source_text

    def summarize(self, text: str, language: str = "English") -> str:
        """
        Summarize text while maintaining key information.
        
        Args:
            text (str): Text to summarize
            language (str): Language of the text
            
        Returns:
            str: Summarized text
        """
        try:
            summary_prompt = (
                "Summarize the following passage in one clear and intuitive paragraph, "
                "focusing on the central theme and essential details without using introductory phrases: "
            )
            
            preprocessed_text = self.text_processor.preprocess(text)
            summarized_text = self._get_completion(
                prompt=summary_prompt + preprocessed_text,
                system_message=f"You are an expert at summarizing {language} text.",
                model=self.config.summary_model
            )
            
            if summarized_text:
                return self.text_processor.postprocess(summarized_text)
            return text

        except Exception as e:
            print(f"Summarization error: {e}")
            return text

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
            # Single chunk translation with three-step refinement
            return self._single_chunk_translation(
                source_lang, target_lang, source_text, country
            )
        else:
            # Multi-chunk translation with optimal chunk sizing
            chunks = self._split_into_chunks(source_text, max_tokens)
            translations = self._translate_chunks(
                source_lang, target_lang, chunks, country
            )
            return self._join_translations(translations)

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
            # Add context from previous translation if available
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

    def _join_translations(self, translations: List[str]) -> str:
        """Join translated chunks while ensuring smooth transitions."""
        return " ".join(translations)

    def _single_chunk_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
        context: str = ""
    ) -> str:
        """Execute three-step translation process for single chunk."""
        # Step 1: Initial translation
        translation_1 = self._get_initial_translation(
            source_lang, target_lang, source_text, context
        )
        if not translation_1:
            raise Exception("Initial translation failed")
        
        # Step 2: Get expert reflection
        reflection = self._get_translation_reflection(
            source_lang, target_lang, source_text, translation_1, country
        )
        if not reflection:
            return translation_1
        
        # Step 3: Generate improved translation
        translation_2 = self._get_improved_translation(
            source_lang, target_lang, source_text, translation_1, reflection
        )
        return translation_2 if translation_2 else translation_1

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

    def _get_initial_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        context: str = ""
    ) -> str:
        """Generate initial translation with context awareness."""
        system_message = f"""You are an expert linguist and professional translator, 
specializing in {source_lang} to {target_lang} translation.
Your goal is to provide accurate, natural, and high-quality translations."""
        
        prompt = f"""{context}Please translate this {source_lang} text into {target_lang}.
Focus on accuracy and natural expression in the target language.

{source_lang} text:
{source_text}

{target_lang} translation:"""

        result = self._get_completion(prompt, system_message)
        if not result:
            # Fallback to Google Translate
            try:
                translator = GoogleTranslator(source='auto', target='zh-CN')
                result = translator.translate(source_text)
            except:
                result = source_text
        return result

    def _get_translation_reflection(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation: str,
        country: str
    ) -> str:
        """Generate detailed reflection on translation quality."""
        system_message = f"""You are an expert linguist and translation reviewer, 
specializing in {source_lang} to {target_lang} translation quality assessment.
Your task is to provide detailed, constructive feedback for improving translations."""
        
        country_context = f"The translation should match {target_lang} as used in {country}. " if country else ""
        
        prompt = f"""Analyze this translation from {source_lang} to {target_lang} and provide specific improvement suggestions.
{country_context}

Source text:
{source_text}

Current translation:
{translation}

Focus your analysis on:
1. Accuracy - Are there any mistranslations or omissions?
2. Natural expression - Does it sound natural in {target_lang}?
3. Cultural appropriateness - Is it culturally appropriate for {country if country else target_lang} readers?
4. Terminology consistency
5. Style and tone

Provide specific suggestions for improvement:"""

        return self._get_completion(prompt, system_message) or ""

    def _get_improved_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation: str,
        reflection: str
    ) -> str:
        """Generate improved translation based on expert reflection."""
        system_message = f"""You are an expert translator and editor, 
specializing in {source_lang} to {target_lang} translation refinement.
Your task is to improve the translation based on expert feedback 
while maintaining accuracy and natural expression."""
        
        prompt = f"""Improve this translation based on the expert feedback provided.

Source text ({source_lang}):
{source_text}

Current translation:
{translation}

Expert feedback and suggestions:
{reflection}

Provide the improved translation:"""

        result = self._get_completion(prompt, system_message)
        return result if result else translation

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
        try:
            lang_type = self.detect_language(text)
            if lang_type != 'en':
                return self.translate(
                    source_lang=lang_type,
                    target_lang='English',
                    source_text=text,
                    country='United States'
                )
            return text
        except Exception as e:
            print(f"Error translating to English: {e}")
            return text

    def detect_and_translate_to_chinese(self, text: str) -> str:
        """
        Detect language and translate to Chinese if needed.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text in Chinese
        """
        try:
            lang_type = self.detect_language(text)
            if lang_type != 'zh-cn':
                return self.translate(
                    source_lang=lang_type,
                    target_lang='Chinese',
                    source_text=text,
                    country='China'
                )
            return text
        except Exception as e:
            print(f"Error translating to Chinese: {e}")
            return text