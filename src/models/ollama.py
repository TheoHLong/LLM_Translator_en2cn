import subprocess
import requests
import json
import time
import platform
import os
from typing import Optional, Union, Dict, Any, List
from tqdm import tqdm
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config

class OllamaService:
    """Handles all Ollama-related operations including translation and summarization."""
    
    def __init__(self):
        self.summary_model = config.ollama.SUMMARY_MODEL
        self.translation_model = config.ollama.TRANSLATION_MODEL
        self.max_tokens = config.ollama.MAX_TOKENS
        self.base_url = config.ollama.BASE_URL
        self.models_pulled = False
        self.process = None

    # Rest of the class implementation remains the same
    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            subprocess.run(["ollama", "--version"], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            return False

    def install_ollama(self) -> bool:
        """Install Ollama based on the operating system."""
        system = platform.system().lower()
        
        try:
            if system == "darwin":  # macOS
                print("Installing Ollama for macOS...")
                install_script = """
                curl -fsSL https://ollama.ai/install.sh -o install.sh
                chmod +x install.sh
                ./install.sh
                rm install.sh
                """
                subprocess.run(install_script, shell=True, check=True)
                return True
                
            elif system == "linux":
                print("Installing Ollama for Linux...")
                install_script = """
                curl -fsSL https://ollama.ai/install.sh -o install.sh
                chmod +x install.sh
                sudo ./install.sh
                rm install.sh
                """
                subprocess.run(install_script, shell=True, check=True)
                return True
                
            elif system == "windows":
                print("For Windows, please install Ollama manually from: https://ollama.ai/download")
                return False
                
            else:
                print(f"Unsupported operating system: {system}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error installing Ollama: {e}")
            return False

    def ensure_ollama_installed(self) -> bool:
        """Ensure Ollama is installed, install if necessary."""
        if not self.is_ollama_installed():
            print("Ollama is not installed. Attempting to install...")
            if self.install_ollama():
                print("Ollama installed successfully!")
                return True
            else:
                print("Failed to install Ollama automatically.")
                return False
        return True

    def are_models_pulled(self) -> bool:
        """Check if required models are pulled."""
        return self.models_pulled

    def pull_models(self) -> None:
        """Pull required models."""
        try:
            for model in [self.summary_model, self.translation_model]:
                print(f"Pulling model: {model}")
                subprocess.run(["ollama", "pull", model], check=True)
                time.sleep(1)
            self.models_pulled = True
        except Exception as e:
            print(f"Error pulling models: {e}")
            raise

    def start_server(self) -> None:
        """Start the Ollama server in the background."""
        if not self.ensure_ollama_installed():
            raise RuntimeError("Ollama is not installed and could not be installed automatically.")
            
        try:
            # Check if server is already running
            try:
                requests.get(f"{self.base_url}/tags")
                print("Ollama server is already running")
                return
            except requests.exceptions.ConnectionError:
                pass

            # Start server if not running
            self.process = subprocess.Popen(["ollama", "serve"])
            time.sleep(2)  # Wait for server to start
            print("Ollama server started successfully")
            
            # Verify server is responding
            try:
                requests.get(f"{self.base_url}/tags")
            except requests.exceptions.ConnectionError:
                raise RuntimeError("Ollama server failed to start properly")
                
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            raise

    def get_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> Union[str, dict]:
        """Generate completion using Ollama API."""
        if model is None:
            model = self.translation_model

        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": f"{system_message}\n{prompt}",
            "stream": False,
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            
            if json_mode:
                return response.json()
            return response.json().get('response', '')
        except Exception as e:
            print(f"Error in get_completion: {e}")
            return None

    def summarize(self, content: str, prompt_sum: str) -> Optional[str]:
        """Summarize content using the summary model."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.summary_model,
            "prompt": prompt_sum + content,
            "stream": False
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                response_text = response.text
                data = json.loads(response_text)
                return data['response']
            else:
                print("Error:", response.status_code)
                return None
        except Exception as e:
            print(f"Error in summarize: {e}")
            return None

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str = "",
        max_tokens: Optional[int] = None
    ) -> str:
        """Translate text using the translation model."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        num_tokens = self._count_tokens(source_text)

        if num_tokens < max_tokens:
            return self._single_chunk_translation(
                source_lang, target_lang, source_text, country
            )
        else:
            return self._multi_chunk_translation(
                source_lang, target_lang, source_text, country, max_tokens
            )

    def _count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """Count the number of tokens in text."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    def _single_chunk_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str
    ) -> str:
        """Handle translation of a single chunk of text."""
        translation_1 = self._initial_translation(source_lang, target_lang, source_text)
        reflection = self._reflect_on_translation(
            source_lang, target_lang, source_text, translation_1, country
        )
        translation_2 = self._improve_translation(
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
        """Handle translation of text that needs to be split into chunks."""
        token_count = self._count_tokens(source_text)
        chunk_size = self._calculate_chunk_size(token_count, max_tokens)
        
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
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

    def _initial_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str
    ) -> str:
        """Generate initial translation."""
        system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        prompt = f"""Provide the {target_lang} translation for this text. Do not provide explanations.
{source_lang}: {source_text}

{target_lang}:"""
        
        return self.get_completion(prompt, system_message)

    def _reflect_on_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation: str,
        country: str
    ) -> str:
        """Generate reflection and suggestions for improving the translation."""
        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}."
        
        if country:
            extra_context = f"The final style should match {target_lang} colloquially spoken in {country}."
        else:
            extra_context = ""

        prompt = f"""Analyze this translation and provide specific improvement suggestions:

Source ({source_lang}):
{source_text}

Translation ({target_lang}):
{translation}

{extra_context}

Focus on improving:
1. Accuracy (fixing mistranslations, omissions)
2. Fluency (grammar, natural flow)
3. Style (maintaining source text style)
4. Terminology (consistency, appropriate idioms)

Provide only the specific suggestions for improvements."""

        return self.get_completion(prompt, system_message)

    def _improve_translation(
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

        return self.get_completion(prompt, system_message)

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