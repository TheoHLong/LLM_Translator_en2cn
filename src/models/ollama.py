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
from pywebio.output import put_warning, put_info

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OllamaService:
    """Handles all Ollama-related operations including translation and summarization."""
    
    def __init__(self):
        self.summary_model = config.ollama.SUMMARY_MODEL
        self.translation_model = config.ollama.TRANSLATION_MODEL
        self.max_tokens = config.ollama.MAX_TOKENS
        self.base_url = config.ollama.BASE_URL
        self.timeout = config.ollama.TIMEOUT
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
