# LLM Translator (English to Chinese)

A web-based application that uses LLMs (Large Language Models) to translate and summarize documents from English to Chinese. Supports multiple file formats including PDF, DOCX, TXT, and HTML.

## Features

- Document translation from English to Chinese
- Document summarization
- Support for multiple file formats (PDF, DOCX, TXT, HTML)
- Web-based interface using PyWebIO
- Uses Ollama for local LLM processing
- Metadata extraction for academic papers

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLM_Translator_en2cn.git
cd LLM_Translator_en2cn
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
Follow instructions at [Ollama's official documentation](https://github.com/ollama/ollama)

## Usage

1. Start the application:
```bash
cd src
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

3. Choose the processing mode:
   - EXTRACT: Summarize each passage
   - TRANSLATE: Translate each passage

4. Upload your document and wait for processing

## Configuration

The application uses several configuration files:

- `src/config.py`: Main application configuration
- Environment variables (create a `.env` file):
  ```
  OLLAMA_HOST=http://localhost:11434
  MAX_FILE_SIZE=100M
  DEBUG=True
  ```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/
flake8 src/
mypy src/
```

## Project Structure

```
LLM_Translator_en2cn/
├── src/
│   ├── models/         # LLM and text processing models
│   ├── services/       # Core services (translation, summarization)
│   ├── utils/          # Utility functions
│   ├── app.py         # Main application
│   └── config.py      # Configuration
├── tests/             # Test files
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ollama for providing the local LLM capability
- PyWebIO for the web interface framework
