import os
from typing import Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import validators
from validators import ValidationFailure
import mammoth
import codecs
from semanticscholar import SemanticScholar

class DocumentParser:
    """Handles all document parsing operations including PDF, DOCX, and URL content."""

    def __init__(self, semantic_scholar_key: Optional[str] = None):
        """
        Initialize DocumentParser with optional Semantic Scholar API key.
        
        Args:
            semantic_scholar_key (str, optional): API key for Semantic Scholar
        """
        self.s2_api_key = semantic_scholar_key or 'o2MSPjUJ9pxdefSoGbe570y6aCcR6rI1adw0Y6G5'
        self.s2_api_url = 'https://partner.semanticscholar.org/graph/v1'
        self.sch = SemanticScholar(api_key=self.s2_api_key, api_url=self.s2_api_url)

    def get_document_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Public method to get document metadata from a PDF file.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            Optional[Dict[str, Any]]: Document metadata if found, None otherwise
        """
        return self._get_paper_metadata(file_path)

    def init_selenium_driver(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver with appropriate options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        return webdriver.Chrome(options=options)

    def quit_driver_and_cleanup(self, driver: webdriver.Chrome) -> None:
        """Properly close WebDriver and clean up processes."""
        driver.quit()
        try:
            pid = True
            while pid:
                pid = os.waitpid(-1, os.WNOHANG)
                try:
                    if pid[0] == 0:
                        pid = False
                except:
                    pass
        except ChildProcessError:
            pass

    def save_uploaded_file(self, file: Dict[str, Any], dir_name: str = 'temp') -> str:
        """
        Save uploaded file to temporary directory.
        
        Args:
            file (dict): File data dictionary with 'filename' and 'content'
            dir_name (str): Directory name for temporary files
        
        Returns:
            str: Path to saved file
        """
        dir_temp = os.path.join(os.getcwd(), dir_name)
        if not os.path.exists(dir_temp):
            os.makedirs(dir_temp)

        basename, file_extension = os.path.splitext(file['filename'])
        file_path = os.path.join(dir_temp, basename + file_extension)

        with open(file_path, "wb") as saved_file:
            saved_file.write(file['content'])
        return file_path

    def is_valid_url(self, url_string: str) -> bool:
        """Check if string is a valid URL."""
        result = validators.url(url_string)
        return not isinstance(result, ValidationFailure)

    def parse_url_content(self, url: str) -> BeautifulSoup:
        """
        Parse content from URL into BeautifulSoup object.
        
        Args:
            url (str): URL to parse
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        driver = None
        try:
            driver = self.init_selenium_driver()
            driver.implicitly_wait(2)
            driver.get(url)
            page = driver.page_source
            soup = BeautifulSoup(page, 'html.parser')

            # Remove unnecessary elements
            removals_names = [
                'header', 'meta', 'script', '[document]',
                'noscript', 'head', 'input',
            ]
            for removals_name in removals_names:
                removal = soup.select(removals_name)
                for item in removal:
                    item.decompose()

            return soup
        finally:
            if driver:
                self.quit_driver_and_cleanup(driver)

    def parse_docx(self, file_path: str) -> BeautifulSoup:
        """
        Parse DOCX file into BeautifulSoup object.
        
        Args:
            file_path (str): Path to DOCX file
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        try:
            result = mammoth.convert_to_html(file_path)
            html = result.value
            return BeautifulSoup(html, 'html.parser')
        finally:
            os.remove(file_path)

    def parse_pdf(self, file_path: str) -> Tuple[BeautifulSoup, Optional[Dict[str, Any]]]:
        """
        Parse PDF file into BeautifulSoup object and extract metadata.
        
        Args: 
            file_path (str): Path to PDF file
            
        Returns:
            Tuple[BeautifulSoup, dict]: Tuple of (parsed content, metadata)
        """
        driver = None
        try:
            # Try to get paper metadata first
            metadata = self._get_paper_metadata(file_path)
            
            # Convert PDF to HTML
            url_pdf = 'https://papertohtml.org/'
            driver = self.init_selenium_driver()
            
            driver.get(url_pdf)
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.home__upload-input'))
            )
            file_area = driver.find_element(By.CSS_SELECTOR, '.home__upload-input')
            file_area.send_keys(file_path)

            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button'))
            )
            driver.find_element(By.CSS_SELECTOR, 'button').click()

            # Wait for conversion
            try:
                WebDriverWait(driver, 90).until(EC.url_changes(url_pdf))
                
                # Additional wait for page load
                for _ in range(20):
                    page_state = driver.execute_script('return document.readyState;')
                    if page_state == 'complete':
                        break
                    time.sleep(.5)
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # Clean up the HTML
                removals_names = [
                    'header', 'title', 'meta', 'div.app__signup-form',
                    'div.text-center', 'div.paper__head div',
                    'footer.app__footer', 'script', 'form',
                    '.page__description', '.home__icon'
                ]

                for removals_name in removals_names:
                    removal = soup.select(removals_name)
                    for item in removal:
                        item.decompose()

                # Handle navigation section
                nav = soup.find("nav")
                if nav:
                    for a in nav.findAll('a'):
                        a.replaceWithChildren()

                # Add necessary meta tags
                metatag_a = soup.new_tag("meta", charset="utf-8")
                metatag_b = soup.new_tag('meta')
                metatag_b.attrs['content'] = "width=device-width,initial-scale=1,shrink-to-fit=no"
                metatag_b.attrs['name'] = "viewport"
                soup.head.append(metatag_a)
                soup.head.append(metatag_b)

                return soup, metadata
                

            except TimeoutException:
                raise Exception("PDF conversion timeout")
            except Exception as e:
                raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            if driver:
                self.quit_driver_and_cleanup(driver)
            try:
                os.remove(file_path)
            except:
                pass


    def parse_html_file(self, file_path: str) -> BeautifulSoup:
        """
        Parse HTML file into BeautifulSoup object.
        
        Args:
            file_path (str): Path to HTML file
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        with codecs.open(file_path, "r", "utf-8") as html_file:
            return BeautifulSoup(html_file.read(), 'html.parser')

    def _get_paper_metadata(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract paper metadata using PDF title and Semantic Scholar.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: Paper metadata if found
        """
        try:
            # Get PDF title using the title extractor from text_processing
            from models.text_processing import TextPreprocessor
            text_processor = TextPreprocessor()
            paper_title = text_processor.get_pdf_title(pdf_path)

            # Search Semantic Scholar
            results = self.sch.search_paper(paper_title)
            if not results:
                return None

            paper_doi = results[0]['externalIds'].get('DOI')
            if not paper_doi:
                return None

            paper = self.sch.paper(paper_doi)
            return {
                'title': paper['title'],
                'tldr': paper.get('tldr', {}).get('text'),
                'doi': paper_doi,
                'abstract': paper.get('abstract'),
                'year': paper.get('year'),
                'authors': [author['name'] for author in paper.get('authors', [])]
            }

        except Exception as e:
            print(f"Error getting paper metadata: {e}")
            return None

    def parse_document(self, file: Dict[str, Any]) -> Tuple[BeautifulSoup, Optional[Dict[str, Any]]]:
        """
        Main entry point for parsing any document type.
        
        Args:
            file (dict): File data dictionary with 'mime_type' and content
            
        Returns:
            Tuple[BeautifulSoup, dict]: Tuple of (parsed content, metadata)
        """
        file_path = self.save_uploaded_file(file)
        
        if file['mime_type'] == 'application/pdf':
            return self.parse_pdf(file_path)
            
        elif file['mime_type'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.parse_docx(file_path), None
            
        elif file['mime_type'] == 'text/html':
            return self.parse_html_file(file_path), None
            
        else:
            raise ValueError(f"Unsupported file type: {file['mime_type']}")

    def get_document_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from parsed document.
        
        Args:
            soup (BeautifulSoup): Parsed document content
            
        Returns:
            str: Extracted text content
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        return ' '.join(chunk for chunk in chunks if chunk)