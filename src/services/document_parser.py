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
from config import config
from utils.vtt_parser import extract_text_from_vtt

class DocumentParser:
    """Handles all document parsing operations including PDF, DOCX, and URL content."""

    def __init__(self):
        """Initialize DocumentParser."""
        pass

    def init_selenium_driver(self) -> webdriver.Chrome:
        """Initialize Selenium WebDriver with appropriate options."""
        options = webdriver.ChromeOptions()
        for option in config.document_parser.WEBDRIVER_OPTIONS:
            options.add_argument(option)
        options.add_experimental_option('excludeSwitches', 
                                      config.document_parser.WEBDRIVER_EXCLUDED_SWITCHES)
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

    def save_uploaded_file(self, file: Dict[str, Any], dir_name: str = config.file.UPLOAD_FOLDER) -> str:
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
            driver.implicitly_wait(config.document_parser.WEBDRIVER_IMPLICIT_WAIT)
            driver.get(url)
            page = driver.page_source
            soup = BeautifulSoup(page, 'html.parser')

            # Remove unnecessary elements
            for element_name in config.document_parser.HTML_ELEMENTS_TO_REMOVE:
                elements = soup.select(element_name)
                for item in elements:
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

    def parse_pdf(self, file_path: str) -> Tuple[BeautifulSoup, None]:
        """
        Parse PDF file into BeautifulSoup object.
        
        Args: 
            file_path (str): Path to PDF file
            
        Returns:
            Tuple[BeautifulSoup, None]: Tuple of (parsed content, None)
        """
        driver = None
        try:
            driver = self.init_selenium_driver()
            
            driver.get(config.document_parser.PDF_TO_HTML_URL)
            WebDriverWait(driver, config.document_parser.PDF_UPLOAD_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.home__upload-input'))
            )
            file_area = driver.find_element(By.CSS_SELECTOR, '.home__upload-input')
            file_area.send_keys(file_path)

            WebDriverWait(driver, config.document_parser.PDF_UPLOAD_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button'))
            )
            driver.find_element(By.CSS_SELECTOR, 'button').click()

            # Wait for conversion
            try:
                WebDriverWait(driver, config.document_parser.PDF_CONVERSION_WAIT).until(
                    EC.url_changes(config.document_parser.PDF_TO_HTML_URL)
                )
                
                # Additional wait for page load
                for _ in range(20):
                    page_state = driver.execute_script('return document.readyState;')
                    if page_state == 'complete':
                        break
                    time.sleep(config.document_parser.WEBDRIVER_PAGE_LOAD_WAIT)
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # Clean up the HTML using configured elements to remove
                for element_selector in config.document_parser.PDF_ELEMENTS_TO_REMOVE:
                    elements = soup.select(element_selector)
                    for item in elements:
                        item.decompose()

                # Handle navigation section
                nav = soup.find("nav")
                if nav:
                    for a in nav.findAll('a'):
                        a.replaceWithChildren()

                # Add necessary meta tags
                metatag_a = soup.new_tag("meta", charset=config.text.DEFAULT_ENCODING)
                metatag_b = soup.new_tag('meta')
                metatag_b.attrs['content'] = "width=device-width,initial-scale=1,shrink-to-fit=no"
                metatag_b.attrs['name'] = "viewport"
                soup.head.append(metatag_a)
                soup.head.append(metatag_b)

                return soup, None

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
        with codecs.open(file_path, "r", config.text.DEFAULT_ENCODING) as html_file:
            return BeautifulSoup(html_file.read(), 'html.parser')

    def parse_document(self, file: Dict[str, Any]) -> Tuple[BeautifulSoup, None]:
        """
        Main entry point for parsing any document type.
        
        Args:
            file (dict): File data dictionary with 'mime_type' and content
            
        Returns:
            Tuple[BeautifulSoup, None]: Tuple of (parsed content, None)
        """
        file_path = self.save_uploaded_file(file)
        
        if file['mime_type'] == config.file.MIME_TYPES['pdf']:
            return self.parse_pdf(file_path)
            
        elif file['mime_type'] == config.file.MIME_TYPES['docx']:
            return self.parse_docx(file_path), None
            
        elif file['mime_type'] == config.file.MIME_TYPES['html']:
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
        for script in soup(config.text.HTML_ELEMENTS_TO_REMOVE[:2]):  # Just script and style
            script.decompose()

        # Extract text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        return ' '.join(chunk for chunk in chunks if chunk)
    
    def parse_vtt(self, file_path: str) -> BeautifulSoup:
        """
        Parse VTT file into BeautifulSoup object.
        
        Args:
            file_path (str): Path to VTT file
            
        """
        extracted_text = extract_text_from_vtt(file_path)

        return extracted_text