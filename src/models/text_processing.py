import re
import uuid
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import mammoth
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTFigure, LTTextBox, LTTextLine
import string
import unidecode

class TextPreprocessor:
    """Handles all text preprocessing operations including file parsing and text chunking."""
    
    def __init__(self):
        self.placeholder_map: Dict[str, str] = {}
        self.MIN_CHARS = 6
        self.MAX_WORDS = 20
        self.MAX_CHARS = self.MAX_WORDS * 10
        self.TOLERANCE = 1e-06

    def generate_placeholder(self) -> str:
        """Generate a unique placeholder for text substitution."""
        return f"PLACEHOLDER_{uuid.uuid4().hex[:8]}"

    def preprocess(self, text: str) -> str:
        """Preprocess text by preserving special elements."""
        # Handle mathematical notations
        math_pattern = r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]'
        math_matches = re.finditer(math_pattern, text)
        for match in math_matches:
            placeholder = self.generate_placeholder()
            self.placeholder_map[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)

        # Handle special symbols
        special_chars_pattern = r'[^\w\s]'
        special_chars = re.finditer(special_chars_pattern, text)
        for match in special_chars:
            if match.group() not in self.placeholder_map.values():
                placeholder = self.generate_placeholder()
                self.placeholder_map[placeholder] = match.group()
                text = text.replace(match.group(), placeholder)

        return text

    def postprocess(self, text: str) -> str:
        """Restore special elements from placeholders."""
        for placeholder, original in self.placeholder_map.items():
            text = text.replace(placeholder, original)
        return text

    def parse_merge(self, texts: str, n: int = 4700) -> List[str]:
        """Split text by periods into chunks of specified size."""
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
        """Split text by newlines into chunks of specified size."""
        sentences = []
        sentence = ""
        for i in texts.split("\n"):
            if len(sentence) + len(i) > n:
                sentences.append(sentence)
                sentence = ""
            sentence += i + ' '
        sentences.append(sentence)
        return sentences

    def init_driver(self):
        """Initialize Selenium WebDriver with appropriate options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
        driver = webdriver.Chrome(options=options)
        return driver

    def quit_driver_and_reap_children(self, driver):
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

    def url2html(self, url: str) -> BeautifulSoup:
        """Convert URL content to BeautifulSoup object."""
        driver = None
        try:
            driver = self.init_driver()
            driver.implicitly_wait(2)
            driver.get(url)
            page = driver.page_source
            soup = BeautifulSoup(page, 'html.parser')

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
                self.quit_driver_and_reap_children(driver)

    def docx2html(self, file_path: str) -> BeautifulSoup:
        """Convert DOCX file to BeautifulSoup object."""
        try:
            result = mammoth.convert_to_html(file_path)
            html = result.value
            soup = BeautifulSoup(html, 'html.parser')
            return soup
        finally:
            os.remove(file_path)

    def pdf2html(self, file_path: str) -> BeautifulSoup:
        """Extract text from PDF and convert to HTML."""
        driver = None
        try:
            url_pdf = 'https://papertohtml.org/'
            driver = self.init_driver()
            
            # Upload PDF
            driver.get(url_pdf)
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.home__upload-input'))
            )
            file_area = driver.find_element(By.CSS_SELECTOR, '.home__upload-input')
            file_area.send_keys(file_path)

            # Transform to HTML
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button'))
            )
            driver.find_element(By.CSS_SELECTOR, 'button').click()

            # Wait for processing
            WebDriverWait(driver, 90).until(EC.url_changes(url_pdf))
            time.sleep(2)  # Additional wait for content
            
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

            return soup
        finally:
            if driver:
                self.quit_driver_and_reap_children(driver)
            os.remove(file_path)

    def get_pdf_title(self, filename: str) -> str:
        """Extract title from PDF metadata or content."""
        # Try metadata first
        with open(filename, "rb") as f:
            docinfo = PdfReader(f).getDocumentInfo()
            if docinfo and docinfo.title:
                title = docinfo.title
                if self._valid_title(title):
                    return title

        # Try extracting from content
        title = self._extract_text_title(filename)
        if self._valid_title(title):
            return title

        # Fallback to filename
        return os.path.basename(os.path.splitext(filename)[0])

    def _extract_text_title(self, filename: str) -> str:
        """Extract title from PDF text content."""
        largest_text = {'contents': '', 'y0': 0, 'size': 0}
        
        with open(filename, 'rb') as fp:
            parser = PDFParser(fp)
            doc = PDFDocument(parser, '')
            parser.set_document(doc)
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                layout = device.get_result()
                for lt_obj in layout:
                    if isinstance(lt_obj, LTFigure):
                        largest_text = self._extract_figure_text(lt_obj, largest_text)
                    elif isinstance(lt_obj, (LTTextBox, LTTextLine)):
                        stripped = re.sub(r'[ \t\n]', '', lt_obj.get_text().strip())
                        if len(stripped) > self.MAX_CHARS * 2:
                            continue
                        largest_text = self._extract_largest_text(lt_obj, largest_text)
                break  # Only process first page

        text = largest_text['contents'].strip()
        text = re.sub(r'(\(cid:[0-9 \t-]*\))*', '', text)
        text = re.sub(r'[\t\n]', '', text)
        text = re.sub(r'\.', '', text)
        return self._sanitize_title(text)

    def _extract_figure_text(self, lt_obj, largest_text):
        """Extract text from PDF figure objects."""
        text = ''
        line = ''
        y0 = 0
        size = 0
        for child in lt_obj:
            if isinstance(child, LTChar):
                if child.size != size:
                    largest_text = self._update_largest_text(line, y0, size, largest_text)
                    text += line + '\n'
                    line = child.get_text()
                    y0 = child.y0
                    size = child.size
                else:
                    line += child.get_text()
        return largest_text

    def _extract_largest_text(self, obj, largest_text):
        """Extract largest text from PDF text objects."""
        for i, child in enumerate(obj):
            if isinstance(child, LTTextLine):
                for j, child2 in enumerate(child):
                    if j > 1 and isinstance(child2, LTChar):
                        largest_text = self._update_largest_text(
                            child.get_text(), child2.y0, child2.size, largest_text
                        )
                        break
        return largest_text

    def _update_largest_text(self, line: str, y0: float, size: float, largest_text: Dict) -> Dict:
        """Update largest text tracking based on size and position."""
        if size == largest_text['size'] == 0 and (y0 - largest_text['y0'] < -self.TOLERANCE):
            return largest_text

        line = re.sub(r'\n$', ' ', line)

        if (size - largest_text['size'] > self.TOLERANCE):
            largest_text = {
                'contents': line,
                'y0': y0,
                'size': size
            }
        elif abs(size - largest_text['size']) <= self.TOLERANCE:
            largest_text['contents'] = largest_text['contents'] + line
            largest_text['y0'] = y0

        return largest_text

    def _valid_title(self, title: str) -> bool:
        """Check if extracted title is valid."""
        if not title or len(title.strip()) < self.MIN_CHARS:
            return False
            
        is_placeholder = bool(re.search(
            r'^[0-9 \t-]+(abstract|introduction)?\s+$|^(abstract|unknown|title|untitled):?$',
            title.strip().lower()
        ))
        
        is_copyright = bool(re.search(
            r'paper\s+title|technical\s+report|proceedings|preprint|to\s+appear|submission|'
            r'(integrated|international).*conference|transactions\s+on|symposium\s+on|downloaded\s+from\s+http',
            title.lower()
        ))

        stripped_ascii = ''.join([c for c in title.strip() if c in string.ascii_letters])
        ascii_length = len(stripped_ascii)
        stripped_chars = re.sub(r'[ \t\n]', '', title.strip())
        chars_length = len(stripped_chars)
        is_serial_number = ascii_length < chars_length / 2

        return not (is_placeholder or is_copyright or is_serial_number)

    def _sanitize_title(self, title: str) -> str:
        """Clean and format extracted title."""
        # Limit length
        words = title.split(' ')
        title = ' '.join(words[0:self.MAX_WORDS])
        if len(title) > self.MAX_CHARS:
            title = title[0:self.MAX_CHARS]

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