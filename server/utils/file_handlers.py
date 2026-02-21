import os
from pypdf import PdfReader
import easyocr

# Initialize OCR once
ocr_reader = easyocr.Reader(["en"], gpu=False)

def read_pdf(file_path):
    """Extract text from PDF"""
    try:
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        print(f"PDF error: {e}")
        return ""

def read_image(file_path):
    """Extract text from image using OCR"""
    try:
        results = ocr_reader.readtext(file_path)
        text = " ".join([result[1] for result in results if result[1]])
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

def read_txt(file_path):
    """Read text file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        print(f"TXT error: {e}")
        return ""

def extract_text(file_path):
    """Extract text based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        return read_image(file_path)
    elif ext == '.txt':
        return read_txt(file_path)
    else:
        return ""