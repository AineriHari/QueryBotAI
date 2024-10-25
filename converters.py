"""
Module for converting DOCX and TXT files to PDF format.

This module provides functionality to convert files in a specified folder:
- DOCX files are converted directly to PDF using either pandoc or the docx2pdf library, depending on the operating system.
- TXT files are first converted to DOCX format and then to PDF.

Supported Operations:
- Convert DOCX files to PDF.
- Convert TXT files to PDF via an intermediate DOCX format.
"""

import os
import platform
import subprocess
from docx import Document
from docx2pdf import convert

# get the os name
os_name = platform.system()


def convert_to_pdf(input_folder):
    """
    Converts DOCX and TXT files in the provided folder to PDF format.
    - DOCX files are converted using pandoc.
    - TXT files are converted to PDFs by first creating a DOCX version and then using pandoc.
    """
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if filename.endswith(".docx"):
            # Convert DOCX to PDF using pandoc
            pdf_file = os.path.splitext(file_path)[0] + ".pdf"
            if os_name == "Linux":
                subprocess.run(["pandoc", file_path, "-o", pdf_file])
            else:
                convert(file_path, pdf_file)
            print(f"Converted {file_path} to PDF.")
        
        elif filename.endswith(".txt"):
            # First, convert TXT to DOCX (using python-docx)
            docx_file = os.path.splitext(file_path)[0] + ".docx"
            doc = Document()
            with open(file_path, 'r') as txt_file:
                content = txt_file.read()
                doc.add_paragraph(content)
                doc.save(docx_file)
            print(f"Converted {file_path} to DOCX: {docx_file}")

            # Then convert the DOCX file to PDF using pandoc
            pdf_file = os.path.splitext(docx_file)[0] + ".pdf"
            if os_name == "Linux":
                subprocess.run(["pandoc", docx_file, "-o", pdf_file])
            else:
                convert(docx_file, pdf_file)
            print(f"Converted {docx_file} to PDF.")
