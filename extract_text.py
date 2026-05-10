from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted + "\n"

    return text


def chunk_text(text, chunk_size=300, overlap=80):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)

    return chunks