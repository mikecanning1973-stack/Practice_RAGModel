"""Populate a Chroma vector store from PDF files.

This script is a small utility to load PDFs from a directory, split them
into text chunks, compute embeddings (via the project's
`get_embedding_func`) and persist them into a Chroma collection on disk.

Environment variables (in `configuration.env`)
------------------------------------------------
- COLLECTION_PATH: filesystem path where the Chroma persistence directory is stored
- COLLECTION_NAME: optional name for the Chroma collection
- PDF_PATH: directory containing PDF files to ingest

Usage
-----
Run from the repository root. Use `--reset` to remove the existing
Chroma persistence directory before repopulating.

PowerShell example:

    python populate_db.py --reset

Be careful: `--reset` permanently deletes the directory at
`COLLECTION_PATH`.
"""
import argparse
import os
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_func import get_embedding_func
from langchain_chroma import Chroma
from langchain_core.documents import Document
import requests
import subprocess
import time

load_dotenv('configuration.env')

COLLECTION_PATH = os.getenv('COLLECTION_PATH')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
PDF_PATH = os.getenv('PDF_PATH')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))

def main():
    """Script entrypoint.

    Parses the `--reset` flag and performs the full ETL: load PDFs,
    split into chunks, and add to Chroma.
    """    

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()

    if args.reset:
        print("\033[92m" + "Clearing DB..." + "\033[0m")
        clear_database()

    print("\033[92m" + "Populating DB..." + "\033[0m")
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print("\033[92m" + "DB finished populating." + "\033[0m")

def load_documents():
    """Load PDF files from `PDF_PATH` and return a list of Documents.

    Returns
    -------
    List[Document]
        A list of LangChain `Document` objects loaded from the PDFs.
    """
    document_loader = PyPDFDirectoryLoader(PDF_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """Split each document into smaller text chunks.

    The splitter configuration (chunk size and overlap) is tuned for
    typical RAG workflows; adjust `chunk_size`/`chunk_overlap` if needed.
    """ 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(documents)   

def add_to_chroma(chunks: list[Document]):
    """Add (or update) chunks into the Chroma vector store.

    - Initializes Chroma with the embedding function from `get_embedding_func`.
    - Calculates deterministic chunk IDs via `calculate_chunk_ids`.
    - Skips chunks that already exist in the collection (based on IDs).
    - Persists the collection to disk.
    """

    db = Chroma(persist_directory=COLLECTION_PATH,
                collection_name=COLLECTION_NAME,
                embedding_function=get_embedding_func())
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print("\033[92m" + f"Number of existing documents in DB: {len(existing_ids)}" + "\033[0m")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print("\033[92m" + f"adding new documents: {len(new_chunks)}" + "\033[0m")
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
#        db.persist()
    else:
        print("\033[92m" + "There are no new documents to add" + "\033[0m")
    
def calculate_chunk_ids(chunks):
    """Compute deterministic IDs for each text chunk.

    The ID format is: <source>:<page>:<chunk_index>
    where `chunk_index` increments for multiple chunks originating from
    the same source+page. The function mutates `chunk.metadata['id']` and
    returns the same list for convenience.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:

        parsedSource = chunk.metadata.get("source").split("\\")
        idSource = parsedSource[6]
#        source = chunk.metadata.get("source")
        source= idSource
        
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
#        print(current_page_id)

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id

    return chunks

def clear_database():
    """Remove the Chroma persistence directory if it exists.

    WARNING: this deletes files permanently. Use `--reset` only when you
    intend to wipe and rebuild the collection.
    """

    if os.path.exists(COLLECTION_PATH):
        shutil.rmtree(COLLECTION_PATH)

if __name__ == "__main__":
    main()