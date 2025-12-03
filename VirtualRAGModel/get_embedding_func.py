"""Embedding loader and helper for the project.

This module reads the embedding model name from `configuration.env` and
returns an initialized embeddings object compatible with the project's
Chroma vector store. It performs a small self-check (embedding a short
string) to validate connectivity and the model availability.

Environment variables
---------------------
- LLM_EMBEDDING_MODEL: Ollama model identifier to use for embeddings.

Example `configuration.env`:

LLM_EMBEDDING_MODEL=llama3.1:latest

Usage
-----
from get_embedding_func import get_embedding_func
emb = get_embedding_func()
vec = emb.embed_query("some text")

"""
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv('configuration.env')

LLM_EMBEDDING_MODEL = os.getenv('LLM_EMBEDDING_MODEL')

def get_embedding_func():
    """Return an initialized embeddings instance.

    The function performs the following steps:
    1. Validates that `LLM_EMBEDDING_MODEL` is set in `configuration.env`.
    2. Instantiates `OllamaEmbeddings` with that model name.
    3. Runs a quick `embed_query("test")` call to ensure the model is
       available and the Ollama service is reachable.

    Returns
    -------
    embeddings
        An embeddings object with `embed_query` and related methods.

    Raises
    ------
    RuntimeError
        If the model name is missing, the Ollama service is unreachable,
        or the embeddings call returns an empty result.
    """
    try:
        if not LLM_EMBEDDING_MODEL:
            raise ValueError("LLM_EMBEDDING_MODEL not set in configuration.env")

        embeddings = OllamaEmbeddings(model=LLM_EMBEDDING_MODEL)

        # Test the embedding function with a simple string to verify it works
        test_embedding = embeddings.embed_query("test")
        if not test_embedding or len(test_embedding) == 0:
            raise RuntimeError("Embedding function returned empty embedding")

        return embeddings

    except ConnectionError as e:
        # ConnectionError is commonly raised when the Ollama service/daemon
        # is not running or reachable.
        raise RuntimeError("Failed to connect to Ollama. Is the Ollama service running?") from e
    except ValueError as e:
        raise RuntimeError(f"Invalid model configuration: {str(e)}") from e
    except Exception as e:
        # Bubble up more descriptive runtime error to the caller.
        raise RuntimeError(f"Failed to initialize embedding function: {str(e)}") from e
