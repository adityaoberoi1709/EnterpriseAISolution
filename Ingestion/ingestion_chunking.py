import logging
from pathlib import Path
from typing import List
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import (
PyPDFLoader, 
TextLoader, 
UnstructuredHTMLLoader,
Docx2txtLoader)
from langchain_experimental.text_splitter import SemanticChunker
from config.settings import settings
from config.LLM_Factory import get_embedding_model

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".docx": Docx2txtLoader
}


def load_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    loader = loader_cls(str(path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs


def load_directory(dir_path: str) -> List[Document]:
    path = Path(dir_path)
    if not path.is_dir():
        raise ValueError(f"{dir_path} is not a valid directory")
    all_docs = []
    for file in os.listdir(path):
        file_path = path / file
        if file_path.is_file() and file_path.suffix.lower() in LOADER_MAP:
            docs = load_document(str(file_path))
            all_docs.extend(docs)
    logger.info(f"Loaded total {len(all_docs)} documents from directory {dir_path}")
    return all_docs


def _semantic_split(docs: List[Document]) -> List[Document]:
    text_splitter = SemanticChunker(
        embeddings=get_embedding_model(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    return text_splitter.split_documents(docs)


def _recursive_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def _fixed_split(docs: List[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def chunk_documents(docs: List[Document], strategy=None) -> List[Document]:
    strat = strategy or settings.CHUNKING_STRATEGY
    chunking_map = {
        "semantic": _semantic_split,
        "recursive": _recursive_split,
        "fixed": _fixed_split
    }
    fn = chunking_map.get(strat)
    if fn is None:
        raise ValueError(f"Unknown chunking strategy: {strat}")
    chunks = fn(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({"chunk_index": i, "strategy": strat, "char_count": len(chunk.page_content)})
    logger.info(f"Created {len(chunks)} chunks using {strat} strategy")
    return chunks


def ingest_path(path: str, strategy: str | None = None) -> List[Document]:
    if os.path.isdir(path):
        docs = load_directory(path)
    else:
        docs = load_document(path)
    return chunk_documents(docs, strategy=strategy)
