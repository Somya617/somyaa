import os
import logging
from typing import Dict, List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.text_utils import chunk_texts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def records_to_documents(records: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    documents: List[Document] = []
    for r in records:
        full_text = ((r.get("title") or "").strip() + "\n\n" + (r.get("text") or "").strip()).strip()
        chunks = chunk_texts([full_text], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            meta = {
                "id": r.get("id"),
                "chunk_id": idx,
                "subreddit": r.get("subreddit"),
                "author": r.get("author"),
                "created": r.get("created"),
                "url": r.get("url"),
                "permalink": r.get("permalink"),
                "kind": r.get("kind"),
                "score": r.get("score"),
                "num_comments": r.get("num_comments"),
                "title": r.get("title"),
            }
            documents.append(Document(page_content=chunk.strip(), metadata=meta))
    return documents


def build_faiss_index(
    documents: List[Document],
    index_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> FAISS:
    os.makedirs(index_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.from_documents(documents, embedding=embeddings)
    vs.save_local(index_dir)
    logger.info(f"Saved FAISS index to {index_dir} with {len(documents)} docs")
    return vs


def load_faiss_index(index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return vs