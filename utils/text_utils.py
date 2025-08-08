import re
from typing import Iterable, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MULTI_WS_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = CODE_BLOCK_PATTERN.sub(" ", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = MULTI_WS_PATTERN.sub(" ", text)
    return text.strip()


def chunk_texts(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "! ", "? ", ", ", "; ", ": ", " ", ""
        ],
    )
    chunks = []
    for txt in texts:
        if not txt:
            continue
        chunks.extend(splitter.split_text(txt))
    return chunks