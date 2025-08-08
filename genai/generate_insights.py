import os
from typing import Any, Dict, List
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

SYSTEM_PROMPT = """
You are a senior consumer insights strategist for the nutrition industry. Your task is to synthesize concise, actionable business insights from retrieved social discourse (Reddit).

Guidelines:
- Focus on emerging consumer trends, unmet needs, product attributes, usage occasions, purchase drivers, and barriers.
- Be specific to nutrition products and call out segment (e.g., toddlers, probiotics) when evident.
- Include quantitative context (approximate share of mentions, sentiment) when inferable from the evidence.
- Provide 3-7 bullet insights max; each 1-2 sentences, crisp, free of hype.
- For each insight, include 1-2 short supporting quotes (verbatim snippets) with subreddit and date.
- Avoid hallucinations; rely only on retrieved content.
- If evidence is weak, say so.
""".strip()

INSIGHT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nYou have access to the following retrieved excerpts from Reddit (JSON):\n\n{context}\n\nDeliver a JSON object with this exact schema: {output_schema}. Do not include any extra commentary."),
])

OUTPUT_SCHEMA = {
    "query": "string",
    "insights": [
        {
            "insight": "string",
            "supporting_quotes": [
                {
                    "quote": "string",
                    "subreddit": "string",
                    "date": "YYYY-MM-DD",
                    "url": "string"
                }
            ]
        }
    ]
}


def _format_context(docs: List[Document], max_chars: int = 12000) -> str:
    rows = []
    total = 0
    for d in docs:
        meta = d.metadata or {}
        piece = {
            "text": d.page_content[:1000],
            "subreddit": meta.get("subreddit"),
            "date": (meta.get("created") or "")[:10],
            "url": meta.get("url") or ("https://www.reddit.com" + meta.get("permalink", "")),
            "score": meta.get("score"),
            "kind": meta.get("kind"),
            "title": meta.get("title"),
        }
        as_str = str(piece)
        if total + len(as_str) > max_chars:
            break
        rows.append(as_str)
        total += len(as_str)
    return "\n".join(rows)


def generate_insights(
    query: str,
    index_dir: str,
    k: int = 15,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 3, 20)})

    docs: List[Document] = retriever.invoke(query)
    context = _format_context(docs)

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    prompt = INSIGHT_TEMPLATE.format_prompt(question=query, context=context, output_schema=OUTPUT_SCHEMA)
    response = llm.invoke(prompt.to_messages())

    # Best-effort JSON extraction
    import json
    import re
    text = response.content
    match = re.search(r"\{[\s\S]*\}\s*$", text)
    payload: Dict[str, Any]
    if match:
        try:
            payload = json.loads(match.group(0))
        except Exception:
            payload = {"query": query, "insights": []}
    else:
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"query": query, "insights": []}

    if "query" not in payload:
        payload["query"] = query

    # Attach citations metadata for each quote if missing
    def enrich_quote(q: Dict, doc: Document) -> Dict:
        meta = doc.metadata or {}
        q.setdefault("subreddit", meta.get("subreddit", ""))
        q.setdefault("date", (meta.get("created") or "")[:10])
        q.setdefault("url", meta.get("url") or ("https://www.reddit.com" + meta.get("permalink", "")))
        return q

    # If no insights returned, synthesize a high-level note
    if not payload.get("insights"):
        payload["insights"] = [{
            "insight": "The retrieved evidence base was insufficient or too sparse to derive clear emerging consumer trends. Consider broadening the timeframe or query.",
            "supporting_quotes": []
        }]

    return payload