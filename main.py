import os
import json
import argparse
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.reddit_ingest import fetch_reddit_data
from utils.text_utils import clean_text
from vector_store.generate_embeddings import records_to_documents, build_faiss_index
from genai.generate_insights import generate_insights
from export.save_results import save_insights_to_csv


def run_pipeline(
    query: str,
    subreddits: List[str],
    after_days: int,
    max_results: int,
    index_dir: str,
    output_csv: str,
    embedding_model: str,
    gpt_model: str,
    temperature: float,
    k: int,
):
    print(f"Query: {query}")
    print(f"Subreddits: {subreddits or 'ALL'} | Window: last {after_days} days | Max results: {max_results}")

    raw = fetch_reddit_data(query=query, subreddits=subreddits or None, after_days=after_days, max_results=max_results)

    for r in raw:
        r["text"] = clean_text(r.get("text", ""))
        r["title"] = clean_text(r.get("title", ""))

    docs = records_to_documents(raw)

    build_faiss_index(docs, index_dir=index_dir, model_name=embedding_model)

    result = generate_insights(
        query=query,
        index_dir=index_dir,
        k=k,
        model_name=gpt_model,
        temperature=temperature,
        embedding_model=embedding_model,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_insights_to_csv(result, output_csv)
    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="GenAI Insights Engine for Nutrition Product Research")
    parser.add_argument("--query", required=True, help="User question, e.g., 'emerging consumer trends in toddler probiotic nutrition'")
    parser.add_argument("--subreddits", nargs="*", default=None, help="Optional list of subreddits to constrain search")
    parser.add_argument("--after_days", type=int, default=180)
    parser.add_argument("--max_results", type=int, default=1000)
    parser.add_argument("--index_dir", default="/workspace/vector_store/faiss_index")
    parser.add_argument("--output_csv", default="/workspace/export/insights.csv")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gpt_model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=15)

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.index_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is required.")

    run_pipeline(
        query=args.query,
        subreddits=args.subreddits or [],
        after_days=args.after_days,
        max_results=args.max_results,
        index_dir=args.index_dir,
        output_csv=args.output_csv,
        embedding_model=args.embedding_model,
        gpt_model=args.gpt_model,
        temperature=args.temperature,
        k=args.k,
    )