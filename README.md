# GenAI Nutrition Insights Engine

End-to-end pipeline to ingest social discourse (Reddit), embed into a FAISS vector store, and generate concise business insights using GPT-4.

## Components
- `ingestion/reddit_ingest.py`: Fetch Reddit submissions/comments via Pushshift
- `utils/text_utils.py`: Clean and chunk text
- `vector_store/generate_embeddings.py`: Convert records to LangChain `Document`s and build/load FAISS
- `genai/generate_insights.py`: Retrieve with FAISS and synthesize insights using OpenAI GPT
- `export/save_results.py`: Save insights to CSV
- `main.py`: Orchestrates the pipeline

## Setup
1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r /workspace/requirements.txt
   ```
3. Set environment variable:
   ```bash
   export OPENAI_API_KEY=YOUR_KEY
   ```

## Usage
Run a query end-to-end:
```bash
python /workspace/main.py \
  --query "What are the emerging consumer trends in toddler probiotic nutrition?" \
  --subreddits Parenting toddlers nutrition probiotics \
  --after_days 180 \
  --max_results 800 \
  --index_dir /workspace/vector_store/faiss_index \
  --output_csv /workspace/export/insights.csv \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --gpt_model gpt-4o-mini \
  --temperature 0.2 \
  --k 15
```

The resulting insights CSV will be written to `/workspace/export/insights.csv`. 
