import os
import csv
from typing import Any, Dict, List


def save_insights_to_csv(result: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    query = result.get("query", "")
    insights: List[Dict] = result.get("insights", [])

    rows: List[Dict[str, Any]] = []
    for idx, ins in enumerate(insights, start=1):
        base = {
            "query": query,
            "insight_rank": idx,
            "insight": ins.get("insight", ""),
        }
        quotes = ins.get("supporting_quotes", []) or [None]
        for q in quotes:
            row = dict(base)
            if q:
                row.update({
                    "quote": q.get("quote", ""),
                    "subreddit": q.get("subreddit", ""),
                    "date": q.get("date", ""),
                    "url": q.get("url", ""),
                })
            else:
                row.update({"quote": "", "subreddit": "", "date": "", "url": ""})
            rows.append(row)

    fieldnames = ["query", "insight_rank", "insight", "quote", "subreddit", "date", "url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)