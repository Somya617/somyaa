import time
import math
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

PUSHSHIFT_BASE = "https://api.pushshift.io/reddit"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _http_get(url: str, params: Dict) -> Optional[Dict]:
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning(f"HTTP GET failed: {exc} | url={url} params={params}")
        return None


def _paginate_pushshift(endpoint: str, query: str, subreddits: Optional[List[str]], after_ts: int, before_ts: int, max_results: int, kind: str) -> Iterable[Dict]:
    size = 100
    fetched = 0
    before_cursor = before_ts

    while fetched < max_results:
        params = {
            "q": query,
            "size": min(size, max_results - fetched),
            "sort": "desc",
            "sort_type": "created_utc",
            "before": before_cursor,
            "after": after_ts,
        }
        if subreddits:
            params["subreddit"] = ",".join(subreddits)

        data = _http_get(endpoint, params)
        if not data or "data" not in data or not data["data"]:
            break

        batch = data["data"]
        for item in batch:
            item["_kind"] = kind
            yield item

        fetched += len(batch)
        min_created = min(d.get("created_utc", before_cursor) for d in batch)
        if min_created == before_cursor:
            break
        before_cursor = min_created
        time.sleep(1.0)


def _normalize_submission(r: Dict) -> Dict:
    return {
        "id": r.get("id"),
        "created_utc": r.get("created_utc"),
        "created": datetime.fromtimestamp(r.get("created_utc", 0), tz=timezone.utc).isoformat(),
        "author": r.get("author"),
        "subreddit": r.get("subreddit"),
        "title": r.get("title") or "",
        "text": r.get("selftext") or "",
        "url": r.get("full_link") or r.get("url") or "",
        "permalink": r.get("permalink") or "",
        "score": r.get("score"),
        "num_comments": r.get("num_comments"),
        "kind": "submission",
    }


def _normalize_comment(r: Dict) -> Dict:
    link_id = r.get("link_id", "").replace("t3_", "")
    permalink = r.get("permalink") or (f"/r/{r.get('subreddit')}/comments/{link_id}/_/{r.get('id')}" if link_id else "")
    return {
        "id": r.get("id"),
        "created_utc": r.get("created_utc"),
        "created": datetime.fromtimestamp(r.get("created_utc", 0), tz=timezone.utc).isoformat(),
        "author": r.get("author"),
        "subreddit": r.get("subreddit"),
        "title": "",
        "text": r.get("body") or "",
        "url": f"https://www.reddit.com{permalink}" if permalink else "",
        "permalink": permalink,
        "score": r.get("score"),
        "num_comments": None,
        "kind": "comment",
    }


def fetch_reddit_data(
    query: str,
    subreddits: Optional[List[str]] = None,
    after_days: int = 90,
    max_results: int = 2000,
    include: Tuple[str, ...] = ("submissions", "comments"),
) -> List[Dict]:
    now = datetime.now(tz=timezone.utc)
    after_dt = now - timedelta(days=after_days)
    after_ts = int(after_dt.timestamp())
    before_ts = int(now.timestamp())

    records: List[Dict] = []

    if "submissions" in include:
        endpoint = f"{PUSHSHIFT_BASE}/submission/search/"
        for raw in _paginate_pushshift(endpoint, query, subreddits, after_ts, before_ts, max_results, kind="submission"):
            records.append(_normalize_submission(raw))

    if "comments" in include:
        endpoint = f"{PUSHSHIFT_BASE}/comment/search/"
        for raw in _paginate_pushshift(endpoint, query, subreddits, after_ts, before_ts, max_results, kind="comment"):
            records.append(_normalize_comment(raw))

    # sort newest first
    records.sort(key=lambda r: r.get("created_utc", 0), reverse=True)
    logger.info(f"Fetched {len(records)} records from Pushshift for query='{query}'")
    return records


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch Reddit data via Pushshift")
    parser.add_argument("--query", required=True)
    parser.add_argument("--subreddits", nargs="*", default=None)
    parser.add_argument("--after_days", type=int, default=90)
    parser.add_argument("--max_results", type=int, default=500)
    parser.add_argument("--include", nargs="*", default=["submissions", "comments"], choices=["submissions", "comments"])
    args = parser.parse_args()

    out = fetch_reddit_data(args.query, args.subreddits, args.after_days, args.max_results, tuple(args.include))
    print(json.dumps(out[:3], indent=2))