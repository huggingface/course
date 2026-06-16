"""
Utilities for tracking Meilisearch document IDs in a HuggingFace Hub dataset.

The tracker stores the set of all document IDs currently indexed in Meilisearch.
This allows incremental updates: only new/changed documents are embedded and
uploaded, and documents that no longer exist are deleted.

Tracker repo: hf-doc-build/doc-builder-embeddings-tracker
Tracker file: tracker.json
"""

import io
import json
import os
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download, upload_file

HF_TRACKER_REPO = "hf-doc-build/doc-builder-embeddings-tracker"
TRACKER_FILENAME = "tracker.json"


def load_tracker(hf_token: str | None = None) -> set[str]:
    """
    Load existing document IDs from the HF Hub tracker dataset.

    Returns:
        Set of existing document IDs, or an empty set if the tracker doesn't
        exist yet or cannot be loaded.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=HF_TRACKER_REPO,
            filename=TRACKER_FILENAME,
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ids = set(data["ids"])
        print(f"Loaded {len(ids)} existing IDs from tracker ({HF_TRACKER_REPO}/{TRACKER_FILENAME})")
        return ids
    except Exception as e:
        print(f"Could not load tracker, starting fresh: {e}")
        return set()


def save_tracker(ids: set[str], hf_token: str | None = None) -> None:
    """
    Save current document IDs to the HF Hub tracker dataset.

    Args:
        ids: Set of document IDs to persist.
        hf_token: HuggingFace token for write access (falls back to HF_TOKEN env var).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    data = {
        "ids": sorted(ids),
        "count": len(ids),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    content = json.dumps(data, indent=2, ensure_ascii=False)
    buffer = io.BytesIO(content.encode("utf-8"))

    upload_file(
        path_or_fileobj=buffer,
        path_in_repo=TRACKER_FILENAME,
        repo_id=HF_TRACKER_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Update tracker: {len(ids)} IDs",
    )
    print(f"Saved {len(ids)} IDs to tracker on HF Hub ({HF_TRACKER_REPO}/{TRACKER_FILENAME})")
