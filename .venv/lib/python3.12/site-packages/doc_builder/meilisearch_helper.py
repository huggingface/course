import hashlib
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from time import sleep

from meilisearch.client import Client, TaskInfo
from meilisearch.errors import MeilisearchApiError

# References:
# https://www.meilisearch.com/docs/learn/experimental/vector_search
# https://github.com/meilisearch/meilisearch-python/blob/d5a0babe50b4ce5789892845db98b30d4db72203/tests/index/test_index_search_meilisearch.py#L491-L493
# https://github.com/meilisearch/meilisearch-python/blob/d5a0babe50b4ce5789892845db98b30d4db72203/tests/conftest.py#L132-L146

VECTOR_NAME = "embeddings"
VECTOR_DIM = 768  # dim of https://huggingface.co/BAAI/bge-base-en-v1.5

MeilisearchFunc = Callable[..., tuple[Client, TaskInfo]]


def wait_for_task_completion(func: MeilisearchFunc) -> MeilisearchFunc:
    """
    Decorator to wait for MeiliSearch task completion
    A function that is being decorated should return (Client, TaskInfo)
    """

    @wraps(func)
    def wrapped_meilisearch_function(*args, **kwargs):
        try:
            # Extract the Client and Task info from the function's return value
            client, task = func(*args, **kwargs)
            index_id = args[1]  # Adjust this index based on where it actually appears in your arguments
            task_id = task.task_uid

            max_retries = 3
            retry_count = 0

            while True:
                try:
                    # Get the latest task status from the API
                    task = client.index(index_id).get_task(task_id)
                    # Reset retry count on successful API call
                    retry_count = 0
                except (json.JSONDecodeError, MeilisearchApiError) as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        print(
                            f"Warning: API error getting task status (attempt {retry_count}/{max_retries}): {str(e)}"
                        )
                        sleep(30)  # Wait longer before retrying
                        continue
                    else:
                        raise Exception(
                            f"Failed to get task status for task {task_id} after {max_retries} retries: {str(e)}"
                        ) from e
                except Exception as e:
                    # Other unexpected errors should still fail immediately
                    raise Exception(f"Failed to get task status for task {task_id}: {str(e)}") from e

                # task failed
                if task.status == "failed":
                    # Optionally, retrieve more detailed error information if available
                    error_message = task.error.get("message") if task.error else "Unknown error"
                    error_type = task.error.get("type") if task.error else "Unknown"
                    error_link = task.error.get("link") if task.error else "No additional information"

                    # Raise an exception with the error details
                    raise Exception(
                        f"Task {task_id} failed with error type '{error_type}': {error_message}. More info: {error_link}"
                    )
                # task succeeded
                elif task.status == "succeeded":
                    return task
                # task processing - continue waiting
                else:
                    sleep(20)
        except Exception as e:
            # Re-raise any exception that occurs during the meilisearch operation
            raise Exception(f"Meilisearch operation failed: {str(e)}") from e

    return wrapped_meilisearch_function


def wait_for_all_addition_tasks(client: Client, index_name: str, after_started_at: datetime | None = None):
    """
    Wait for all document addition/update tasks to finish for a specific index
    """
    print(f"Waiting for all addition tasks on index '{index_name}' to finish...")

    # Convert datetime to the format expected by MeiliSearch if provided
    after_started_at_str = None
    if after_started_at:
        after_started_at_str = after_started_at.isoformat()

    # Keep checking until there are no more tasks to process
    while True:
        # Get processing tasks for the specific index
        task_params = {
            "indexUids": [index_name],
            "types": ["documentAdditionOrUpdate"],
            "statuses": ["enqueued", "processing"],
        }
        if after_started_at_str:
            task_params["afterStartedAt"] = after_started_at_str

        processing_tasks = client.get_tasks(task_params)

        if len(processing_tasks.results) == 0:
            break

        print(f"Found {len(processing_tasks.results)} tasks still processing on index '{index_name}', waiting...")
        # Wait for one minute before retrying
        sleep(60)

    # Get all failed tasks for the specific index
    failed_task_ids = []
    from_task = None

    while True:
        failed_params = {"indexUids": [index_name], "types": ["documentAdditionOrUpdate"], "statuses": ["failed"]}
        if after_started_at_str:
            failed_params["afterStartedAt"] = after_started_at_str
        if from_task is not None:
            failed_params["from"] = from_task

        failed_tasks = client.get_tasks(failed_params)

        if len(failed_tasks.results) > 0:
            failed_task_ids.extend([task.task_uid for task in failed_tasks.results])

        # Check if there are more results to fetch
        if not hasattr(failed_tasks, "next") or failed_tasks.next is None:
            break
        from_task = failed_tasks.next

    if failed_task_ids:
        print(f"Failed addition task IDs on index '{index_name}': {failed_task_ids}")

    print("Finished waiting for addition tasks on index '{index_name}' to finish.")


@wait_for_task_completion
def create_embedding_db(client: Client, index_name: str):
    index = client.index(index_name)
    task_info = index.update_embedders({VECTOR_NAME: {"source": "userProvided", "dimensions": VECTOR_DIM}})
    return client, task_info


@wait_for_task_completion
def update_db_settings(client: Client, index_name: str):
    index = client.index(index_name)
    task_info = index.update_settings(
        {
            "searchableAttributes": ["heading1", "heading2", "heading3", "heading4", "heading5", "text"],
            "filterableAttributes": ["product"],
        }
    )
    return client, task_info


def delete_embedding_db(client: Client, index_name: str):
    index = client.index(index_name)
    index.delete()


def clear_embedding_db(client: Client, index_name: str):
    """Delete all documents from an index without deleting the index itself."""
    index = client.index(index_name)
    index.delete_all_documents()


def get_all_document_ids(client: Client, index_name: str) -> set[str]:
    """
    Fetch all document IDs from a Meilisearch index via pagination.
    Only retrieves the 'id' field to minimise payload size.
    """
    all_ids = set()
    offset = 0
    limit = 1000

    while True:
        result = client.index(index_name).get_documents({"fields": ["id"], "limit": limit, "offset": offset})
        docs = result.results
        if not docs:
            break
        for doc in docs:
            all_ids.add(doc.id)
        if len(docs) < limit:
            break
        offset += limit

    return all_ids


def delete_documents_from_db(client: Client, index_name: str, doc_ids: list[str]):
    """Delete a batch of documents by ID from a Meilisearch index."""
    index = client.index(index_name)
    index.delete_documents(doc_ids)


def sanitize_for_id(text):
    """
    Sanitize text to only contain valid Meilisearch ID characters.
    Valid: alphanumeric (a-z, A-Z, 0-9), hyphens (-), and underscores (_)
    See: https://www.meilisearch.com/docs/learn/getting_started/primary_key
    """
    # Replace common separators with underscores
    text = text.replace("/", "_").replace(".", "_").replace(" ", "_")
    # Remove any remaining invalid characters
    text = re.sub(r"[^a-zA-Z0-9_-]", "", text)
    # Collapse multiple underscores
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def generate_doc_id(library: str, page: str, text: str) -> str:
    """
    Generate a unique document ID based on library, page, and a hash of content.
    Format: {library}-{page}-{hash}

    See: https://www.meilisearch.com/docs/learn/getting_started/primary_key
    """
    sanitized_library = sanitize_for_id(library)
    sanitized_page = sanitize_for_id(page)
    # Create hash from text content only
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

    return f"{sanitized_library}-{sanitized_page}-{content_hash}"


@wait_for_task_completion
def add_embeddings_to_db(client: Client, index_name: str, embeddings):
    index = client.index(index_name)
    payload_data = [
        {
            "id": generate_doc_id(e.library, e.page, e.text),
            "text": e.text,
            "source_page_url": e.source_page_url,
            "source_page_title": e.source_page_title,
            "product": e.library,
            "heading1": e.heading1,
            "heading2": e.heading2,
            "heading3": e.heading3,
            "heading4": e.heading4,
            "heading5": e.heading5,
            "_vectors": {VECTOR_NAME: e.embedding},
        }
        for e in embeddings
    ]
    task_info = index.add_documents(payload_data)
    return client, task_info


def swap_indexes(
    client: Client,
    index1_name: str,
    index2_name: str,
    temp_index_name: str | None = None,
):
    """
    Swap indexes and wait for all addition tasks to complete on the temporary index first

    Args:
        client: MeiliSearch client
        index1_name: First index name
        index2_name: Second index name
        temp_index_name: Name of the temporary index to wait for additions on. If None, defaults to index2_name
    """
    # Determine which index is the temporary one
    temp_index = temp_index_name if temp_index_name is not None else index2_name

    # Wait for all addition tasks on the temporary index to complete before swapping
    wait_for_all_addition_tasks(client, temp_index)

    print(f"Starting index swap between '{index1_name}' and '{index2_name}'...")

    # Perform the swap
    task_info = client.swap_indexes([{"indexes": [index1_name, index2_name]}])

    # Wait for the swap task itself to complete
    task_id = task_info.task_uid
    while True:
        task = client.get_task(task_id)
        if task.status == "failed":
            error_message = task.error.get("message") if task.error else "Unknown error"
            error_type = task.error.get("type") if task.error else "Unknown"
            error_link = task.error.get("link") if task.error else "No additional information"
            raise Exception(
                f"Swap task {task_id} failed with error type '{error_type}': {error_message}. More info: {error_link}"
            )
        if task.status == "succeeded":
            break
        sleep(60 * 2)  # wait for 2 minutes

    print(f"Index swap between '{index1_name}' and '{index2_name}' completed successfully.")

    return task


# see https://www.meilisearch.com/docs/learn/core_concepts/documents#upload
MEILISEARCH_PAYLOAD_MAX_MB = 95


def get_meili_chunks(obj_list):
    # Convert max_chunk_size_mb to bytes
    max_chunk_size_bytes = MEILISEARCH_PAYLOAD_MAX_MB * 1024 * 1024

    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for obj in obj_list:
        obj_size = sys.getsizeof(obj)

        if current_chunk_size + obj_size > max_chunk_size_bytes:
            # If adding this object exceeds the chunk size, start a new chunk
            chunks.append(current_chunk)
            current_chunk = [obj]
            current_chunk_size = obj_size
        else:
            # Add the object to the current chunk
            current_chunk.append(obj)
            current_chunk_size += obj_size

    # Don't forget to add the last chunk if it contains any objects
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
