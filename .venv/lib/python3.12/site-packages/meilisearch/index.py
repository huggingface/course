from __future__ import annotations

from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
)
from urllib import parse
from warnings import warn

from camel_converter import to_snake

from meilisearch._httprequests import HttpRequests
from meilisearch._utils import iso_to_date_time
from meilisearch.config import Config
from meilisearch.errors import version_error_hint_message
from meilisearch.models.document import Document, DocumentsResults
from meilisearch.models.index import (
    Embedders,
    Faceting,
    HuggingFaceEmbedder,
    IndexStats,
    LocalizedAttributes,
    OllamaEmbedder,
    OpenAiEmbedder,
    Pagination,
    ProximityPrecision,
    RestEmbedder,
    TypoTolerance,
    UserProvidedEmbedder,
)
from meilisearch.models.task import Task, TaskInfo, TaskResults
from meilisearch.task import TaskHandler

if TYPE_CHECKING:
    from json import JSONEncoder


# pylint: disable=too-many-public-methods, too-many-lines
class Index:
    """
    Indexes routes wrapper.

    Index class gives access to all indexes routes and child routes (inherited).
    https://www.meilisearch.com/docs/reference/api/indexes
    """

    def __init__(
        self,
        config: Config,
        uid: str,
        primary_key: Optional[str] = None,
        created_at: Optional[Union[datetime, str]] = None,
        updated_at: Optional[Union[datetime, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Config object containing permission and location of Meilisearch.
        uid:
            UID of the index on which to perform the index actions.
        primary_key:
            Primary-key of the index.
        """
        self.config = config
        self.http = HttpRequests(config)
        self.task_handler = TaskHandler(config)
        self.uid = uid
        self.primary_key = primary_key
        self.created_at = iso_to_date_time(created_at)
        self.updated_at = iso_to_date_time(updated_at)

    def delete(self) -> TaskInfo:
        """Delete the index.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """

        task = self.http.delete(f"{self.config.paths.index}/{self.uid}")

        return TaskInfo(**task)

    def update(self, primary_key: str) -> TaskInfo:
        """Update the index primary-key.

        Parameters
        ----------
        primary_key:
            The primary key to use for the index.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        payload = {"primaryKey": primary_key}
        task = self.http.patch(f"{self.config.paths.index}/{self.uid}", payload)

        return TaskInfo(**task)

    def fetch_info(self) -> Index:
        """Fetch the info of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        index_dict = self.http.get(f"{self.config.paths.index}/{self.uid}")
        self.primary_key = index_dict["primaryKey"]
        self.created_at = iso_to_date_time(index_dict["createdAt"])
        self.updated_at = iso_to_date_time(index_dict["updatedAt"])
        return self

    def get_primary_key(self) -> str | None:
        """Get the primary key.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.fetch_info().primary_key

    @staticmethod
    def create(config: Config, uid: str, options: Optional[Mapping[str, Any]] = None) -> TaskInfo:
        """Create the index.

        Parameters
        ----------
        uid:
            UID of the index.
        options:
            Options passed during index creation (ex: { 'primaryKey': 'name' }).

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if options is None:
            options = {}
        payload = {**options, "uid": uid}
        task = HttpRequests(config).post(config.paths.index, payload)

        return TaskInfo(**task)

    def get_tasks(self, parameters: Optional[MutableMapping[str, Any]] = None) -> TaskResults:
        """Get all tasks of a specific index from the last one.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get tasks route: https://www.meilisearch.com/docs/reference/api/tasks#get-tasks.

        Returns
        -------
        tasks:
        TaskResults instance with attributes:
            - from
            - next
            - limit
            - results : list of Task instances containing all enqueued, processing, succeeded or failed tasks of the index

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is not None:
            parameters.setdefault("indexUids", []).append(self.uid)
        else:
            parameters = {"indexUids": [self.uid]}

        return self.task_handler.get_tasks(parameters=parameters)

    def get_task(self, uid: int) -> Task:
        """Get one task through the route of a specific index.

        Parameters
        ----------
        uid:
            identifier of the task.

        Returns
        -------
        task:
            Task instance containing information about the processed asynchronous task of an index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.get_task(uid)

    def wait_for_task(
        self,
        uid: int,
        timeout_in_ms: int = 5000,
        interval_in_ms: int = 50,
    ) -> Task:
        """Wait until Meilisearch processes a task until it fails or succeeds.

        Parameters
        ----------
        uid:
            identifier of the task to wait for being processed.
        timeout_in_ms (optional):
            time the method should wait before raising a MeilisearchTimeoutError.
        interval_in_ms (optional):
            time interval the method should wait (sleep) between requests.

        Returns
        -------
        task:
            Task instance containing information about the processed asynchronous task.

        Raises
        ------
        MeilisearchTimeoutError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.wait_for_task(uid, timeout_in_ms, interval_in_ms)

    def get_stats(self) -> IndexStats:
        """Get stats of the index.

        Get information about the number of documents, field frequencies, ...
        https://www.meilisearch.com/docs/reference/api/stats

        Returns
        -------
        stats:
            IndexStats instance containing information about the given index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        stats = self.http.get(f"{self.config.paths.index}/{self.uid}/{self.config.paths.stat}")
        return IndexStats(stats)

    @version_error_hint_message
    def search(self, query: str, opt_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Search in the index.

        Parameters
        ----------
        query:
            String containing the searched word(s)
        opt_params (optional):
            Dictionary containing optional query parameters.
            Note: The vector parameter is only available in Meilisearch >= v1.13.0
            https://www.meilisearch.com/docs/reference/api/search#search-in-an-index

        Returns
        -------
        results:
            Dictionary with hits, offset, limit, processingTime and initial query

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if opt_params is None:
            opt_params = {}
        body = {"q": query, **opt_params}
        return self.http.post(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.search}",
            body=body,
        )

    @version_error_hint_message
    def facet_search(
        self,
        facet_name: str,
        facet_query: Optional[str] = None,
        opt_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a facet search based on the given facet query and facet name.

        Parameters
        ----------
        facet_name:
            String containing the name of the facet on which the search is performed.
        facet_query (optional):
            String containing the searched words
        opt_params (optional):
            Dictionary containing optional query parameters.

        Returns
        -------
        results:
            Dictionary with facetHits, processingTime and initial facet query

        """
        if opt_params is None:
            opt_params = {}
        body = {"facetName": facet_name, "facetQuery": facet_query, **opt_params}
        return self.http.post(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.facet_search}",
            body=body,
        )

    def get_document(
        self, document_id: Union[str, int], parameters: Optional[MutableMapping[str, Any]] = None
    ) -> Document:
        """Get one document with given document identifier.

        Parameters
        ----------
        document_id:
            Unique identifier of the document.
        parameters (optional):
            parameters accepted by the get document route: https://www.meilisearch.com/docs/reference/api/documents#get-one-document

        Returns
        -------
        document:
            Document instance containing the documents information.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is None:
            parameters = {}
        elif "fields" in parameters and isinstance(parameters["fields"], list):
            parameters["fields"] = ",".join(parameters["fields"])

        document = self.http.get(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}/{document_id}?{parse.urlencode(parameters)}"
        )
        return Document(document)

    @version_error_hint_message
    def get_documents(
        self, parameters: Optional[MutableMapping[str, Any]] = None
    ) -> DocumentsResults:
        """Get a set of documents from the index.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get documents route: https://www.meilisearch.com/docs/reference/api/documents#get-documents
            Note: The filter parameter is only available in Meilisearch >= 1.2.0.

        Returns
        -------
        documents:
        DocumentsResults instance with attributes:
            - total
            - offset
            - limit
            - results : list of Document instances containing the documents information

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is None:
            parameters = {}
        response = self.http.post(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}/fetch",
            body=parameters,
        )
        return DocumentsResults(response)

    def get_similar_documents(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        """Get the documents similar to a document.

        Parameters
        ----------
        parameters:
            parameters accepted by the get similar documents route: https://www.meilisearch.com/docs/reference/api/similar#body
            "id" and "embedder" are required.

        Returns
        -------
        results:
            Dictionary with hits, offset, limit, processingTimeMs, and id

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.post(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.similar}",
            body=parameters,
        )

    def add_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        primary_key: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Add documents to the index.

        Parameters
        ----------
        documents:
            List of documents. Each document should be a dictionary.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        url = self._build_url(primary_key)
        add_document_task = self.http.post(url, documents, serializer=serializer)
        return TaskInfo(**add_document_task)

    def add_documents_in_batches(
        self,
        documents: Sequence[Mapping[str, Any]],
        batch_size: int = 1000,
        primary_key: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> List[TaskInfo]:
        """Add documents to the index in batches.

        Parameters
        ----------
        documents:
            List of documents. Each document should be a dictionary.
        batch_size (optional):
            The number of documents that should be included in each batch. Default = 1000
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        tasks_info:
            List of TaskInfo instances containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request.
            Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """

        tasks: List[TaskInfo] = []

        for document_batch in self._batch(documents, batch_size):
            task = self.add_documents(document_batch, primary_key, serializer=serializer)
            tasks.append(task)

        return tasks

    def add_documents_json(
        self,
        str_documents: bytes,
        primary_key: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Add documents to the index from a byte-encoded JSON string.

        Parameters
        ----------
        str_documents:
            Byte-encoded JSON string.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.add_documents_raw(
            str_documents, primary_key, "application/json", serializer=serializer
        )

    def add_documents_csv(
        self,
        str_documents: bytes,
        primary_key: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
    ) -> TaskInfo:
        """Add documents to the index from a byte-encoded CSV string.

        Parameters
        ----------
        str_documents:
            Byte-encoded CSV string.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        csv_delimiter:
            One ASCII character used to customize the delimiter for CSV. Comma used by default.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.add_documents_raw(str_documents, primary_key, "text/csv", csv_delimiter)

    def add_documents_ndjson(
        self,
        str_documents: bytes,
        primary_key: Optional[str] = None,
    ) -> TaskInfo:
        """Add documents to the index from a byte-encoded NDJSON string.

        Parameters
        ----------
        str_documents:
            Byte-encoded NDJSON string.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.add_documents_raw(str_documents, primary_key, "application/x-ndjson")

    def add_documents_raw(
        self,
        str_documents: bytes,
        primary_key: Optional[str] = None,
        content_type: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Add documents to the index from a byte-encoded string.

        Parameters
        ----------
        str_documents:
            Byte-encoded string.
        content_type:
            The content MIME type: 'application/json', 'application/x-dnjson', or 'text/csv'.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        csv_delimiter (optional):
            One ASCII character used to customize the delimiter for CSV.
            Note: The csv delimiter can only be used with the Content-Type text/csv.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        url = self._build_url(primary_key=primary_key, csv_delimiter=csv_delimiter)
        response = self.http.post(url, str_documents, content_type, serializer=serializer)
        return TaskInfo(**response)

    def update_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        primary_key: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Update documents in the index.

        Parameters
        ----------
        documents:
            List of documents. Each document should be a dictionary.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        url = self._build_url(primary_key)
        response = self.http.put(url, documents, serializer=serializer)
        return TaskInfo(**response)

    def update_documents_ndjson(
        self,
        str_documents: str,
        primary_key: Optional[str] = None,
    ) -> TaskInfo:
        """Update documents as a ndjson string in the index.

        Parameters
        ----------
        str_documents:
            String of document from a NDJSON file.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.update_documents_raw(str_documents, primary_key, "application/x-ndjson")

    def update_documents_json(
        self,
        str_documents: str,
        primary_key: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Update documents as a json string in the index.

        Parameters
        ----------
        str_documents:
            String of document from a JSON file.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.update_documents_raw(
            str_documents, primary_key, "application/json", serializer=serializer
        )

    def update_documents_csv(
        self,
        str_documents: str,
        primary_key: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
    ) -> TaskInfo:
        """Update documents as a csv string in the index.

        Parameters
        ----------
        str_documents:
            String of document from a CSV file.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        csv_delimiter:
            One ASCII character used to customize the delimiter for CSV. Comma used by default.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.update_documents_raw(str_documents, primary_key, "text/csv", csv_delimiter)

    def update_documents_raw(
        self,
        str_documents: str,
        primary_key: Optional[str] = None,
        content_type: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
        *,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> TaskInfo:
        """Update documents as a string in the index.

        Parameters
        ----------
        str_documents:
            String of document.
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        type:
            The type of document. Type available: 'csv', 'json', 'jsonl'
        csv_delimiter:
            One ASCII character used to customize the delimiter for CSV.
            Note: The csv delimiter can only be used with the Content-Type text/csv.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        url = self._build_url(primary_key=primary_key, csv_delimiter=csv_delimiter)
        response = self.http.put(url, str_documents, content_type, serializer=serializer)
        return TaskInfo(**response)

    def update_documents_in_batches(
        self,
        documents: Sequence[Mapping[str, Any]],
        batch_size: int = 1000,
        primary_key: Optional[str] = None,
        serializer: Optional[Type[JSONEncoder]] = None,
    ) -> List[TaskInfo]:
        """Update documents to the index in batches.

        Parameters
        ----------
        documents:
            List of documents. Each document should be a dictionary.
        batch_size (optional):
            The number of documents that should be included in each batch. Default = 1000
        primary_key (optional):
            The primary-key used in index. Ignored if already set up.
        serializer (optional):
            A custom JSONEncode to handle serializing fields that the build in json.dumps
            cannot handle, for example UUID and datetime.

        Returns
        -------
        tasks_info:
            List of TaskInfo instances containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request.
            Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """

        tasks = []

        for document_batch in self._batch(documents, batch_size):
            update_task = self.update_documents(document_batch, primary_key, serializer=serializer)
            tasks.append(update_task)

        return tasks

    def delete_document(self, document_id: Union[str, int]) -> TaskInfo:
        """Delete one document from the index.

        Parameters
        ----------
        document_id:
            Unique identifier of the document.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.delete(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}/{document_id}"
        )
        return TaskInfo(**response)

    @version_error_hint_message
    def delete_documents(
        self,
        ids: Optional[List[Union[str, int]]] = None,
        *,
        filter: Optional[  # pylint: disable=redefined-builtin
            Union[str, List[Union[str, List[str]]]]
        ] = None,
    ) -> TaskInfo:
        """Delete multiple documents from the index by id or filter.

        Parameters
        ----------
        ids:
            List of unique identifiers of documents. Note: using ids is depreciated and will be
            removed in a future version.
        filter:
            The filter value information.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if ids:
            warn(
                "The use of ids is depreciated and will be removed in the future",
                DeprecationWarning,
            )
            response = self.http.post(
                f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}/delete-batch",
                [str(i) for i in ids],
            )
        else:
            response = self.http.post(
                f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}/delete",
                body={"filter": filter},
            )
        return TaskInfo(**response)

    def delete_all_documents(self) -> TaskInfo:
        """Delete all documents from the index.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.delete(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}"
        )
        return TaskInfo(**response)

    # GENERAL SETTINGS ROUTES

    def get_settings(self) -> Dict[str, Any]:
        """Get settings of the index.

        https://www.meilisearch.com/docs/reference/api/settings

        Returns
        -------
        settings
            Dictionary containing the settings of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        settings = self.http.get(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.setting}"
        )

        if settings.get("embedders"):
            embedders: dict[
                str,
                OpenAiEmbedder
                | HuggingFaceEmbedder
                | OllamaEmbedder
                | RestEmbedder
                | UserProvidedEmbedder,
            ] = {}
            for k, v in settings["embedders"].items():
                if v.get("source") == "openAi":
                    embedders[k] = OpenAiEmbedder(**v)
                elif v.get("source") == "ollama":
                    embedders[k] = OllamaEmbedder(**v)
                elif v.get("source") == "huggingFace":
                    embedders[k] = HuggingFaceEmbedder(**v)
                elif v.get("source") == "rest":
                    embedders[k] = RestEmbedder(**v)
                else:
                    embedders[k] = UserProvidedEmbedder(**v)

            settings["embedders"] = embedders

        return settings

    def update_settings(self, body: MutableMapping[str, Any]) -> TaskInfo:
        """Update settings of the index.

        https://www.meilisearch.com/docs/reference/api/settings#update-settings

        Parameters
        ----------
        body:
            Dictionary containing the settings of the index.
            More information:
            https://www.meilisearch.com/docs/reference/api/settings#update-settings

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if body.get("embedders"):
            for _, v in body["embedders"].items():
                if "documentTemplateMaxBytes" in v and v["documentTemplateMaxBytes"] is None:
                    del v["documentTemplateMaxBytes"]

        task = self.http.patch(
            f"{self.config.paths.index}/{self.uid}/{self.config.paths.setting}", body
        )

        return TaskInfo(**task)

    def reset_settings(self) -> TaskInfo:
        """Reset settings of the index to default values.

        https://www.meilisearch.com/docs/reference/api/settings#reset-settings

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(f"{self.config.paths.index}/{self.uid}/{self.config.paths.setting}")

        return TaskInfo(**task)

    # RANKING RULES SUB-ROUTES

    def get_ranking_rules(self) -> List[str]:
        """Get ranking rules of the index.

        Returns
        -------
        settings: list
            List containing the ranking rules of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.ranking_rules))

    def update_ranking_rules(self, body: Union[List[str], None]) -> TaskInfo:
        """Update ranking rules of the index.

        Parameters
        ----------
        body:
            List containing the ranking rules.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.ranking_rules), body)

        return TaskInfo(**task)

    def reset_ranking_rules(self) -> TaskInfo:
        """Reset ranking rules of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.ranking_rules),
        )

        return TaskInfo(**task)

    # DISTINCT ATTRIBUTE SUB-ROUTES

    def get_distinct_attribute(self) -> Optional[str]:
        """Get distinct attribute of the index.

        Returns
        -------
        settings:
            String containing the distinct attribute of the index. If no distinct attribute None is returned.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.distinct_attribute))

    def update_distinct_attribute(self, body: str) -> TaskInfo:
        """Update distinct attribute of the index.

        Parameters
        ----------
        body:
            String containing the distinct attribute.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.distinct_attribute), body)

        return TaskInfo(**task)

    def reset_distinct_attribute(self) -> TaskInfo:
        """Reset distinct attribute of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.distinct_attribute),
        )

        return TaskInfo(**task)

    # SEARCHABLE ATTRIBUTES SUB-ROUTES

    def get_searchable_attributes(self) -> List[str]:
        """Get searchable attributes of the index.

        Returns
        -------
        settings:
            List containing the searchable attributes of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.searchable_attributes))

    def update_searchable_attributes(self, body: Union[List[str], None]) -> TaskInfo:
        """Update searchable attributes of the index.

        Parameters
        ----------
        body:
            List containing the searchable attributes.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.searchable_attributes), body)

        return TaskInfo(**task)

    def reset_searchable_attributes(self) -> TaskInfo:
        """Reset searchable attributes of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.searchable_attributes),
        )

        return TaskInfo(**task)

    # DISPLAYED ATTRIBUTES SUB-ROUTES

    def get_displayed_attributes(self) -> List[str]:
        """Get displayed attributes of the index.

        Returns
        -------
        settings:
            List containing the displayed attributes of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.displayed_attributes))

    def update_displayed_attributes(self, body: Union[List[str], None]) -> TaskInfo:
        """Update displayed attributes of the index.

        Parameters
        ----------
        body:
            List containing the displayed attributes.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.displayed_attributes), body)

        return TaskInfo(**task)

    def reset_displayed_attributes(self) -> TaskInfo:
        """Reset displayed attributes of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.displayed_attributes),
        )

        return TaskInfo(**task)

    # STOP WORDS SUB-ROUTES

    def get_stop_words(self) -> List[str]:
        """Get stop words of the index.

        Returns
        -------
        settings:
            List containing the stop words of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.stop_words))

    def update_stop_words(self, body: Union[List[str], None]) -> TaskInfo:
        """Update stop words of the index.

        Parameters
        ----------
        body: list
            List containing the stop words.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.stop_words), body)

        return TaskInfo(**task)

    def reset_stop_words(self) -> TaskInfo:
        """Reset stop words of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.stop_words),
        )

        return TaskInfo(**task)

    # SYNONYMS SUB-ROUTES

    def get_synonyms(self) -> Dict[str, List[str]]:
        """Get synonyms of the index.

        Returns
        -------
        settings: dict
            Dictionary containing the synonyms of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.synonyms))

    def update_synonyms(self, body: Union[Dict[str, List[str]], None]) -> TaskInfo:
        """Update synonyms of the index.

        Parameters
        ----------
        body: dict
            Dictionary containing the synonyms.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.synonyms), body)

        return TaskInfo(**task)

    def reset_synonyms(self) -> TaskInfo:
        """Reset synonyms of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.synonyms),
        )

        return TaskInfo(**task)

    # FILTERABLE ATTRIBUTES SUB-ROUTES

    def get_filterable_attributes(self) -> List[str]:
        """Get filterable attributes of the index.

        Returns
        -------
        settings:
            List containing the filterable attributes of the index

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.filterable_attributes))

    def update_filterable_attributes(self, body: Union[List[str], None]) -> TaskInfo:
        """Update filterable attributes of the index.

        Parameters
        ----------
        body:
            List containing the filterable attributes.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.filterable_attributes), body)

        return TaskInfo(**task)

    def reset_filterable_attributes(self) -> TaskInfo:
        """Reset filterable attributes of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.filterable_attributes),
        )

        return TaskInfo(**task)

    # SORTABLE ATTRIBUTES SUB-ROUTES

    def get_sortable_attributes(self) -> List[str]:
        """Get sortable attributes of the index.

        Returns
        -------
        settings:
            List containing the sortable attributes of the index

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.sortable_attributes))

    def update_sortable_attributes(self, body: Union[List[str], None]) -> TaskInfo:
        """Update sortable attributes of the index.

        Parameters
        ----------
        body:
            List containing the sortable attributes.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.sortable_attributes), body)

        return TaskInfo(**task)

    def reset_sortable_attributes(self) -> TaskInfo:
        """Reset sortable attributes of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.sortable_attributes),
        )

        return TaskInfo(**task)

    # TYPO TOLERANCE SUB-ROUTES

    def get_typo_tolerance(self) -> TypoTolerance:
        """Get typo tolerance of the index.

        Returns
        -------
        settings:
            The typo tolerance settings of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        typo_tolerance = self.http.get(self.__settings_url_for(self.config.paths.typo_tolerance))

        return TypoTolerance(**typo_tolerance)

    def update_typo_tolerance(self, body: Union[Mapping[str, Any], None]) -> TaskInfo:
        """Update typo tolerance of the index.

        Parameters
        ----------
        body: dict
            Dictionary containing the typo tolerance.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.patch(self.__settings_url_for(self.config.paths.typo_tolerance), body)

        return TaskInfo(**task)

    def reset_typo_tolerance(self) -> TaskInfo:
        """Reset typo tolerance of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.typo_tolerance),
        )

        return TaskInfo(**task)

    def get_pagination_settings(self) -> Pagination:
        """Get pagination settngs of the index.

        Returns
        -------
        settings:
            The pagination settings of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        pagination = self.http.get(self.__settings_url_for(self.config.paths.pagination))

        return Pagination(**pagination)

    def update_pagination_settings(self, body: Union[Dict[str, Any], None]) -> TaskInfo:
        """Update the pagination settings of the index.

        Parameters
        ----------
        body: dict
            Dictionary containing the pagination settings.
            https://www.meilisearch.com/docs/reference/api/settings#update-pagination-settings

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.patch(
            path=self.__settings_url_for(self.config.paths.pagination), body=body
        )

        return TaskInfo(**task)

    def reset_pagination_settings(self) -> TaskInfo:
        """Reset pagination settings of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(self.__settings_url_for(self.config.paths.pagination))

        return TaskInfo(**task)

    def get_faceting_settings(self) -> Faceting:
        """Get the faceting settings of an index.

        Returns
        -------
        settings:
            The faceting settings of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """

        faceting = self.http.get(self.__settings_url_for(self.config.paths.faceting))

        return Faceting(**faceting)

    def update_faceting_settings(self, body: Union[Mapping[str, Any], None]) -> TaskInfo:
        """Update the faceting settings of the index.

        Parameters
        ----------
        body: dict
            Dictionary containing the faceting settings.
            https://www.meilisearch.com/docs/reference/api/settings#update-pagination-settings

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.patch(path=self.__settings_url_for(self.config.paths.faceting), body=body)

        return TaskInfo(**task)

    def reset_faceting_settings(self) -> TaskInfo:
        """Reset faceting settings of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(self.__settings_url_for(self.config.paths.faceting))

        return TaskInfo(**task)

    # USER DICTIONARY SUB-ROUTES

    def get_dictionary(self) -> List[str]:
        """Get the dictionary entries of the index.

        Returns
        -------
        settings:
            List containing the dictionary entries of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.dictionary))

    def update_dictionary(self, body: Union[List[str], None]) -> TaskInfo:
        """Update the dictionary of the index.

        Parameters
        ----------
        body:
            List of the new dictionary entries.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.dictionary), body)

        return TaskInfo(**task)

    def reset_dictionary(self) -> TaskInfo:
        """Clear all entries in dictionary

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.dictionary),
        )

        return TaskInfo(**task)

    # TEXT SEPARATOR SUB-ROUTES

    def get_separator_tokens(self) -> List[str]:
        """Get the additional text separator tokens set on this index.

        Returns
        -------
        settings:
            List containing the separator tokens of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.separator_tokens))

    def get_non_separator_tokens(self) -> List[str]:
        """Get the list of disabled text separator tokens on this index.

        Returns
        -------
        settings:
            List containing the disabled separator tokens of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.non_separator_tokens))

    def update_separator_tokens(self, body: Union[List[str], None]) -> TaskInfo:
        """Update the additional separator tokens of the index.

        Parameters
        ----------
        body:
            List of the new separator tokens.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.separator_tokens), body)

        return TaskInfo(**task)

    def update_non_separator_tokens(self, body: Union[List[str], None]) -> TaskInfo:
        """Update the disabled separator tokens of the index.

        Parameters
        ----------
        body:
            List of the newly disabled separator tokens.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.non_separator_tokens), body)

        return TaskInfo(**task)

    def reset_separator_tokens(self) -> TaskInfo:
        """Clear all additional separator tokens

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.separator_tokens),
        )

        return TaskInfo(**task)

    def reset_non_separator_tokens(self) -> TaskInfo:
        """Clear all disabled separator tokens

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.non_separator_tokens),
        )

        return TaskInfo(**task)

    # EMBEDDERS SUB-ROUTES

    def get_embedders(self) -> Embedders | None:
        """Get embedders of the index.

        Returns
        -------
        settings:
            The embedders settings of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.get(self.__settings_url_for(self.config.paths.embedders))

        if not response:
            return None

        embedders: dict[
            str,
            OpenAiEmbedder
            | HuggingFaceEmbedder
            | OllamaEmbedder
            | RestEmbedder
            | UserProvidedEmbedder,
        ] = {}
        for k, v in response.items():
            if v.get("source") == "openAi":
                embedders[k] = OpenAiEmbedder(**v)
            elif v.get("source") == "ollama":
                embedders[k] = OllamaEmbedder(**v)
            elif v.get("source") == "huggingFace":
                embedders[k] = HuggingFaceEmbedder(**v)
            elif v.get("source") == "rest":
                embedders[k] = RestEmbedder(**v)
            else:
                embedders[k] = UserProvidedEmbedder(**v)

        return Embedders(embedders=embedders)

    def update_embedders(self, body: Union[MutableMapping[str, Any], None]) -> TaskInfo:
        """Update embedders of the index.

        Parameters
        ----------
        body: dict
            Dictionary containing the embedders.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """

        if body:
            for _, v in body.items():
                if "documentTemplateMaxBytes" in v and v["documentTemplateMaxBytes"] is None:
                    del v["documentTemplateMaxBytes"]

        task = self.http.patch(self.__settings_url_for(self.config.paths.embedders), body)

        return TaskInfo(**task)

    def reset_embedders(self) -> TaskInfo:
        """Reset embedders of the index to default values.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.embedders),
        )

        return TaskInfo(**task)

    # SEARCH CUTOFF MS SETTINGS

    def get_search_cutoff_ms(self) -> int | None:
        """Get the search cutoff in ms of the index.

        Returns
        -------
        settings:
            Integer value of search cutoff in ms of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.__settings_url_for(self.config.paths.search_cutoff_ms))

    def update_search_cutoff_ms(self, body: Union[int, None]) -> TaskInfo:
        """Update the search cutoff in ms of the index.

        Parameters
        ----------
        body:
            Integer value of the search cutoff time in ms.

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.search_cutoff_ms), body)

        return TaskInfo(**task)

    def reset_search_cutoff_ms(self) -> TaskInfo:
        """Reset the search cutoff of the index

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.search_cutoff_ms),
        )

        return TaskInfo(**task)

    # PROXIMITY PRECISION SETTINGS

    def get_proximity_precision(self) -> ProximityPrecision:
        """Get the proximity_precision of the index.

        Returns
        -------
        settings:
            proximity_precision of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.get(self.__settings_url_for(self.config.paths.proximity_precision))
        return ProximityPrecision[to_snake(response).upper()]

    def update_proximity_precision(self, body: Union[ProximityPrecision, None]) -> TaskInfo:
        """Update the proximity_precision of the index.

        Parameters
        ----------
        body:
            proximity_precision

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.proximity_precision), body)

        return TaskInfo(**task)

    def reset_proximity_precision(self) -> TaskInfo:
        """Reset the proximity_precision of the index

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.proximity_precision),
        )

        return TaskInfo(**task)

    # LOCALIZED ATTRIBUTES SETTINGS

    def get_localized_attributes(self) -> Union[List[LocalizedAttributes], None]:
        """Get the localized_attributes of the index.

        Returns
        -------
        settings:
            localized_attributes of the index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.get(self.__settings_url_for(self.config.paths.localized_attributes))

        if not response:
            return None

        return [LocalizedAttributes(**attrs) for attrs in response]

    def update_localized_attributes(
        self, body: Union[List[Mapping[str, List[str]]], None]
    ) -> TaskInfo:
        """Update the localized_attributes of the index.

        Parameters
        ----------
        body:
            localized_attributes

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.put(self.__settings_url_for(self.config.paths.localized_attributes), body)

        return TaskInfo(**task)

    def reset_localized_attributes(self) -> TaskInfo:
        """Reset the localized_attributes of the index

        Returns
        -------
        task_info:
            TaskInfo instance containing information about a task to track the progress of an asynchronous process.
            https://www.meilisearch.com/docs/reference/api/tasks#get-one-task

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.delete(
            self.__settings_url_for(self.config.paths.localized_attributes),
        )

        return TaskInfo(**task)

    @staticmethod
    def _batch(
        documents: Sequence[Mapping[str, Any]], batch_size: int
    ) -> Generator[Sequence[Mapping[str, Any]], None, None]:
        total_len = len(documents)
        for i in range(0, total_len, batch_size):
            yield documents[i : i + batch_size]

    def __settings_url_for(self, sub_route: str) -> str:
        return f"{self.config.paths.index}/{self.uid}/{self.config.paths.setting}/{sub_route}"

    def _build_url(
        self,
        primary_key: Optional[str] = None,
        csv_delimiter: Optional[str] = None,
    ) -> str:
        parameters = {}
        if primary_key:
            parameters["primaryKey"] = primary_key
        if csv_delimiter:
            parameters["csvDelimiter"] = csv_delimiter
        if primary_key is None and csv_delimiter is None:
            return f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}"
        return f"{self.config.paths.index}/{self.uid}/{self.config.paths.document}?{parse.urlencode(parameters)}"
