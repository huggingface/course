# pylint: disable=too-many-public-methods

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from urllib import parse

from meilisearch._httprequests import HttpRequests
from meilisearch.config import Config
from meilisearch.errors import MeilisearchError
from meilisearch.index import Index
from meilisearch.models.key import Key, KeysResults
from meilisearch.models.task import Batch, BatchResults, Task, TaskInfo, TaskResults
from meilisearch.task import TaskHandler


class Client:
    """
    A client for the Meilisearch API

    A client instance is needed for every Meilisearch API method to know the location of
    Meilisearch and its permissions.
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        client_agents: Optional[Tuple[str, ...]] = None,
        custom_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the Meilisearch API (ex: http://localhost:7700)
        api_key:
            The optional API key for Meilisearch
        timeout (optional):
            The amount of time in seconds that the client will wait for a response before timing
            out.
        client_agents (optional):
            Used to send additional client agent information for clients extending the functionality
            of this client.
        custom_headers (optional):
            Custom headers to add when sending data to Meilisearch.
        """

        self.config = Config(url, api_key, timeout=timeout, client_agents=client_agents)

        self.http = HttpRequests(self.config, custom_headers)

        self.task_handler = TaskHandler(self.config)

    def create_index(self, uid: str, options: Optional[Mapping[str, Any]] = None) -> TaskInfo:
        """Create an index.

        Parameters
        ----------
        uid: str
            UID of the index.
        options (optional): dict
            Options passed during index creation (ex: primaryKey).

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
        return Index.create(self.config, uid, options)

    def delete_index(self, uid: str) -> TaskInfo:
        """Deletes an index

        Parameters
        ----------
        uid:
            UID of the index.

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

        task = self.http.delete(f"{self.config.paths.index}/{uid}")

        return TaskInfo(**task)

    def get_indexes(self, parameters: Optional[Mapping[str, Any]] = None) -> Dict[str, List[Index]]:
        """Get all indexes.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get indexes route: https://www.meilisearch.com/docs/reference/api/indexes#list-all-indexes

        Returns
        -------
        indexes:
            Dictionary with limit, offset, total and results a list of Index instances.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is None:
            parameters = {}
        response = self.http.get(f"{self.config.paths.index}?{parse.urlencode(parameters)}")
        response["results"] = [
            Index(
                self.config,
                index["uid"],
                index["primaryKey"],
                index["createdAt"],
                index["updatedAt"],
            )
            for index in response["results"]
        ]
        return response

    def get_raw_indexes(
        self, parameters: Optional[Mapping[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get all indexes in dictionary format.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get indexes route: https://www.meilisearch.com/docs/reference/api/indexes#list-all-indexes

        Returns
        -------
        indexes:
            Dictionary with limit, offset, total and results a list of indexes in dictionary format. (e.g [{ 'uid': 'movies' 'primaryKey': 'objectID' }])

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is None:
            parameters = {}
        return self.http.get(f"{self.config.paths.index}?{parse.urlencode(parameters)}")

    def get_index(self, uid: str) -> Index:
        """Get the index.
        This index should already exist.

        Parameters
        ----------
        uid:
            UID of the index.

        Returns
        -------
        index:
            An Index instance containing the information of the fetched index.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return Index(self.config, uid).fetch_info()

    def get_raw_index(self, uid: str) -> Dict[str, Any]:
        """Get the index as a dictionary.
        This index should already exist.

        Parameters
        ----------
        uid:
            UID of the index.

        Returns
        -------
        index:
            An index in dictionary format. (e.g { 'uid': 'movies' 'primaryKey': 'objectID' })

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(f"{self.config.paths.index}/{uid}")

    def index(self, uid: str) -> Index:
        """Create a local reference to an index identified by UID, without doing an HTTP call.
        Calling this method doesn't create an index in the Meilisearch instance, but grants access to all the other methods in the Index class.

        Parameters
        ----------
        uid:
            UID of the index.

        Returns
        -------
        index:
            An Index instance.
        """
        if uid is not None:
            return Index(self.config, uid=uid)
        raise ValueError("The index UID should not be None")

    def multi_search(
        self, queries: Sequence[Mapping[str, Any]], federation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Multi-index search.

        Parameters
        ----------
        queries:
            List of dictionaries containing the specified indexes and their search queries
            https://www.meilisearch.com/docs/reference/api/search#search-in-an-index
        federation: (optional):
            Dictionary containing offset and limit
            https://www.meilisearch.com/docs/reference/api/multi_search

        Returns
        -------
        results:
            Dictionary of results for each search query

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.post(
            f"{self.config.paths.multi_search}",
            body={"queries": queries, "federation": federation},
        )

    def update_documents_by_function(
        self, index_uid: str, queries: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Update Documents by function
        Parameters
        ----------
        index_uid:
            The index_uid where you want to update documents of.
        queries:
            List of dictionaries containing functions with or without filters that you want to use to update documents.

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
        return self.http.post(
            path=f"{self.config.paths.index}/{index_uid}/{self.config.paths.document}/{self.config.paths.edit}",
            body=dict(queries),
        )

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all stats of Meilisearch

        Get information about database size and all indexes
        https://www.meilisearch.com/docs/reference/api/stats

        Returns
        -------
        stats:
            Dictionary containing stats about your Meilisearch instance.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.config.paths.stat)

    def health(self) -> Dict[str, str]:
        """Get health of the Meilisearch server.

        Returns
        -------
        health:
            Dictionary containing the status of the Meilisearch instance.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.config.paths.health)

    def is_healthy(self) -> bool:
        """Get health of the Meilisearch server."""
        try:
            self.health()
        except MeilisearchError:
            return False
        return True

    def get_key(self, key_or_uid: str) -> Key:
        """Gets information about a specific API key.

        Parameters
        ----------
        key_or_uid:
            The key or the uid for which to retrieve the information.

        Returns
        -------
        key:
            The API key.
            https://www.meilisearch.com/docs/reference/api/keys#get-key

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        key = self.http.get(f"{self.config.paths.keys}/{key_or_uid}")

        return Key(**key)

    def get_keys(self, parameters: Optional[Mapping[str, Any]] = None) -> KeysResults:
        """Gets the Meilisearch API keys.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get keys route: https://www.meilisearch.com/docs/reference/api/keys#get-all-keys

        Returns
        -------
        keys:
            API keys.
            https://www.meilisearch.com/docs/reference/api/keys#get-keys

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        if parameters is None:
            parameters = {}
        keys = self.http.get(f"{self.config.paths.keys}?{parse.urlencode(parameters)}")

        return KeysResults(**keys)

    def create_key(self, options: Mapping[str, Any]) -> Key:
        """Creates a new API key.

        Parameters
        ----------
        options:
            Options, the information to use in creating the key (ex: { 'actions': ['*'], 'indexes': ['movies'], 'description': 'Search Key', 'expiresAt': '22-01-01' }).
            An `actions`, an `indexes` and a `expiresAt` fields are mandatory,`None` should be specified for no expiration date.
            `actions`: A list of actions permitted for the key. ["*"] for all actions.
            `indexes`: A list of indexes permitted for the key. ["*"] for all indexes.
            Note that if an expires_at value is included it should be in UTC time.

        Returns
        -------
        key:
            The new API key.
            https://www.meilisearch.com/docs/reference/api/keys#get-keys

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.post(f"{self.config.paths.keys}", options)

        return Key(**task)

    def update_key(self, key_or_uid: str, options: Mapping[str, Any]) -> Key:
        """Update an API key.

        Parameters
        ----------
        key_or_uid:
            The key or the uid of the key for which to update the information.
        options:
            The information to use in creating the key (ex: { 'description': 'Search Key', 'expiresAt': '22-01-01' }). Note that if an
            expires_at value is included it should be in UTC time.

        Returns
        -------
        key:
            The updated API key.
            https://www.meilisearch.com/docs/reference/api/keys#get-keys

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        url = f"{self.config.paths.keys}/{key_or_uid}"
        key = self.http.patch(url, options)

        return Key(**key)

    def delete_key(self, key_or_uid: str) -> int:
        """Deletes an API key.

        Parameters
        ----------
        key:
            The key or the uid of the key to delete.

        Returns
        -------
        keys:
            The Response status code. 204 signifies a successful delete.
            https://www.meilisearch.com/docs/reference/api/keys#get-keys

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        response = self.http.delete(f"{self.config.paths.keys}/{key_or_uid}")

        return response.status_code

    def get_version(self) -> Dict[str, str]:
        """Get version Meilisearch

        Returns
        -------
        version:
            Information about the version of Meilisearch.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.http.get(self.config.paths.version)

    def version(self) -> Dict[str, str]:
        """Alias for get_version

        Returns
        -------
        version:
            Information about the version of Meilisearch.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.get_version()

    def create_dump(self) -> TaskInfo:
        """Trigger the creation of a Meilisearch dump.

        Returns
        -------
        Dump:
            Information about the dump.
            https://www.meilisearch.com/docs/reference/api/dump#create-a-dump

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        task = self.http.post(self.config.paths.dumps)

        return TaskInfo(**task)

    def create_snapshot(self) -> TaskInfo:
        """Trigger the creation of a Meilisearch snapshot.

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
        task = self.http.post(self.config.paths.snapshots)

        return TaskInfo(**task)

    def swap_indexes(self, parameters: List[Mapping[str, List[str]]]) -> TaskInfo:
        """Swap two indexes.

        Parameters
        ----------
        indexes:
            List of indexes to swap (ex: [{"indexes": ["indexA", "indexB"]}).

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
        return TaskInfo(**self.http.post(self.config.paths.swap, parameters))

    def get_tasks(self, parameters: Optional[MutableMapping[str, Any]] = None) -> TaskResults:
        """Get all tasks.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get tasks route: https://www.meilisearch.com/docs/reference/api/tasks#get-tasks.

        Returns
        -------
        task:
            TaskResult instance containing limit, from, next and results containing a list of all
            enqueued, processing, succeeded or failed tasks.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.get_tasks(parameters=parameters)

    def get_task(self, uid: int) -> Task:
        """Get one task.

        Parameters
        ----------
        uid:
            Identifier of the task.

        Returns
        -------
        task:
            Task instance containing information about the processed asynchronous task.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.get_task(uid)

    def cancel_tasks(self, parameters: MutableMapping[str, Any]) -> TaskInfo:
        """Cancel a list of enqueued or processing tasks.

        Parameters
        ----------
        parameters:
            parameters accepted by the cancel tasks route:https://www.meilisearch.com/docs/reference/api/tasks#cancel-tasks.

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
        return self.task_handler.cancel_tasks(parameters=parameters)

    def delete_tasks(self, parameters: MutableMapping[str, Any]) -> TaskInfo:
        """Delete a list of finished tasks.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the delete tasks route:https://www.meilisearch.com/docs/reference/api/tasks#delete-task.
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
        return self.task_handler.delete_tasks(parameters=parameters)

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
            Identifier of the task to wait for being processed.
        timeout_in_ms (optional):
            Time the method should wait before raising a MeilisearchTimeoutError
        interval_in_ms (optional):
            Time interval the method should wait (sleep) between requests

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

    def get_batches(self, parameters: Optional[MutableMapping[str, Any]] = None) -> BatchResults:
        """Get all batches.

        Parameters
        ----------
        parameters (optional):
            parameters accepted by the get batches route: https://www.meilisearch.com/docs/reference/api/batches#get-batches.

        Returns
        -------
        batch:
            BatchResult instance containing limit, from, next and results containing a list of all batches.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.get_batches(parameters=parameters)

    def get_batch(self, uid: int) -> Batch:
        """Get one tasks batch.

        Parameters
        ----------
        uid:
            Identifier of the batch.

        Returns
        -------
        batch:
            Batch instance containing information about the progress of the asynchronous batch.

        Raises
        ------
        MeilisearchApiError
            An error containing details about why Meilisearch can't process your request. Meilisearch error codes are described here: https://www.meilisearch.com/docs/reference/errors/error_codes#meilisearch-errors
        """
        return self.task_handler.get_batch(uid)

    def generate_tenant_token(
        self,
        api_key_uid: str,
        search_rules: Union[Mapping[str, Any], Sequence[str]],
        *,
        expires_at: Optional[datetime.datetime] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Generate a JWT token for the use of multitenancy.

        Parameters
        ----------
        api_key_uid:
            The uid of the API key used as issuer of the token.
        search_rules:
            A Dictionary or list of string which contains the rules to be enforced at search time for all or specific
            accessible indexes for the signing API Key.
            In the specific case where you do not want to have any restrictions you can also use a list ["*"].
        expires_at (optional):
            Date and time when the key will expire. Note that if an expires_at value is included it should be in UTC time.
        api_key (optional):
            The API key parent of the token. If you leave it empty the client API Key will be used.

        Returns
        -------
        jwt_token:
           A string containing the jwt tenant token.
           Note: If your token does not work remember that the search_rules is mandatory and should be well formatted.
           `exp` must be a `datetime` in the future. It's not possible to create a token from the master key.
        """
        # Validate all fields
        if api_key == "" or api_key is None and self.config.api_key is None:
            raise ValueError(
                "An api key is required in the client or should be passed as an argument."
            )
        if api_key_uid == "" or api_key_uid is None or self._valid_uuid(api_key_uid) is False:
            raise ValueError("An uid is required and must comply to the uuid4 format.")
        if not search_rules or search_rules == [""]:
            raise ValueError("The search_rules field is mandatory and should be defined.")
        if expires_at and expires_at < datetime.datetime.now(tz=datetime.timezone.utc):
            raise ValueError("The date expires_at should be in the future.")

        # Standard JWT header for encryption with SHA256/HS256 algorithm
        header = {"typ": "JWT", "alg": "HS256"}

        api_key = str(self.config.api_key) if api_key is None else api_key

        # Add the required fields to the payload
        payload = {
            "apiKeyUid": api_key_uid,
            "searchRules": search_rules,
            "exp": int(datetime.datetime.timestamp(expires_at)) if expires_at is not None else None,
        }

        # Serialize the header and the payload
        json_header = json.dumps(header, separators=(",", ":")).encode()
        json_payload = json.dumps(payload, separators=(",", ":")).encode()

        # Encode the header and the payload to Base64Url String
        header_encode = self._base64url_encode(json_header)
        payload_encode = self._base64url_encode(json_payload)

        secret_encoded = api_key.encode()
        # Create Signature Hash
        signature = hmac.new(
            secret_encoded,
            (header_encode + "." + payload_encode).encode(),
            hashlib.sha256,
        ).digest()
        # Create JWT
        jwt_token = header_encode + "." + payload_encode + "." + self._base64url_encode(signature)

        return jwt_token

    @staticmethod
    def _base64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").replace("=", "")

    @staticmethod
    def _valid_uuid(uuid: str) -> bool:
        uuid4hex = re.compile(
            r"^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}",
            re.I,
        )
        match = uuid4hex.match(uuid)
        return bool(match)
