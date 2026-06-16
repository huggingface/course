from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import requests

from meilisearch.config import Config
from meilisearch.errors import (
    MeilisearchApiError,
    MeilisearchCommunicationError,
    MeilisearchTimeoutError,
)
from meilisearch.models.index import ProximityPrecision
from meilisearch.version import qualified_version


class HttpRequests:
    def __init__(self, config: Config, custom_headers: Optional[Mapping[str, str]] = None) -> None:
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": _build_user_agent(config.client_agents),
        }

        if custom_headers is not None:
            self.headers.update(custom_headers)

    def send_request(
        self,
        http_method: Callable,
        path: str,
        body: Optional[
            Union[
                Mapping[str, Any],
                Sequence[Mapping[str, Any]],
                List[str],
                bytes,
                str,
                int,
                ProximityPrecision,
            ]
        ] = None,
        content_type: Optional[str] = None,
        *,
        serializer: Optional[Type[json.JSONEncoder]] = None,
    ) -> Any:
        if content_type:
            self.headers["Content-Type"] = content_type
        try:
            request_path = self.config.url + "/" + path
            if http_method.__name__ == "get":
                request = http_method(
                    request_path,
                    timeout=self.config.timeout,
                    headers=self.headers,
                )
            elif isinstance(body, bytes):
                request = http_method(
                    request_path,
                    timeout=self.config.timeout,
                    headers=self.headers,
                    data=body,
                )
            else:
                serialize_body = isinstance(body, dict) or body
                data = (
                    json.dumps(body, cls=serializer)
                    if serialize_body
                    else "" if body == "" else "null"
                )

                request = http_method(
                    request_path, timeout=self.config.timeout, headers=self.headers, data=data
                )
            return self.__validate(request)

        except requests.exceptions.Timeout as err:
            raise MeilisearchTimeoutError(str(err)) from err
        except requests.exceptions.ConnectionError as err:
            raise MeilisearchCommunicationError(str(err)) from err

    def get(self, path: str) -> Any:
        return self.send_request(requests.get, path)

    def post(
        self,
        path: str,
        body: Optional[
            Union[Mapping[str, Any], Sequence[Mapping[str, Any]], List[str], bytes, str]
        ] = None,
        content_type: Optional[str] = "application/json",
        *,
        serializer: Optional[Type[json.JSONEncoder]] = None,
    ) -> Any:
        return self.send_request(requests.post, path, body, content_type, serializer=serializer)

    def patch(
        self,
        path: str,
        body: Optional[
            Union[Mapping[str, Any], Sequence[Mapping[str, Any]], List[str], bytes, str]
        ] = None,
        content_type: Optional[str] = "application/json",
    ) -> Any:
        return self.send_request(requests.patch, path, body, content_type)

    def put(
        self,
        path: str,
        body: Optional[
            Union[
                Mapping[str, Any],
                Sequence[Mapping[str, Any]],
                List[str],
                bytes,
                str,
                int,
                ProximityPrecision,
            ]
        ] = None,
        content_type: Optional[str] = "application/json",
        *,
        serializer: Optional[Type[json.JSONEncoder]] = None,
    ) -> Any:
        return self.send_request(requests.put, path, body, content_type, serializer=serializer)

    def delete(
        self,
        path: str,
        body: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]], List[str]]] = None,
    ) -> Any:
        return self.send_request(requests.delete, path, body)

    @staticmethod
    def __to_json(request: requests.Response) -> Any:
        if request.content == b"":
            return request
        return request.json()

    @staticmethod
    def __validate(request: requests.Response) -> Any:
        try:
            request.raise_for_status()
            return HttpRequests.__to_json(request)
        except requests.exceptions.HTTPError as err:
            raise MeilisearchApiError(str(err), request) from err


@lru_cache(maxsize=1)
def _build_user_agent(client_agents: Optional[Tuple[str, ...]] = None) -> str:
    user_agent = qualified_version()
    if not client_agents:
        return user_agent

    return f"{user_agent};{';'.join(client_agents)}"
