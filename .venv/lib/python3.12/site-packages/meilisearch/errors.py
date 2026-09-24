from __future__ import annotations

import json
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from requests import Response

if TYPE_CHECKING:  # pragma: no cover
    from meilisearch.client import Client
    from meilisearch.index import Index
    from meilisearch.task import TaskHandler

T = TypeVar("T")


class MeilisearchError(Exception):  # pragma: no cover
    """Generic class for Meilisearch error handling"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"MeilisearchError. Error message: {self.message}"


class MeilisearchApiError(MeilisearchError):
    """Error sent by Meilisearch API"""

    def __init__(self, error: str, request: Response) -> None:
        self.status_code = request.status_code
        self.code = None
        self.link = None
        self.type = None

        if request.text:
            json_data = json.loads(request.text)
            self.message = json_data.get("message")
            self.code = json_data.get("code")
            self.link = json_data.get("link")
            self.type = json_data.get("type")
        else:
            self.message = error
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.code and self.link:  # pragma: no cover
            return f"MeilisearchApiError. Error code: {self.code}. Error message: {self.message} Error documentation: {self.link} Error type: {self.type}"

        return f"MeilisearchApiError. {self.message}"


class MeilisearchCommunicationError(MeilisearchError):
    """Error when connecting to Meilisearch"""

    def __str__(self) -> str:  # pragma: no cover
        return f"MeilisearchCommunicationError, {self.message}"


class MeilisearchTimeoutError(MeilisearchError):
    """Error when Meilisearch operation takes longer than expected"""

    def __str__(self) -> str:  # pragma: no cover
        return f"MeilisearchTimeoutError, {self.message}"


def version_error_hint_message(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except MeilisearchApiError as exc:
            exc.message = f"{exc.message}. Hint: It might not be working because you're not up to date with the Meilisearch version that {func.__name__} call requires."
            raise exc

    return wrapper
