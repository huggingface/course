from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from camel_converter import dict_to_camel as dict_to_camel_func
from camel_converter import dict_to_pascal as dict_to_pascal_func
from camel_converter import dict_to_snake as dict_to_snake_func


def dict_to_camel(func: Callable) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)

        if isinstance(result, dict):
            return dict_to_camel_func(result)

        return result

    return wrapper


def dict_to_pascal(func: Callable) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)

        if isinstance(result, dict):
            return dict_to_pascal_func(result)

        return result

    return wrapper


def dict_to_snake(func: Callable) -> Any:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)

        if isinstance(result, dict):
            return dict_to_snake_func(result)

        return result

    return wrapper
