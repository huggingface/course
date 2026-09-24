from datetime import datetime
from functools import lru_cache
from typing import Union

import pydantic


@lru_cache(maxsize=1)
def is_pydantic_2() -> bool:
    try:
        # __version__ was added with Pydantic 2 so we know if this errors the version is < 2.
        # Still check the version as a fail safe incase __version__ gets added to verion 1.
        if int(pydantic.__version__[:1]) >= 2:  # type: ignore[attr-defined]
            return True

        # Raise an AttributeError to match the AttributeError on __version__ because in either
        # case we need to get to the same place.
        raise AttributeError  # pragma: no cover
    except AttributeError:  # pragma: no cover
        return False


def iso_to_date_time(iso_date: Union[datetime, str, None]) -> Union[datetime, None]:
    """Handle conversion of iso string to datetime.

    The microseconds from Meilisearch are sometimes too long for python to convert so this
    strips off the last digits to shorten it when that happens.
    """
    if not iso_date:
        return None

    if isinstance(iso_date, datetime):
        return iso_date

    try:
        return datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        split = iso_date.split(".")
        if len(split) < 2:
            raise
        reduce = len(split[1]) - 6
        reduced = f"{split[0]}.{split[1][:-reduce]}Z"
        return datetime.strptime(reduced, "%Y-%m-%dT%H:%M:%S.%fZ")
