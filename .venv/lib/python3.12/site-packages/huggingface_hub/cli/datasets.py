# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to interact with datasets on the Hugging Face Hub.

Usage:
    # list datasets on the Hub
    hf datasets ls

    # list datasets with a search query
    hf datasets ls --search "code"

    # get info about a dataset
    hf datasets info HuggingFaceFW/fineweb
"""

import enum
from typing import Annotated, get_args

import typer

from huggingface_hub._dataset_viewer import execute_raw_sql_query
from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import DatasetSort_T, ExpandDatasetProperty_T

from ._cli_utils import (
    AuthorOpt,
    FilterOpt,
    FormatWithAutoOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    api_object_to_dict,
    get_hf_api,
    make_expand_properties_parser,
    typer_factory,
)
from ._output import OutputFormatWithAuto, out


_EXPAND_PROPERTIES = sorted(get_args(ExpandDatasetProperty_T))
_SORT_OPTIONS = get_args(DatasetSort_T)
DatasetSortEnum = enum.Enum("DatasetSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    typer.Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


datasets_cli = typer_factory(help="Interact with datasets on the Hub.")


@datasets_cli.command(
    "list | ls",
    examples=[
        "hf datasets ls",
        "hf datasets ls --sort downloads --limit 10",
        'hf datasets ls --search "code"',
    ],
)
def datasets_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        DatasetSortEnum | None,
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """List datasets on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(dataset_info)
        for dataset_info in api.list_datasets(
            filter=filter,
            author=author,
            search=search,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore
        )
    ]
    out.table(results)


@datasets_cli.command(
    "info",
    examples=[
        "hf datasets info HuggingFaceFW/fineweb",
        "hf datasets info my-dataset --expand downloads,likes,tags",
    ],
)
def datasets_info(
    dataset_id: Annotated[str, typer.Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Get info about a dataset on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.dataset_info(repo_id=dataset_id, revision=revision, expand=expand)  # type: ignore
    except RepositoryNotFoundError as e:
        raise CLIError(f"Dataset '{dataset_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{dataset_id}'.") from e
    out.dict(info)


@datasets_cli.command(
    "parquet",
    examples=[
        "hf datasets parquet cfahlgren1/hub-stats",
        "hf datasets parquet cfahlgren1/hub-stats --subset models",
        "hf datasets parquet cfahlgren1/hub-stats --split train",
        "hf datasets parquet cfahlgren1/hub-stats --format json",
    ],
)
def datasets_parquet(
    dataset_id: Annotated[str, typer.Argument(help="The dataset ID (e.g. `username/repo-name`).")],
    subset: Annotated[str | None, typer.Option("--subset", help="Filter parquet entries by subset/config.")] = None,
    split: Annotated[str | None, typer.Option(help="Filter parquet entries by split.")] = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """List parquet file URLs available for a dataset."""
    api = get_hf_api(token=token)
    entries = api.list_dataset_parquet_files(repo_id=dataset_id, config=subset)
    filtered = [entry for entry in entries if split is None or entry.split == split]
    results = [
        {"subset": entry.config, "split": entry.split, "url": entry.url, "size": entry.size} for entry in filtered
    ]
    out.table(results, headers=["subset", "split", "url", "size"], id_key="url")


@datasets_cli.command(
    "sql",
    examples=[
        "hf datasets sql \"SELECT COUNT(*) AS rows FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet')\"",
        "hf datasets sql \"SELECT * FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet') LIMIT 5\" --format json",
    ],
)
def datasets_sql(
    sql: Annotated[str, typer.Argument(help="Raw SQL query to execute.")],
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Execute a raw SQL query with DuckDB against dataset parquet URLs."""
    try:
        result = execute_raw_sql_query(sql_query=sql, token=token)
    except ImportError as e:
        raise CLIError(str(e)) from e
    out.table(result)
