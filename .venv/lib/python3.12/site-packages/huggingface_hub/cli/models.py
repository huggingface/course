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
"""Contains commands to interact with models on the Hugging Face Hub.

Usage:
    # list models on the Hub
    hf models ls

    # list models with a search query
    hf models ls --search "llama"

    # get info about a model
    hf models info Lightricks/LTX-2
"""

import enum
from typing import Annotated, get_args

import typer

from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelSort_T

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


_EXPAND_PROPERTIES = sorted(get_args(ExpandModelProperty_T))
_SORT_OPTIONS = get_args(ModelSort_T)
ModelSortEnum = enum.Enum("ModelSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    typer.Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=downloads,likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]


models_cli = typer_factory(help="Interact with models on the Hub.")


@models_cli.command(
    "list | ls",
    examples=[
        "hf models ls --sort downloads --limit 10",
        'hf models ls --search "llama" --author meta-llama',
        "hf models ls --num-parameters min:6B,max:128B --sort likes",
    ],
)
def models_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    num_parameters: Annotated[
        str | None,
        typer.Option(help="Filter by parameter count, e.g. 'min:6B,max:128B'."),
    ] = None,
    sort: Annotated[
        ModelSortEnum | None,
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """List models on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(model_info)
        for model_info in api.list_models(
            filter=filter,
            author=author,
            search=search,
            num_parameters=num_parameters,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore
        )
    ]
    out.table(results)


@models_cli.command(
    "info",
    examples=[
        "hf models info meta-llama/Llama-3.2-1B-Instruct",
        "hf models info Qwen/Qwen3.5-9B --expand downloads,likes,tags",
    ],
)
def models_info(
    model_id: Annotated[str, typer.Argument(help="The model ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Get info about a model on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.model_info(repo_id=model_id, revision=revision, expand=expand)  # type: ignore
    except RepositoryNotFoundError as e:
        raise CLIError(f"Model '{model_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{model_id}'.") from e
    out.dict(info)
