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
"""Contains commands to interact with spaces on the Hugging Face Hub.

Usage:
    # list spaces on the Hub
    hf spaces ls

    # list spaces with a search query
    hf spaces ls --search "chatbot"

    # get info about a space
    hf spaces info enzostvs/deepsite
"""

import enum
import functools
import itertools
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from typing import Annotated, Literal, get_args

import typer
from packaging import version
from typing_extensions import assert_never

from huggingface_hub._hot_reload.client import multi_replica_reload_events
from huggingface_hub._hot_reload.types import ApiGetReloadEventSourceData, ReloadRegion
from huggingface_hub._space_api import SpaceStage
from huggingface_hub.errors import CLIError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import ExpandSpaceProperty_T, HfApi, SpaceSort_T
from huggingface_hub.utils import StatusLine, are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

from ._cli_utils import (
    AuthorOpt,
    FilterOpt,
    FormatWithAutoOpt,
    LimitOpt,
    RevisionOpt,
    SearchOpt,
    TokenOpt,
    VolumesOpt,
    api_object_to_dict,
    get_hf_api,
    make_expand_properties_parser,
    parse_volumes,
    typer_factory,
)
from ._output import OutputFormatWithAuto, out


HOT_RELOADING_MIN_GRADIO = "6.1.0"


_EXPAND_PROPERTIES = sorted(get_args(ExpandSpaceProperty_T))
_SORT_OPTIONS = get_args(SpaceSort_T)
SpaceSortEnum = enum.Enum("SpaceSortEnum", {s: s for s in _SORT_OPTIONS}, type=str)  # type: ignore[misc]


ExpandOpt = Annotated[
    str | None,
    typer.Option(
        help=f"Comma-separated properties to return. When used, only the listed properties (and id) are returned. Example: '--expand=likes,tags'. Valid: {', '.join(_EXPAND_PROPERTIES)}.",
        callback=make_expand_properties_parser(_EXPAND_PROPERTIES),
    ),
]

spaces_cli = typer_factory(help="Interact with spaces on the Hub.")
volumes_cli = typer_factory(help="Manage volumes for a Space on the Hub.")
spaces_cli.add_typer(volumes_cli, name="volumes")


@spaces_cli.command(
    "list | ls",
    examples=[
        "hf spaces ls --limit 10",
        'hf spaces ls --search "chatbot" --author huggingface',
    ],
)
def spaces_ls(
    search: SearchOpt = None,
    author: AuthorOpt = None,
    filter: FilterOpt = None,
    sort: Annotated[
        SpaceSortEnum | None,
        typer.Option(help="Sort results."),
    ] = None,
    limit: LimitOpt = 10,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """List spaces on the Hub."""
    api = get_hf_api(token=token)
    sort_key = sort.value if sort else None
    results = [
        api_object_to_dict(space_info)
        for space_info in api.list_spaces(
            filter=filter,
            author=author,
            search=search,
            sort=sort_key,
            limit=limit,
            expand=expand,  # type: ignore[arg-type]
        )
    ]
    out.table(results)


@spaces_cli.command(
    "info",
    examples=[
        "hf spaces info enzostvs/deepsite",
        "hf spaces info gradio/theme_builder --expand sdk,runtime,likes",
    ],
)
def spaces_info(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    revision: RevisionOpt = None,
    expand: ExpandOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Get info about a space on the Hub."""
    api = get_hf_api(token=token)
    try:
        info = api.space_info(repo_id=space_id, revision=revision, expand=expand)  # type: ignore[arg-type]
    except RepositoryNotFoundError as e:
        raise CLIError(f"Space '{space_id}' not found.") from e
    except RevisionNotFoundError as e:
        raise CLIError(f"Revision '{revision}' not found on '{space_id}'.") from e
    out.dict(info)


@spaces_cli.command(
    "search",
    examples=[
        'hf spaces search "generate image"',
        'hf spaces search "identify objects in pictures" --sdk gradio --limit 5',
        'hf spaces search "remove background from photo" --description --json',
    ],
)
def spaces_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    filter: FilterOpt = None,
    sdk: Annotated[list[str] | None, typer.Option(help="Filter by SDK (e.g. gradio, docker, static).")] = None,
    include_non_running: Annotated[bool, typer.Option(help="Include non-running spaces in results.")] = False,
    description: Annotated[bool, typer.Option(help="Show AI-generated descriptions.")] = False,
    limit: LimitOpt = 10,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Search spaces on the Hub using semantic search."""
    api = get_hf_api(token=token)
    results = api.search_spaces(
        query=query,
        filter=filter,
        sdk=sdk,
        include_non_running=include_non_running,
        token=token,
    )
    items = []
    for r in itertools.islice(results, limit):
        item: dict = {
            "id": r.id,
            "title": r.title,
            "sdk": r.sdk,
            "likes": r.likes,
            "stage": r.runtime.stage if r.runtime else None,
            "category": r.ai_category,
            "score": round(r.semantic_relevancy_score, 2) if r.semantic_relevancy_score is not None else None,
        }
        if description:
            item["description"] = r.ai_short_description
        items.append(item)
    out.table(items)
    if not description:
        out.hint("Use --description to show AI-generated descriptions.")


@spaces_cli.command(
    "dev-mode",
    examples=[
        "hf spaces dev-mode my-user-name/deepsite",
    ],
)
def dev_mode(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    stop: Annotated[bool, typer.Option(help="Stop dev mode.")] = False,
    token: TokenOpt = None,
):
    """
    Enable or disable dev mode on a Space.

    Spaces Dev Mode eases the debugging of your application and makes iterating on Spaces faster by allowing you to
    restart your application without stopping the Space container itself. This feature is available as part of a PRO
    or Team & Enterprise plan.

    See docs: https://huggingface.co/docs/hub/spaces-dev-mode
    """
    api = get_hf_api(token=token)
    if stop:
        api.disable_space_dev_mode(space_id)
        print(f"Dev mode disabled for '{space_id}'")
        return
    api.enable_space_dev_mode(space_id)
    info = api.space_info(space_id)
    folder = getattr(info.card_data, "dev-mode-folder", "" if info.sdk == "docker" else "/home/user/app")
    folder_query_param = f"folder={folder}" if folder else ""
    print(f"Dev mode is currently building, track the progress here: https://huggingface.co/spaces/{info.id}")
    intermediate_statuses_and_messages = {
        SpaceStage.BUILDING: "building...",
        SpaceStage.RUNNING_BUILDING: "building...",
        SpaceStage.APP_STARTING: "app starting...",
        SpaceStage.RUNNING_APP_STARTING: "app starting...",
    }
    status = StatusLine()
    while True:
        info = api.space_info(space_id)
        if info.runtime is None:
            print("Runtime of the space unavailable")
            return
        if info.runtime.stage not in intermediate_statuses_and_messages:
            break
        status.update(intermediate_statuses_and_messages[info.runtime.stage])
        time.sleep(1)
    if info.runtime.stage != SpaceStage.RUNNING:
        status.done(f"Dev mode is not ready (stage='{info.runtime.stage}')")
        return
    status.done("Dev mode ready!")
    print("Connect to dev environment:")
    print("")
    print("Web:")
    vscode_web_url = f"https://huggingface.co/spaces/{info.id}/dev-mode/vscode-web"
    if folder_query_param:
        vscode_web_url += f"?{folder_query_param}"
    ssh_host = f"{info.subdomain}@ssh.hf.space"
    print(f"  * VSCode: {vscode_web_url}")
    print("")
    print("Local:")
    print("1. Add your SSH key to https://huggingface.co/settings/keys")
    print(f"2. SSH with `ssh -i <your_key> {ssh_host}`")
    print("   Or open")
    print(f"  * VSCode: vscode://vscode-remote/ssh-remote+{ssh_host}{folder}")
    print(f"  * Cursor: cursor://vscode-remote/ssh-remote+{ssh_host}{folder}")
    print("")
    print("PS: Dev mode stops after 48h of inactivity, don't forget to save your changes regularly.")


@spaces_cli.command(
    "logs",
    examples=[
        "hf spaces logs username/my-space",
        "hf spaces logs username/my-space --build",
        "hf spaces logs -f username/my-space",
        "hf spaces logs -n 50 username/my-space",
    ],
)
def spaces_logs(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    build: Annotated[
        bool,
        typer.Option(
            "--build",
            help="Fetch the container build logs instead of the run logs. Useful when a Space is stuck in BUILD_ERROR.",
        ),
    ] = False,
    follow: Annotated[
        bool,
        typer.Option(
            "-f",
            "--follow",
            help="Follow log output (stream until the server closes the stream). Without this flag, only currently available logs are printed.",
        ),
    ] = False,
    tail: Annotated[
        int | None,
        typer.Option(
            "-n",
            "--tail",
            help="Number of lines to show from the end of the logs.",
        ),
    ] = None,
    token: TokenOpt = None,
) -> None:
    """Fetch the run or build logs of a Space.

    By default, prints currently available run logs and exits (non-blocking, like
    `docker logs`). Use --follow/-f to stream until the server closes the stream.
    Use --build to see the container build logs instead (useful when a Space is
    stuck in BUILD_ERROR).
    """
    if follow and tail is not None:
        raise CLIError(
            "Cannot use --follow and --tail together. Use --follow to stream logs or --tail to show recent logs."
        )

    api = get_hf_api(token=token)
    logs = api.fetch_space_logs(space_id, build=build, follow=follow)
    if tail is not None:
        logs = deque(logs, maxlen=tail)
    found_logs = False
    for line in logs:
        clean_line = line.strip()
        out.text(clean_line)
        if clean_line:
            found_logs = True
    if not found_logs and not build:
        out.hint(f"No run logs found for space {space_id}. Try passing --build to fetch build logs instead.")


@spaces_cli.command(
    "hot-reload",
    examples=[
        "hf spaces hot-reload username/repo-name app.py     # Open an interactive editor to the remote app.py file",
        "hf spaces hot-reload username/repo-name -f app.py  # Take local version from ./app.py and patch app.py remotely",
        "hf spaces hot-reload username/repo-name app.py -f src/app.py # Take local version from ./src/app.py",
    ],
)
def spaces_hot_reload(
    space_id: Annotated[
        str,
        typer.Argument(
            help="The space ID (e.g. `username/repo-name`).",
        ),
    ],
    filename: Annotated[
        str | None,
        typer.Argument(
            help="Path to the Python file in the Space repository. Can be omitted when --local-file is specified and path in repository matches."
        ),
    ] = None,
    local_file: Annotated[
        str | None,
        typer.Option(
            "--local-file",
            "-f",
            help="Path of local file. Interactive editor mode if not specified",
        ),
    ] = None,
    skip_checks: Annotated[bool, typer.Option(help="Skip hot-reload compatibility checks.")] = False,
    skip_summary: Annotated[bool, typer.Option(help="Skip summary display after hot-reload is triggered")] = False,
    token: TokenOpt = None,
) -> None:
    """
    Hot-reload any Python file of a Space without a full rebuild + restart.

    ⚠ This feature is experimental ⚠

    Only works with Gradio SDK (6.1+)
    Opens an interactive editor unless --local-file/-f is specified.

    This command patches the live Python process using https://github.com/breuleux/jurigged
    (AST-based diffing, in-place function updates, etc.), integrated with Gradio's native hot-reload support
    (meaning that Gradio demo object changes are reflected in the UI)

    The command creates a remote commit.
    If you are working from a local clone, run `git pull --autostash` afterwards
    to bring the commit back and keep your local git state in sync.
    """

    typer.secho("This feature is experimental and subject to change", fg=typer.colors.BRIGHT_BLACK)

    api = get_hf_api(token=token)

    if not skip_checks:
        space_info = api.space_info(space_id)
        if space_info.sdk != "gradio":
            raise CLIError(f"Hot-reloading is only available on Gradio SDK. Found {space_info.sdk} SDK")
        if (card_data := space_info.card_data) is None:
            raise CLIError(f"Unable to read cardData for Space {space_id}")
        if (sdk_version := card_data.sdk_version) is None:
            raise CLIError(f"Unable to read sdk_version from {space_id} cardData")
        if version.parse(sdk_version) < version.Version(HOT_RELOADING_MIN_GRADIO):
            raise CLIError(f"Hot-reloading requires Gradio >= {HOT_RELOADING_MIN_GRADIO} (found {sdk_version})")

    if local_file:
        local_path = local_file
        filename = local_file if filename is None else filename
    elif filename:
        if not skip_checks:
            try:
                api.auth_check(
                    repo_type="space",
                    repo_id=space_id,
                    write=True,
                )
            except RepositoryNotFoundError as e:
                raise CLIError(
                    f"Write access check to {space_id} repository failed. Make sure that you are authenticated"
                ) from e
        temp_dir = tempfile.TemporaryDirectory()
        local_path = os.path.join(temp_dir.name, filename)
        if not (pbar_disabled := are_progress_bars_disabled()):
            disable_progress_bars()
        try:
            hf_hub_download(
                repo_type="space",
                repo_id=space_id,
                filename=filename,
                local_dir=temp_dir.name,
            )
        finally:
            if not pbar_disabled:
                enable_progress_bars()
        editor_res = _editor_open(local_path)
        if editor_res == "no-tty":
            raise CLIError("Cannot open an editor (no TTY). Use -f flag to hot-reload from local path")
        if editor_res == "no-editor":
            raise CLIError("No editor found in local environment. Use -f flag to hot-reload from local path")
        if editor_res != 0:
            raise CLIError(f"Editor returned a non-zero exit code while attempting to edit {local_path}")
    else:
        raise CLIError("Either filename or --local-file/-f must be specified")

    commit_info = api.upload_file(
        repo_type="space",
        repo_id=space_id,
        path_or_fileobj=local_path,
        path_in_repo=filename,
        _hot_reload=True,
    )

    if not skip_summary:
        _spaces_hot_reload_summary(
            api=api,
            space_id=space_id,
            commit_sha=commit_info.oid,
            local_path=local_path if local_file else os.path.basename(local_path),
            token=token,
        )


def _spaces_hot_reload_summary(
    api: HfApi,
    space_id: str,
    commit_sha: str,
    local_path: str | None,
    token: str | None,
) -> None:
    space_info = api.space_info(space_id)
    if (runtime := space_info.runtime) is None:
        raise CLIError(f"Unable to read SpaceRuntime from {space_id} infos")
    if (hot_reloading := runtime.hot_reloading) is None:
        raise CLIError(f"Space {space_id} current running version has not been hot-reloaded")
    if hot_reloading.status != "created":
        typer.echo(f"Failed creating hot-reloaded commit. {hot_reloading.replica_statuses=}")
        return

    if (space_host := space_info.host) is None:
        raise CLIError("Unexpected None host on hotReloaded Space")
    if (space_subdomain := space_info.subdomain) is None:
        raise CLIError("Unexpected None subdomain on hotReloaded Space")

    def render_region(region: ReloadRegion) -> str:
        res = ""
        if local_path is not None:
            res += f"{local_path}, "
        if region["startLine"] == region["endLine"]:
            res += f"line {region['startLine'] - 1}"
        else:
            res += f"lines {region['startLine'] - 1}-{region['endLine']}"
        return res

    def display_event(event: ApiGetReloadEventSourceData) -> None:
        if event["data"]["kind"] == "error":
            typer.secho("✘ Unexpected hot-reloading error", bold=True)
            typer.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "exception":
            typer.secho(f"✘ Exception at {render_region(event['data']['region'])}", bold=True)
            typer.secho(event["data"]["traceback"], italic=True)
        elif event["data"]["kind"] == "add":
            typer.secho(f"✔︎ Created {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "delete":
            typer.secho(f"∅ Deleted {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "update":
            typer.secho(f"✔︎ Updated {event['data']['objectName']} {event['data']['objectType']}", bold=True)
        elif event["data"]["kind"] == "run":
            typer.secho(f"▶ Run {render_region(event['data']['region'])}", bold=True)
            typer.secho(event["data"]["codeLines"], italic=True)
        elif event["data"]["kind"] == "ui":
            if event["data"]["updated"]:
                typer.secho("⟳ UI updated", bold=True)
            else:
                typer.secho("∅ UI untouched", bold=True)
        else:
            assert_never(event["data"]["kind"])

    for replica_stream_event in multi_replica_reload_events(
        commit_sha=commit_sha,
        host=space_host,
        subdomain=space_subdomain,
        replica_hashes=[hash for hash, _ in hot_reloading.replica_statuses],
        token=token,
    ):
        if replica_stream_event["kind"] == "event":
            display_event(replica_stream_event["event"])
        elif replica_stream_event["kind"] == "replicaHash":
            typer.secho(f"---- Replica {replica_stream_event['hash']} ----")
        elif replica_stream_event["kind"] == "fullMatch":
            typer.echo("✔︎ Same as first replica")
        else:
            assert_never(replica_stream_event)


PREFERRED_EDITORS = (
    ("code", "code --wait"),
    ("nvim", "nvim"),
    ("nano", "nano"),
    ("vim", "vim"),
    ("vi", "vi"),
)


@functools.cache
def _get_editor_command() -> str | None:
    for env in ("HF_EDITOR", "VISUAL", "EDITOR"):
        if command := os.getenv(env, "").strip():
            return command
    for binary_path, editor_command in PREFERRED_EDITORS:
        if shutil.which(binary_path) is not None:
            return editor_command
    return None


def _editor_open(local_path: str) -> int | Literal["no-tty", "no-editor"]:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return "no-tty"
    if (editor_command := _get_editor_command()) is None:
        return "no-editor"
    command = [*shlex.split(editor_command), local_path]
    res = subprocess.run(command, start_new_session=True)
    return res.returncode


@volumes_cli.command(
    "list | ls",
    examples=[
        "hf spaces volumes ls username/my-space",
    ],
)
def volumes_ls(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """List volumes mounted in a Space."""
    api = get_hf_api(token=token)
    info = api.space_info(space_id)
    if info.runtime is None:
        raise CLIError(f"Runtime not available for Space '{space_id}'.")
    volumes = info.runtime.volumes or []
    items = [api_object_to_dict(v) for v in volumes]
    out.table(items)
    out.hint(
        f"Use `hf spaces volumes set {space_id} -v hf://<repo_type>/<repo_id>:/<mount_path>` to set volumes for a Space."
    )


@volumes_cli.command(
    "set",
    examples=[
        "hf spaces volumes set username/my-space -v hf://models/username/my-model:/models",
        "hf spaces volumes set username/my-space -v hf://buckets/username/my-bucket:/data -v hf://datasets/username/my-dataset:/datasets:ro",
    ],
)
def volumes_set(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    volume: VolumesOpt = None,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Set (replace) volumes for a Space."""
    volumes = parse_volumes(volume)
    if not volumes:
        raise CLIError("At least one volume must be specified with -v/--volume.")
    api = get_hf_api(token=token)
    api.set_space_volumes(space_id, volumes=volumes)
    out.result("Volumes set", space_id=space_id, volumes=[v.to_hf_handle() for v in volumes])
    out.hint(f"Use `hf spaces volumes ls {space_id}` to list volumes for a Space.")


@volumes_cli.command(
    "delete",
    examples=[
        "hf spaces volumes delete username/my-space",
        "hf spaces volumes delete username/my-space --yes",
    ],
)
def volumes_delete(
    space_id: Annotated[str, typer.Argument(help="The space ID (e.g. `username/repo-name`).")],
    yes: Annotated[
        bool,
        typer.Option(
            "-y",
            "--yes",
            help="Answer Yes to prompt automatically.",
        ),
    ] = False,
    format: FormatWithAutoOpt = OutputFormatWithAuto.auto,
    token: TokenOpt = None,
) -> None:
    """Remove all volumes from a Space."""
    out.confirm(f"You are about to remove all volumes from Space '{space_id}'. Proceed?", yes=yes)
    api = get_hf_api(token=token)
    api.delete_space_volumes(space_id)
    out.result("Volumes deleted", space_id=space_id)
    out.hint(
        f"Use `hf spaces volumes set {space_id} -v hf://<repo_type>/<repo_id>:/<mount_path>` to set volumes for a Space."
    )
