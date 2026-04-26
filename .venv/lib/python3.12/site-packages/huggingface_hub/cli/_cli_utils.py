# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Contains CLI utilities (styling, helpers)."""

import dataclasses
import datetime
import difflib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar, Union, cast

import click
import typer
from typer.core import TyperCommand, TyperGroup

from huggingface_hub import Volume, __version__, constants
from huggingface_hub.errors import CLIError
from huggingface_hub.utils import (
    ANSI,
    disable_progress_bars,
    get_session,
    hf_raise_for_status,
    installation_method,
    logging,
    tabulate,
)
from huggingface_hub.utils._dotenv import load_dotenv

from ._output import OutputFormatWithAuto, out


logger = logging.get_logger()

# Arbitrary maximum length of a cell in a table output
_MAX_CELL_LENGTH = 35

if TYPE_CHECKING:
    from huggingface_hub.hf_api import HfApi


def get_hf_api(token: str | None = None) -> "HfApi":
    # Import here to avoid circular import
    from huggingface_hub.hf_api import HfApi

    return HfApi(token=token, library_name="huggingface-cli", library_version=__version__)


#### TYPER UTILS

CLI_REFERENCE_URL = "https://huggingface.co/docs/huggingface_hub/en/guides/cli"


def generate_epilog(examples: list[str], docs_anchor: str | None = None) -> str:
    """Generate an epilog with examples and a Learn More section.

    Args:
        examples: List of example commands (without the `$ ` prefix).
        docs_anchor: Optional anchor for the docs URL (e.g., "#hf-download").

    Returns:
        Formatted epilog string.
    """
    docs_url = f"{CLI_REFERENCE_URL}{docs_anchor}" if docs_anchor else CLI_REFERENCE_URL
    examples_str = "\n".join(f"  $ {ex}" for ex in examples)
    return f"""\
Examples
{examples_str}

Learn more
  Use `hf <command> --help` for more information about a command.
  Read the documentation at {docs_url}
"""


TOPIC_T = Literal["main", "help"] | str
FallbackHandlerT = Callable[[list[str], set[str]], int | None]
ExpandPropertyT = TypeVar("ExpandPropertyT", bound=str)


def _format_epilog_no_indent(epilog: str | None, ctx: click.Context, formatter: click.HelpFormatter) -> None:
    """Write the epilog without indentation."""
    if epilog:
        formatter.write_paragraph()
        for line in epilog.split("\n"):
            formatter.write_text(line)


_ALIAS_SPLIT = re.compile(r"\s*\|\s*")


class HFCliTyperGroup(TyperGroup):
    """
    Typer Group that:
    - lists commands alphabetically within sections.
    - separates commands by topic (main, help, etc.).
    - formats epilog without extra indentation.
    - supports aliases via pipe-separated names (e.g. ``name="list | ls"``).
    - rewrites ``--json`` to ``--format json`` for commands that accept ``--format``.
    - rewrites ``spaces/user/repo`` to ``user/repo --type space`` for commands that accept ``--type``.
    - enriches "No such option" / "No such command" errors with available options or commands.
    """

    def invoke(self, ctx: click.Context) -> None:
        """Enrich unknown-option errors with available options or subcommands.

        Catches `NoSuchOption` raised during subcommand `make_context()`
        (option parsing).  For leaf commands (e.g. `hf repos create --test`)
        we list the command's options; for groups (e.g. `hf cache --test`)
        we list subcommands since groups have no user-facing options.
        """
        try:
            return super().invoke(ctx)
        except click.NoSuchOption as e:
            if e.ctx is not None and e.ctx.command is not None:
                cmd = e.ctx.command
                if isinstance(cmd, click.Group):
                    # Group has no user-facing options -> show subcommands instead
                    items = [
                        (name, sub.get_short_help_str(limit=80))
                        for name in cmd.list_commands(e.ctx)
                        if (sub := cmd.get_command(e.ctx, name)) is not None and not sub.hidden
                    ]
                    _enrich_usage_error(e, "commands", items)
                else:
                    # Leaf command -> show its options using Click's rich formatting
                    items = [
                        record
                        for p in cmd.get_params(e.ctx)
                        if isinstance(p, click.Option) and not p.hidden and (record := p.get_help_record(e.ctx))
                    ]
                    _enrich_usage_error(e, "options", items)
            raise

    def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
        cmd_name = args[0] if args and not args[0].startswith("-") else None
        cmd = self.get_command(ctx, cmd_name) if cmd_name else None

        if cmd is not None:
            self._rewrite_json_shorthand(cmd, args)
            self._rewrite_quiet_shorthand(cmd, args)
            self._rewrite_repo_type_prefix(cmd, args)

        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            # Unknown subcommand -> add fuzzy suggestions and list available commands.
            if cmd is None and cmd_name is not None:
                # Expand aliases ("list | ls" → ["list", "ls"]) for accurate fuzzy matching.
                visible_names = [
                    alias
                    for key, registered in self.commands.items()
                    if not registered.hidden
                    for alias in _ALIAS_SPLIT.split(key)
                ]
                matches = difflib.get_close_matches(cmd_name, visible_names)
                if matches:
                    suggestions = ", ".join(f"'{m}'" for m in matches)
                    e.message = f"{e.message.rstrip('.')}. Did you mean {suggestions}?"
                items = [
                    (name, sub.get_short_help_str(limit=80))
                    for name in self.list_commands(ctx)
                    if (sub := self.get_command(ctx, name)) is not None and not sub.hidden
                ]
                _enrich_usage_error(e, "commands", items)
            raise

    @staticmethod
    def _rewrite_json_shorthand(cmd: click.Command, args: list[str]) -> None:
        """Rewrite hidden ``--json`` shorthand to ``--format json``.

        Only applies to commands that accept ``--format``.  This avoids rewriting
        ``--json`` for commands that pass args through to external binaries
        (e.g. ``hf extensions exec``) or that simply don't support ``--format``.
        """
        if "--json" not in args:
            return
        has_format_option = any(isinstance(param, click.Option) and "--format" in param.opts for param in cmd.params)
        if has_format_option:
            if any(arg == "--format" or arg.startswith("--format=") for arg in args):
                raise click.UsageError("'--json' and '--format' are mutually exclusive.")
            idx = args.index("--json")
            args[idx : idx + 1] = ["--format", "json"]

    @staticmethod
    def _rewrite_quiet_shorthand(cmd: click.Command, args: list[str]) -> None:
        """Rewrite ``-q`` / ``--quiet`` shorthand to ``--format quiet``.

        Only applies to commands that accept ``--format`` but do NOT already
        have their own ``--quiet`` / ``-q`` option.
        """
        has_quiet = "-q" in args or "--quiet" in args
        if not has_quiet:
            return
        has_format_option = any(isinstance(param, click.Option) and "--format" in param.opts for param in cmd.params)
        has_quiet_option = any(
            isinstance(param, click.Option) and ("--quiet" in param.opts or "-q" in param.opts) for param in cmd.params
        )
        if has_format_option and not has_quiet_option:
            if any(arg == "--format" or arg.startswith("--format=") for arg in args):
                raise click.UsageError("'--quiet' and '--format' are mutually exclusive.")
            flag = "-q" if "-q" in args else "--quiet"
            idx = args.index(flag)
            args[idx : idx + 1] = ["--format", "quiet"]

    @staticmethod
    def _rewrite_repo_type_prefix(cmd: click.Command, args: list[str]) -> None:
        """Rewrite prefixed repo IDs (e.g. ``spaces/user/repo``) to ``user/repo --type space``.

        Only applies to commands that have a ``--type`` / ``--repo-type`` option and
        at least one repo-ID positional argument (any ``click.Argument`` whose name
        ends with ``_id``, e.g. ``repo_id``, ``from_id``, ``to_id``).  When the
        token that maps to such an argument matches ``{prefix}/org/repo`` (where
        *prefix* is one of ``spaces``, ``datasets``, or ``models``), the prefix is
        stripped and an implicit ``--type {type}`` is appended.  An error is raised
        if ``--type`` is also provided explicitly or if multiple prefixed arguments
        disagree on the repo type.

        Only repo-ID positional slots are inspected so that other positional
        arguments (filenames, local paths, patterns …) are never misinterpreted as
        prefixed repo IDs.
        """
        has_type_option = any(isinstance(param, click.Option) and "--type" in param.opts for param in cmd.params)
        if not has_type_option:
            return

        # Locate all repo-ID positional arguments and their indices among Arguments.
        repo_id_positions: set[int] = set()
        arg_idx = 0
        for param in cmd.params:
            if isinstance(param, click.Argument):
                if param.name in ("repo_id", "from_id", "to_id"):
                    repo_id_positions.add(arg_idx)
                arg_idx += 1

        if not repo_id_positions:
            return

        # Build a set of option names that consume a following value token.
        value_options: set[str] = set()
        for param in cmd.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in (*param.opts, *param.secondary_opts):
                    value_options.add(opt)

        # Walk through args (skipping args[0] = command name) to map positional
        # slots to their indices in `args`.
        positional_count = 0
        repo_id_arg_indices: list[int] = []
        i = 1
        while i < len(args):
            arg = args[i]
            if arg == "--":
                break  # everything after -- is positional literal; stop rewriting
            if arg.startswith("-"):
                if "=" in arg or arg not in value_options:
                    i += 1  # flag or --opt=val — single token
                else:
                    i += 2  # value-taking option — skip the value too
            else:
                if positional_count in repo_id_positions:
                    repo_id_arg_indices.append(i)
                positional_count += 1
                i += 1

        if not repo_id_arg_indices:
            return

        # Check each repo-ID arg for a type prefix and collect rewrites.
        inferred_type: str | None = None
        first_prefix: str | None = None
        rewrites: list[tuple[int, str]] = []  # (args index, new value without prefix)

        for arg_index in repo_id_arg_indices:
            parts = args[arg_index].split("/", 2)
            if len(parts) != 3 or parts[0] not in constants.REPO_TYPES_MAPPING:
                continue
            prefix = parts[0]
            mapped_type = constants.REPO_TYPES_MAPPING[prefix]
            if inferred_type is not None and mapped_type != inferred_type:
                raise click.UsageError(f"Conflicting repo type prefixes: '{first_prefix}/' and '{prefix}/'.")
            inferred_type = mapped_type
            first_prefix = prefix
            rewrites.append((arg_index, f"{parts[1]}/{parts[2]}"))

        if not rewrites:
            return

        # Error if --type / --repo-type was also provided explicitly.
        if any(
            arg == "--type" or arg.startswith("--type=") or arg == "--repo-type" or arg.startswith("--repo-type=")
            for arg in args
        ):
            raise click.UsageError(
                f"Ambiguous repo type: got prefix '{first_prefix}/' in repo ID and explicit --type. Use one or the other."
            )

        # Apply all rewrites and append --type once.
        for arg_index, new_value in rewrites:
            args[arg_index] = new_value
        args.extend(["--type", inferred_type])  # type: ignore

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # Try exact match first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # Fall back to alias lookup: check if cmd_name matches any alias
        # taken from https://github.com/fastapi/typer/issues/132#issuecomment-2417492805
        for registered_name, registered_cmd in self.commands.items():
            aliases = _ALIAS_SPLIT.split(registered_name)
            if cmd_name in aliases:
                return registered_cmd
        return None

    def _alias_map(self) -> dict[str, list[str]]:
        """Build a mapping from primary command name to its aliases (if any)."""
        result: dict[str, list[str]] = {}
        for registered_name in self.commands:
            parts = _ALIAS_SPLIT.split(registered_name)
            primary = parts[0]
            result[primary] = parts[1:]
        return result

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        topics: dict[str, list] = {}
        alias_map = self._alias_map()

        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            help_text = cmd.get_short_help_str(limit=formatter.width)
            aliases = alias_map.get(name, [])
            if aliases:
                help_text = f"{help_text} [alias: {', '.join(aliases)}]"
            topic = getattr(cmd, "topic", "main")
            topics.setdefault(topic, []).append((name, help_text))

        with formatter.section("Main commands"):
            formatter.write_dl(topics["main"])
        for topic in sorted(topics.keys()):
            if topic == "main":
                continue
            with formatter.section(f"{topic.capitalize()} commands"):
                formatter.write_dl(topics[topic])

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Collect only the first example from each command (to keep group help concise)
        # Full examples are shown in individual subcommand help (e.g. `hf buckets sync --help`)
        all_examples: list[str] = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            cmd_examples = getattr(cmd, "examples", [])
            if cmd_examples:
                all_examples.append(cmd_examples[0])

        if all_examples:
            epilog = generate_epilog(all_examples)
            _format_epilog_no_indent(epilog, ctx, formatter)
        elif self.epilog:
            _format_epilog_no_indent(self.epilog, ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:  # type: ignore[name-defined]
        # For aliased commands ("list | ls"), use the primary name (first entry).
        primary_names: list[str] = []
        for name in self.commands:
            primary = _ALIAS_SPLIT.split(name)[0]
            primary_names.append(primary)
        return sorted(primary_names)


def _enrich_usage_error(error: click.UsageError, label: str, items: list[tuple[str, str]]) -> None:
    """Append a list of available options or commands to a usage error message."""
    if not items or error.ctx is None or f"Available {label} for" in error.message:
        return
    cmd_path = error.ctx.command_path
    lines = [f"\n\nAvailable {label} for '{cmd_path}':"]
    for name, help_text in items:
        lines.append(f"  {name:30s} {help_text}")
    lines.append(f"\nRun '{cmd_path} --help' for full details.")
    if isinstance(error, click.NoSuchOption) and error.possibilities:
        lines.append(f"\nDid you mean: {', '.join(sorted(error.possibilities))}?")
        error.possibilities = []
    error.message += "\n".join(lines)


def fallback_typer_group_factory(
    fallback_handler: FallbackHandlerT,
    extra_commands_provider: Callable[[], list[tuple[str, str]]] | None = None,
) -> type[HFCliTyperGroup]:
    """Return a Typer group class that runs a fallback handler before command resolution."""

    class FallbackTyperGroup(HFCliTyperGroup):
        def resolve_command(self, ctx: click.Context, args: list[str]) -> tuple:
            fallback_exit_code = fallback_handler(args, set(self.commands.keys()))
            if fallback_exit_code is not None:
                raise SystemExit(fallback_exit_code)
            return super().resolve_command(ctx, args)

        def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
            super().format_commands(ctx, formatter)
            if extra_commands_provider is not None:
                entries = extra_commands_provider()
                if entries:
                    with formatter.section("Extension commands"):
                        formatter.write_dl(entries)

    return FallbackTyperGroup


def HFCliCommand(topic: TOPIC_T, examples: list[str] | None = None) -> type[TyperCommand]:
    def format_epilog(self: click.Command, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _format_epilog_no_indent(self.epilog, ctx, formatter)

    return type(
        f"TyperCommand{topic.capitalize()}",
        (TyperCommand,),
        {"topic": topic, "examples": examples or [], "format_epilog": format_epilog},
    )


class HFCliApp(typer.Typer):
    """Custom Typer app for Hugging Face CLI."""

    def command(  # type: ignore
        self,
        name: str | None = None,
        *,
        topic: TOPIC_T = "main",
        examples: list[str] | None = None,
        context_settings: dict[str, Any] | None = None,
        help: str | None = None,
        epilog: str | None = None,
        short_help: str | None = None,
        options_metavar: str = "[OPTIONS]",
        add_help_option: bool = True,
        no_args_is_help: bool = False,
        hidden: bool = False,
        deprecated: bool = False,
        rich_help_panel: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        # Generate epilog from examples if not explicitly provided
        if epilog is None and examples:
            epilog = generate_epilog(examples)

        def _inner(func: Callable[..., Any]) -> Callable[..., Any]:
            return super(HFCliApp, self).command(
                name,
                cls=HFCliCommand(topic, examples),
                context_settings=context_settings,
                help=help,
                epilog=epilog,
                short_help=short_help,
                options_metavar=options_metavar,
                add_help_option=add_help_option,
                no_args_is_help=no_args_is_help,
                hidden=hidden,
                deprecated=deprecated,
                rich_help_panel=rich_help_panel,
            )(func)

        return _inner


def typer_factory(help: str, epilog: str | None = None, cls: type[TyperGroup] | None = None) -> "HFCliApp":
    """Create a Typer app with consistent settings.

    Args:
        help: Help text for the app.
        epilog: Optional epilog text (use `generate_epilog` to create one).
        cls: Optional Click group class to use (defaults to `HFCliTyperGroup`).

    Returns:
        A configured Typer app.
    """
    if cls is None:
        cls = HFCliTyperGroup
    return HFCliApp(
        help=help,
        epilog=epilog,
        add_completion=True,
        no_args_is_help=True,
        cls=cls,
        # Disable rich completely for consistent experience
        rich_markup_mode=None,
        rich_help_panel=None,
        pretty_exceptions_enable=False,
        # Disable TyperGroup's suggest_commands, it matches against raw aliased
        # keys ("list | ls") leaking pipe syntax into user-facing messages.
        # HFCliTyperGroup.resolve_command() handles suggestions with expanded names.
        suggest_commands=False,
        # Increase max content width for better readability
        context_settings={
            "max_content_width": 120,
            "help_option_names": ["-h", "--help"],
        },
    )


class RepoType(str, Enum):
    model = "model"
    dataset = "dataset"
    space = "space"


RepoIdArg = Annotated[
    str,
    typer.Argument(
        help="The ID of the repo (e.g. `username/repo-name` or `spaces/username/repo-name`).",
    ),
]


RepoTypeOpt = Annotated[
    RepoType,
    typer.Option(
        "--type",
        "--repo-type",
        help="The type of repository (model, dataset, or space).",
    ),
]

TokenOpt = Annotated[
    str | None,
    typer.Option(
        help="A User Access Token generated from https://huggingface.co/settings/tokens.",
    ),
]

PrivateOpt = Annotated[
    bool | None,
    typer.Option(
        help="Whether to create a private repo if repo doesn't exist on the Hub. Ignored if the repo already exists.",
    ),
]

RevisionOpt = Annotated[
    str | None,
    typer.Option(
        help="Git revision id which can be a branch name, a tag, or a commit hash.",
    ),
]


LimitOpt = Annotated[
    int,
    typer.Option(help="Limit the number of results."),
]

AuthorOpt = Annotated[
    str | None,
    typer.Option(help="Filter by author or organization."),
]

FilterOpt = Annotated[
    list[str] | None,
    typer.Option(help="Filter by tags (e.g. 'text-classification'). Can be used multiple times."),
]

SearchOpt = Annotated[
    str | None,
    typer.Option(help="Search query."),
]


# --- Env / Secrets shared options and parsing helpers (used by jobs, repos, etc.) ---

EnvOpt = Annotated[
    list[str] | None,
    typer.Option(
        "-e",
        "--env",
        help="Set environment variables. E.g. --env ENV=value",
    ),
]

SecretsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "-s",
        "--secrets",
        help=(
            "Set secret environment variables. E.g. --secrets SECRET=value"
            " or `--secrets HF_TOKEN` to pass your Hugging Face token."
        ),
    ),
]

EnvFileOpt = Annotated[
    str | None,
    typer.Option(
        "--env-file",
        help="Read in a file of environment variables.",
    ),
]

SecretsFileOpt = Annotated[
    str | None,
    typer.Option(
        help="Read in a file of secret environment variables.",
    ),
]


def _get_extended_environ() -> dict[str, str]:
    """Return a copy of ``os.environ`` with the user's HF token injected (if available)."""
    from huggingface_hub import get_token

    extended_environ = os.environ.copy()
    if (token := get_token()) is not None:
        extended_environ["HF_TOKEN"] = token
    return extended_environ


def parse_env_map(
    env: list[str] | None = None,
    env_file: str | None = None,
) -> dict[str, str | None]:
    """Parse ``-e``/``--env``/``-s``/``--secrets`` and ``--env-file``/``--secrets-file`` CLI args into a dict.

    Uses an extended environment that includes the user's HF token so that
    bare ``--secrets HF_TOKEN`` resolves correctly.
    """
    extended_environ = _get_extended_environ()
    env_map: dict[str, str | None] = {}
    if env_file:
        env_map.update(load_dotenv(Path(env_file).read_text(), environ=extended_environ))
    for env_value in env or []:
        env_map.update(load_dotenv(env_value, environ=extended_environ))
    return env_map


def env_map_to_key_value_list(env_map: dict[str, str | None]) -> list[dict[str, str]] | None:
    """Convert an env/secrets dict to the ``[{"key": ..., "value": ...}]`` format used by the Hub API."""
    if not env_map:
        return None
    return [{"key": k, "value": v or ""} for k, v in env_map.items()]


VolumesOpt = Annotated[
    list[str] | None,
    typer.Option(
        "-v",
        "--volume",
        help="Mount one or more volumes. Format: hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]. "
        "TYPE is one of: models, datasets, spaces, buckets. "
        "TYPE defaults to models if omitted. "
        "models, datasets and spaces are always mounted read-only. buckets are read+write by default. "
        "E.g. -v hf://gpt2:/data or -v hf://datasets/org/ds:/data or -v hf://buckets/org/b:/mnt:ro",
    ),
]

_HF_PREFIX = "hf://"
_HF_VOLUME_TYPES = {
    "models": constants.REPO_TYPE_MODEL,
    "datasets": constants.REPO_TYPE_DATASET,
    "spaces": constants.REPO_TYPE_SPACE,
    "buckets": "bucket",
}


def parse_volumes(volumes: list[str] | None) -> "list[Volume] | None":
    """Parse volume specs from CLI arguments.

    Format: hf://[TYPE/]SOURCE[/PATH]:/MOUNT_PATH[:ro|:rw]
    Where TYPE is one of: models, datasets, spaces, buckets (defaults to models if omitted).
    SOURCE is the repo/bucket identifier (e.g. 'username/my-model').
    PATH is an optional subfolder inside the repo/bucket.
    MOUNT_PATH starts with '/'.
    Optional ':ro' or ':rw' suffix for read-only or read-write.

    Examples:
        hf://gpt2:/data                          (model, implicit type)
        hf://my-org/my-model:/data                (model, implicit type)
        hf://models/my-org/my-model:/data         (model, explicit type)
        hf://datasets/my-org/my-dataset:/data:ro
        hf://buckets/my-org/my-bucket:/mnt
        hf://spaces/my-org/my-space:/app
        hf://datasets/org/ds/train:/data          (with path inside repo)
        hf://buckets/org/b/sub/dir:/mnt           (with path inside bucket)
    """

    if not volumes:
        return None

    result: list[Volume] = []
    for raw_spec in volumes:
        # Strip :ro/:rw suffix
        spec = raw_spec
        read_only = None
        if spec.endswith(":ro"):
            read_only = True
            spec = spec[:-3]
        elif spec.endswith(":rw"):
            read_only = False
            spec = spec[:-3]

        # Validate hf:// prefix
        if not spec.startswith(_HF_PREFIX):
            raise CLIError(
                f"Invalid volume format: '{raw_spec}'. Source must start with 'hf://'. "
                f"Expected hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]. E.g. hf://gpt2:/data"
            )
        spec = spec[len(_HF_PREFIX) :]

        # Find the mount path: look for :/ pattern
        colon_slash_idx = spec.find(":/")
        if colon_slash_idx == -1:
            raise CLIError(
                f"Invalid volume format: '{raw_spec}'. Expected hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]. E.g. hf://gpt2:/data"
            )
        source_part = spec[:colon_slash_idx]
        mount_path = spec[colon_slash_idx + 1 :]

        # Parse type from source_part (first segment before /)
        # Then split remaining into source (namespace/name or name) and optional path.
        slash_idx = source_part.find("/")
        if slash_idx == -1:
            # No slash: bare source like "gpt2" -> model type
            vol_type_str = constants.REPO_TYPE_MODEL
            source = source_part
            path = None
        else:
            first_segment = source_part[:slash_idx]
            if first_segment in _HF_VOLUME_TYPES:
                vol_type_str = _HF_VOLUME_TYPES[first_segment]
                remaining = source_part[slash_idx + 1 :]
            else:
                # First segment isn't a known type -> model type
                vol_type_str = constants.REPO_TYPE_MODEL
                remaining = source_part

            # Split remaining into source (namespace/name) and optional path.
            # Repo/bucket IDs are "namespace/name" (2 segments) or "name" (1 segment).
            # Any extra segments are the path inside the repo/bucket.
            parts = remaining.split("/", 2)
            if len(parts) >= 3:
                source = parts[0] + "/" + parts[1]
                path = parts[2]
            else:
                source = remaining
                path = None

        result.append(
            Volume(
                type=vol_type_str,
                source=source,
                mount_path=mount_path,
                read_only=read_only,
                path=path,
            )
        )
    return result


class OutputFormat(str, Enum):
    """Output format for CLI list commands."""

    table = "table"
    json = "json"


FormatOpt = Annotated[
    OutputFormat,
    typer.Option(
        help="Output format (table or json).",
    ),
]


def _set_output_mode(value: OutputFormatWithAuto) -> OutputFormatWithAuto:
    out.set_mode(value)
    if out.mode != OutputFormatWithAuto.human:
        disable_progress_bars()
    return value


FormatWithAutoOpt = Annotated[
    OutputFormatWithAuto,
    typer.Option(
        help="Output format.",
        callback=_set_output_mode,
    ),
]

QuietOpt = Annotated[
    bool,
    typer.Option(
        "-q",
        "--quiet",
        help="Print only IDs (one per line).",
    ),
]


def _to_header(name: str) -> str:
    """Convert a camelCase or PascalCase string to SCREAMING_SNAKE_CASE to be used as table header."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return s.upper()


def _format_value(value: Any) -> str:
    """Convert a value to string for terminal display."""
    if not value:
        return ""
    if isinstance(value, bool):
        return "✔" if value else ""
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return value[:10]
    if isinstance(value, list):
        return ", ".join(_format_value(v) for v in value)
    elif isinstance(value, dict):
        if "name" in value:  # Likely to be a user or org => print name
            return str(value["name"])
        # TODO: extend if needed
        return json.dumps(value)
    return str(value)


def _format_cell(value: Any, max_len: int = _MAX_CELL_LENGTH) -> str:
    """Format a value + truncate it for table display."""
    cell = _format_value(value)
    if len(cell) > max_len:
        cell = cell[: max_len - 3] + "..."
    return cell


def print_as_table(
    items: Sequence[dict[str, Any]],
    headers: list[str],
    row_fn: Callable[[dict[str, Any]], list[str]],
    alignments: dict[str, str] | None = None,
) -> None:
    """Print items as a formatted table.

    Args:
        items: Sequence of dictionaries representing the items to display.
        headers: List of column headers.
        row_fn: Function that takes an item dict and returns a list of string values for each column.
        alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
    """
    if not items:
        print("No results found.")
        return
    rows = cast(list[list[Union[str, int]]], [row_fn(item) for item in items])
    screaming_headers = [_to_header(h) for h in headers]
    # Remap alignments keys to screaming case to match tabulate headers
    screaming_alignments = {_to_header(k): v for k, v in (alignments or {}).items()}
    print(tabulate(rows, headers=screaming_headers, alignments=screaming_alignments))


def print_list_output(
    items: Sequence[dict[str, Any]],
    format: OutputFormat,
    quiet: bool,
    id_key: str = "id",
    headers: list[str] | None = None,
    row_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    alignments: dict[str, str] | None = None,
) -> None:
    """Print list command output in the specified format.

    Args:
        items: Sequence of dictionaries representing the items to display.
        format: Output format.
        quiet: If True, print only IDs (one per line).
        id_key: Key to use for extracting IDs in quiet mode.
        headers: Optional list of column names for headers. If not provided, auto-detected from keys.
        row_fn: Optional function to extract row values. If not provided, uses _format_cell on each column.
        alignments: Optional mapping of header name to "left" or "right". Defaults to "left".
    """
    if quiet:
        for item in items:
            print(item[id_key])
        return

    if format == OutputFormat.json:
        print(json.dumps(list(items), indent=2, default=str))
        return

    if headers is None:
        all_columns = list(items[0].keys()) if items else [id_key]
        headers = [col for col in all_columns if any(_format_cell(item.get(col)) for item in items)]

    if row_fn is None:

        def row_fn(item: dict[str, Any]) -> list[str]:
            return [_format_cell(item.get(col)) for col in headers]  # type: ignore[union-attr]

    print_as_table(items, headers=headers, row_fn=row_fn, alignments=alignments)


def _serialize_value(v: object) -> object:
    """Recursively serialize a value to be JSON-compatible."""
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    elif isinstance(v, dict):
        return {key: _serialize_value(val) for key, val in v.items() if val is not None}
    elif isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def api_object_to_dict(info: Any) -> dict[str, Any]:
    """Convert repo info dataclasses to json-serializable dicts."""
    return {k: _serialize_value(v) for k, v in dataclasses.asdict(info).items() if v is not None}


def make_expand_properties_parser(valid_properties: Sequence[ExpandPropertyT]):
    """Create a callback to parse and validate comma-separated expand properties."""

    def _parse_expand_properties(value: str | None) -> list[ExpandPropertyT] | None:
        if value is None:
            return None
        properties = [p.strip() for p in value.split(",")]
        for prop in properties:
            if prop not in valid_properties:
                raise typer.BadParameter(
                    f"Invalid expand property: '{prop}'. Valid values are: {', '.join(valid_properties)}"
                )
        return [cast(ExpandPropertyT, prop) for prop in properties]

    return _parse_expand_properties


### PyPI VERSION CHECKER


def check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    """
    Check whether a newer version of a library is available on PyPI.

    If a newer version is found and stdin/stderr are attached to a TTY, prompt the user to update interactively.
    Otherwise (non-TTY or update command cannot be determined), print a warning to stderr.

    If current version is a pre-release (e.g. `1.0.0.rc1`), or a dev version (e.g. `1.0.0.dev1`), no check is performed.

    This function is called at the entry point of the CLI. It only performs the check once every 24 hours, and any error
    during the check is caught and logged, to avoid breaking the CLI.

    Args:
        library: The library to check for updates. Currently supports "huggingface_hub" and "transformers".
    """
    try:
        _check_cli_update(library)
    except Exception:
        # We don't want the CLI to fail on version checks, no matter the reason.
        logger.debug("Error while checking for CLI update.", exc_info=True)


def _check_cli_update(library: Literal["huggingface_hub", "transformers"]) -> None:
    current_version = importlib.metadata.version(library)

    # Skip if current version is a pre-release or dev version
    if any(tag in current_version for tag in ["rc", "dev"]):
        return

    # Skip if already checked in the last 24 hours
    if os.path.exists(constants.CHECK_FOR_UPDATE_DONE_PATH):
        mtime = os.path.getmtime(constants.CHECK_FOR_UPDATE_DONE_PATH)
        if (time.time() - mtime) < 24 * 3600:
            return

    # Touch the file to mark that we did the check now
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(constants.CHECK_FOR_UPDATE_DONE_PATH).touch()

    # Check latest version from PyPI
    response = get_session().get(f"https://pypi.org/pypi/{library}/json", timeout=2)
    hf_raise_for_status(response)
    data = response.json()
    latest_version = data["info"]["version"]

    if current_version == latest_version:
        return

    if library == "huggingface_hub":
        update_command = _get_huggingface_hub_update_command()
    else:
        update_command = _get_transformers_update_command()

    if sys.stdin.isatty() and sys.stderr.isatty() and update_command is not None:
        _prompt_autoupdate(library, current_version, latest_version, update_command)
    else:
        display_cmd = " ".join(update_command) if update_command else None
        update_hint = f"To update, run: {ANSI.bold(display_cmd)}" if display_cmd else ""
        click.echo(
            ANSI.yellow(
                f"A new version of {library} ({latest_version}) is available! "
                f"You are using version {current_version}." + (f"\n{update_hint}" if update_hint else "") + "\n"
            ),
            file=sys.stderr,
        )


def _prompt_autoupdate(
    library: str,
    current_version: str,
    latest_version: str,
    update_command: list[str],
) -> None:
    """Interactively ask the user if they want to update, and run the update command if accepted.

    After a successful update the CLI exits so the user can re-run their command with the new version.
    All output goes to stderr to keep stdout clean for command output.
    """
    display_cmd = " ".join(update_command)

    click.echo("", file=sys.stderr)
    click.echo(
        ANSI.yellow(f"  A new version of {library} is available: {current_version} → {latest_version}"),
        file=sys.stderr,
    )
    click.echo("", file=sys.stderr)

    click.echo(
        ANSI.yellow("  Do you want to update now? [Y/n] ") + ANSI.gray(f"({display_cmd})") + " ",
        file=sys.stderr,
        nl=False,
    )
    try:
        raw_answer = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        click.echo("", file=sys.stderr)
        return

    if raw_answer == "":
        # EOF (e.g. Ctrl+D) — treat as cancellation, not acceptance
        click.echo("", file=sys.stderr)
        return

    answer = raw_answer.strip().lower()  # Note: if user press 'Enter', raw_answer is `\n`
    if answer in ("", "y", "yes"):
        click.echo("", file=sys.stderr)
        click.echo(ANSI.gray(f"  Running: {display_cmd}"), file=sys.stderr)
        click.echo("", file=sys.stderr)
        returncode = subprocess.call(update_command)
        if returncode == 0:
            click.echo("", file=sys.stderr)
            click.echo(
                ANSI.green(f"  ✓ Successfully updated {library} to {latest_version}. Please re-run your command."),
                file=sys.stderr,
            )
            raise SystemExit(0)
        else:
            click.echo("", file=sys.stderr)
            click.echo(
                ANSI.red(f"  ✗ Update failed (exit code {returncode}). Please update manually."),
                file=sys.stderr,
            )
    else:
        click.echo(
            ANSI.gray(f"  Skipped. You can update later with: {display_cmd}"),
            file=sys.stderr,
        )
    click.echo("", file=sys.stderr)


def _get_huggingface_hub_update_command() -> list[str] | None:
    """Return the command to update huggingface_hub as an argv list, or None if the installation method is unknown."""
    match installation_method():
        case "brew":
            return ["brew", "upgrade", "hf"]
        case "hf_installer" if os.name == "nt":
            return ["powershell", "-NoProfile", "-Command", "iwr -useb https://hf.co/cli/install.ps1 | iex"]
        case "hf_installer":
            return ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash -"]
        case "pip":
            return [sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"]
        case _:
            return None


def _get_transformers_update_command() -> list[str] | None:
    """Return the command to update transformers as an argv list, or None if the installation method is unknown."""
    match installation_method():
        case "hf_installer" if os.name == "nt":
            return [
                "powershell",
                "-NoProfile",
                "-Command",
                "iwr -useb https://hf.co/cli/install.ps1 | iex -WithTransformers",
            ]
        case "hf_installer":
            return ["bash", "-c", "curl -LsSf https://hf.co/cli/install.sh | bash -s -- --with-transformers"]
        case "pip":
            return [sys.executable, "-m", "pip", "install", "-U", "transformers"]
        case _:
            return None
