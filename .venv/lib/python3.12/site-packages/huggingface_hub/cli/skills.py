# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Contains commands to manage skills for AI assistants.

Usage:
    # install the hf-cli skill in common .agents/skills directory (either in current directory or user-level)
    hf skills add
    hf skills add --global

    # install the hf-cli skill for Claude (project-level, in current directory)
    hf skills add --claude

    # install globally (user-level)
    hf skills add --claude --global

    # install to a custom directory
    hf skills add --dest=~/my-skills

    # overwrite an existing skill
    hf skills add --claude --force
"""

import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from click import Command, Context, Group
from typer.main import get_command

from huggingface_hub.errors import CLIError

from . import _skills
from ._cli_utils import typer_factory


DEFAULT_SKILL_ID = "hf-cli"

_SKILL_DESCRIPTION = (
    "Hugging Face Hub CLI (`hf`) for downloading, uploading, and managing"
    " models, datasets, spaces, buckets, repos, papers, jobs, and more on the Hugging Face Hub."
    " Use when: handling authentication;"
    " managing local cache;"
    " managing Hugging Face Buckets;"
    " running or scheduling jobs on Hugging Face infrastructure;"
    " managing Hugging Face repos;"
    " discussions and pull requests;"
    " browsing models, datasets and spaces;"
    " reading, searching, or browsing academic papers;"
    " managing collections;"
    " querying datasets;"
    " configuring spaces;"
    " setting up webhooks;"
    " or deploying and managing HF Inference Endpoints."
    " Make sure to use this skill whenever the user mentions"
    " 'hf', 'huggingface', 'Hugging Face', 'huggingface-cli', or 'hugging face cli',"
    " or wants to do anything related to the Hugging Face ecosystem and to AI and ML in general."
    " Also use for cloud storage needs like training checkpoints, data pipelines, or agent traces."
    " Use even if the user doesn't explicitly ask for a CLI command."
    " Replaces the deprecated `huggingface-cli`."
)

_SKILL_YAML_PREFIX = f"""\
---
name: hf-cli
description: "{_SKILL_DESCRIPTION}"
---

Install: `curl -LsSf https://hf.co/cli/install.sh | bash -s`.

The Hugging Face Hub CLI tool `hf` is available. IMPORTANT: The `hf` command replaces the deprecated `huggingface-cli` command.

Use `hf --help` to view available functions. Note that auth commands are now all under `hf auth` e.g. `hf auth whoami`.
"""

_SKILL_TIPS = """
## Mounting repos as local filesystems

To mount Hub repositories or buckets as local filesystems — no download, no copy, no waiting — use `hf-mount`. Files are fetched on demand. GitHub: https://github.com/huggingface/hf-mount

Install: `curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh`

Some command examples:
- `hf-mount start repo openai-community/gpt2 /tmp/gpt2` — mount a repo (read-only)
- `hf-mount start --hf-token $HF_TOKEN bucket myuser/my-bucket /tmp/data` — mount a bucket (read-write)
- `hf-mount status` / `hf-mount stop /tmp/data` — list or unmount

## Tips

- Use `hf <command> --help` for full options, descriptions, usage, and real-world examples
- Authenticate with `HF_TOKEN` env var (recommended) or with `--token`
"""

CENTRAL_LOCAL = Path(".agents/skills")
CENTRAL_GLOBAL = Path("~/.agents/skills")
CLAUDE_LOCAL = Path(".claude/skills")
CLAUDE_GLOBAL = Path("~/.claude/skills")
# Flags worth explaining in the common-options glossary. Self-explanatory flags
# (--namespace, --yes, --private, …) are omitted even if they appear frequently.
_COMMON_FLAG_ALLOWLIST = {"--token", "--quiet", "--type", "--format", "--revision"}
# Keep token out of inline command signatures to encourage env based auth.
_INLINE_FLAG_EXCLUDE = {"--token"}

_COMMON_FLAG_HELP_OVERRIDES: dict[str, str] = {
    "--format": "Output format: `--format json` (or `--json`) or `--format table` (default).",
    "--token": "Use a User Access Token. Prefer setting `HF_TOKEN` env var instead of passing `--token`.",
}

skills_cli = typer_factory(help="Manage skills for AI assistants.")


def _format_params(cmd: Command) -> str:
    """Format required params: positional as UPPER_CASE, options as ``--name TYPE``."""
    parts = []
    for p in cmd.params:
        if not p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = next((o for o in getattr(p, "opts", []) if o.startswith("--")), None)
        if long_name is not None:
            type_name = getattr(p.type, "name", "").upper() or "VALUE"
            parts.append(f"{long_name} {type_name}")
        elif p.name:
            parts.append(p.human_readable_name)
    return " ".join(parts)


def _collect_leaf_commands(group: Group, ctx: Context, path_parts: list[str]) -> list[tuple[list[str], Command]]:
    """Recursively walk a Click Group, returning (full_path_parts, cmd) for every leaf command."""
    leaves: list[tuple[list[str], Command]] = []
    sub_ctx = Context(group, parent=ctx, info_name=path_parts[-1])
    for name in group.list_commands(sub_ctx):
        cmd = group.get_command(sub_ctx, name)
        if cmd is None or cmd.hidden:
            continue
        child_path = [*path_parts, name]
        if isinstance(cmd, Group):
            leaves.extend(_collect_leaf_commands(cmd, sub_ctx, child_path))
        else:
            leaves.append((child_path, cmd))
    return leaves


def _iter_optional_params(cmd: Command):
    """Yield (param, long_name, short_name) for each optional, non-internal param."""
    for p in cmd.params:
        if p.required or p.human_readable_name == "--help":
            continue
        if p.name and p.name.startswith("_"):
            continue
        long_name = None
        short_name = None
        for opt in getattr(p, "opts", []):
            if opt.startswith("--"):
                long_name = long_name or opt
            elif opt.startswith("-"):
                short_name = opt
        if long_name:
            yield p, long_name, short_name


def _get_flag_names(cmd: Command, *, exclude: set[str] | None = None) -> list[str]:
    """Return long-form flag names (--foo) for optional, non-internal params.

    Boolean flags are bare (``--dry-run``).  Value-taking options include a
    type hint (``--include TEXT``, ``--max-workers INTEGER``).
    """
    flags: list[str] = []
    for p, long_name, _short in _iter_optional_params(cmd):
        if exclude and long_name in exclude:
            continue
        if getattr(p, "is_flag", False):
            flags.append(long_name)
        else:
            type_name = getattr(p.type, "name", "").upper() or "VALUE"
            flags.append(f"{long_name} {type_name}")
    return flags


def _compute_common_flags(
    leaf_commands: list[tuple[list[str], Command]],
) -> dict[str, tuple[str, str]]:
    """Collect display info for flags in the allowlist."""
    flag_info: dict[str, tuple[str, str]] = {}

    for _path, cmd in leaf_commands:
        for p, long_name, short_name in _iter_optional_params(cmd):
            if long_name not in _COMMON_FLAG_ALLOWLIST:
                continue
            # Prefer the version with a short form (e.g. "-q / --quiet" over just "--quiet")
            if long_name not in flag_info or (short_name and " / " not in flag_info[long_name][0]):
                display = f"{short_name} / {long_name}" if short_name else long_name
                help_text = (getattr(p, "help", None) or "").split("\n")[0].strip()
                flag_info[long_name] = (display, help_text)

    return flag_info


def _render_leaf(path_parts: list[str], cmd: Command) -> str:
    """Render a single leaf command as a markdown list entry."""
    help_text = (cmd.help or "").split("\n")[0].strip()
    params = _format_params(cmd)
    parts = ["hf", *path_parts] + ([params] if params else [])
    entry = f"- `{' '.join(parts)}` — {help_text}"
    flags = _get_flag_names(cmd, exclude=_INLINE_FLAG_EXCLUDE)
    if flags:
        entry += f" `[{' '.join(flags)}]`"
    return entry


def build_skill_md() -> str:
    # Lazy import to avoid circular dependency (hf.py imports skills_cli from this module)
    from huggingface_hub import __version__
    from huggingface_hub.cli.hf import app

    click_app = get_command(app)
    ctx = Context(click_app, info_name="hf")

    top_level: list[tuple[list[str], Command]] = []
    groups: list[tuple[str, Group]] = []
    for name in sorted(click_app.list_commands(ctx)):  # type: ignore[attr-defined]
        cmd = click_app.get_command(ctx, name)  # type: ignore[attr-defined]
        if cmd is None or cmd.hidden:
            continue
        if isinstance(cmd, Group):
            groups.append((name, cmd))
        else:
            top_level.append(([name], cmd))

    group_leaves: list[tuple[str, list[tuple[list[str], Command]]]] = []
    all_leaf_commands: list[tuple[list[str], Command]] = list(top_level)
    for name, group in groups:
        leaves = _collect_leaf_commands(group, ctx, [name])
        group_leaves.append((name, leaves))
        all_leaf_commands.extend(leaves)

    common_flags = _compute_common_flags(all_leaf_commands)

    # wrap in list to widen list[LiteralString] -> list[str] for `ty`
    lines: list[str] = list(_SKILL_YAML_PREFIX.splitlines())
    lines.append("")
    lines.append(f"Generated with `huggingface_hub v{__version__}`. Run `hf skills add --force` to regenerate.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")

    for path_parts, cmd in top_level:
        lines.append(_render_leaf(path_parts, cmd))

    groups_dict = dict(groups)
    for name, leaves in group_leaves:
        group_cmd = groups_dict[name]
        help_text = (group_cmd.help or "").split("\n")[0].strip()
        lines.append("")
        lines.append(f"### `hf {name}` — {help_text}")
        lines.append("")
        for path_parts, cmd in leaves:
            lines.append(_render_leaf(path_parts, cmd))

    if common_flags:
        lines.append("")
        lines.append("## Common options")
        lines.append("")
        for long_name, (display, help_text) in sorted(common_flags.items()):
            help_text = _COMMON_FLAG_HELP_OVERRIDES.get(long_name, help_text)
            if help_text:
                lines.append(f"- `{display}` — {help_text}")
            else:
                lines.append(f"- `{display}`")

    lines.extend(_SKILL_TIPS.splitlines())

    return "\n".join(lines)


def _remove_existing(path: Path, force: bool) -> None:
    """Remove existing file/directory/symlink if force is True, otherwise raise an error."""
    if not (path.exists() or path.is_symlink()):
        return
    if not force:
        raise CLIError(f"Skill already exists at {path}.\nRe-run with --force to overwrite.")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_to(skills_dir: Path, skill_name: str, force: bool) -> Path:
    """Install a marketplace skill into a skills directory. Returns the installed path."""
    skill = _skills.get_marketplace_skill(skill_name)
    try:
        return _skills.install_marketplace_skill(skill, skills_dir, force=force)
    except FileExistsError as exc:
        raise CLIError(f"{exc}\nRe-run with --force to overwrite.") from exc


def _create_symlink(agent_skills_dir: Path, skill_name: str, central_skill_path: Path, force: bool) -> Path:
    """Create a relative symlink from agent directory to the central skill location."""
    agent_skills_dir = agent_skills_dir.expanduser().resolve()
    agent_skills_dir.mkdir(parents=True, exist_ok=True)
    link_path = agent_skills_dir / skill_name

    _remove_existing(link_path, force)
    link_path.symlink_to(os.path.relpath(central_skill_path, agent_skills_dir))

    return link_path


def _resolve_update_roots(
    *,
    claude: bool,
    global_: bool,
    dest: Path | None,
) -> list[Path]:
    if dest is not None:
        if claude or global_:
            raise CLIError("--dest cannot be combined with --claude or --global.")
        return [dest.expanduser().resolve()]

    roots: list[Path] = [CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL]
    if claude:
        roots.append(CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL)
    return [root.expanduser().resolve() for root in roots]


@skills_cli.command("preview")
def skills_preview() -> None:
    """Print the generated `hf-cli` SKILL.md to stdout."""
    print(build_skill_md())


@skills_cli.command(
    "add",
    examples=[
        "hf skills add",
        "hf skills add huggingface-gradio --dest=~/my-skills",
        "hf skills add --global",
        "hf skills add --claude",
        "hf skills add huggingface-gradio --claude --global",
    ],
)
def skills_add(
    name: Annotated[
        str,
        typer.Argument(help="Marketplace skill name.", show_default=False),
    ] = DEFAULT_SKILL_ID,
    claude: Annotated[bool, typer.Option("--claude", help="Install for Claude.")] = False,
    global_: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Install globally (user-level) instead of in the current project directory.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        typer.Option(
            help="Install into a custom destination (path to skills directory).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing skills in the destination.",
        ),
    ] = False,
) -> None:
    """Download a Hugging Face skill and install it for an AI assistant.

    Default location is in the current directory (.agents/skills) or user-level (~/.agents/skills).
    If `--claude` is specified, the skill is also symlinked into Claude's legacy skills directory.
    """
    if dest is not None:
        if claude or global_:
            raise CLIError("--dest cannot be combined with --claude or --global.")
        skill_dest = _install_to(dest, name, force)
        print(f"Installed '{name}' to {skill_dest}")
        return

    # Install to central location
    central_path = CENTRAL_GLOBAL if global_ else CENTRAL_LOCAL
    central_skill_path = _install_to(central_path, name, force)
    print(f"Installed '{name}' to central location: {central_skill_path}")

    if claude:
        agent_target = CLAUDE_GLOBAL if global_ else CLAUDE_LOCAL
        link_path = _create_symlink(agent_target, name, central_skill_path, force)
        print(f"Created symlink: {link_path}")


@skills_cli.command(
    "upgrade",
    examples=[
        "hf skills upgrade",
        "hf skills upgrade hf-cli",
        "hf skills upgrade huggingface-gradio --dest=~/my-skills",
        "hf skills upgrade --claude",
    ],
)
def skills_upgrade(
    name: Annotated[
        str | None,
        typer.Argument(help="Optional installed skill name to upgrade.", show_default=False),
    ] = None,
    claude: Annotated[bool, typer.Option("--claude", help="Upgrade skills installed for Claude.")] = False,
    global_: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Use global skills directories instead of the current project.",
        ),
    ] = False,
    dest: Annotated[
        Path | None,
        typer.Option(
            help="Upgrade skills in a custom skills directory.",
        ),
    ] = None,
) -> None:
    """Upgrade installed Hugging Face marketplace skills."""
    roots = _resolve_update_roots(claude=claude, global_=global_, dest=dest)

    results = _skills.apply_updates(roots, selector=name)
    if not results:
        print("No installed skills found.")
        return

    for result in results:
        detail = f" ({result.detail})" if result.detail else ""
        print(f"{result.name}: {result.status}{detail}")
