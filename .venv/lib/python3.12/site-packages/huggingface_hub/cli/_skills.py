"""Internal helpers for Hugging Face marketplace skill installation and upgrades."""

import base64
import io
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from huggingface_hub.errors import CLIError
from huggingface_hub.utils import get_session


DEFAULT_SKILLS_REPO_ID = "huggingface/skills"
DEFAULT_SKILLS_REPO_OWNER, DEFAULT_SKILLS_REPO_NAME = DEFAULT_SKILLS_REPO_ID.split("/")
DEFAULT_SKILLS_REF = "main"
MARKETPLACE_PATH = ".claude-plugin/marketplace.json"
GITHUB_API_TIMEOUT = 10
SKILL_MANIFEST_FILENAME = ".hf-skill-manifest.json"
SKILL_MANIFEST_SCHEMA_VERSION = 1

SkillUpdateStatus = Literal[
    "up_to_date",
    "update_available",
    "updated",
    "unmanaged",
    "invalid_metadata",
    "source_unreachable",
]


@dataclass(frozen=True)
class MarketplaceSkill:
    name: str
    repo_path: str


@dataclass(frozen=True)
class InstalledSkillManifest:
    schema_version: int
    installed_revision: str


@dataclass(frozen=True)
class SkillUpdateInfo:
    name: str
    skill_dir: Path
    status: SkillUpdateStatus
    detail: str | None = None
    current_revision: str | None = None
    available_revision: str | None = None


def load_marketplace_skills() -> list[MarketplaceSkill]:
    """Load skills from the default Hugging Face marketplace."""
    payload = _load_marketplace_payload()
    plugins = payload.get("plugins")
    if not isinstance(plugins, list):
        raise CLIError("Invalid marketplace payload: expected a top-level 'plugins' list.")

    skills: list[MarketplaceSkill] = []
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        name = plugin.get("name")
        source = plugin.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        skills.append(MarketplaceSkill(name=name, repo_path=_normalize_repo_path(source)))
    return skills


def get_marketplace_skill(selector: str) -> MarketplaceSkill:
    """Resolve a marketplace skill by name."""
    selected = _select_marketplace_skill(load_marketplace_skills(), selector)
    if selected is None:
        raise CLIError(
            f"Skill '{selector}' not found in {DEFAULT_SKILLS_REPO_ID}. "
            "Try `hf skills add` to install `hf-cli` or use a known skill name."
        )
    return selected


def install_marketplace_skill(skill: MarketplaceSkill, destination_root: Path, force: bool = False) -> Path:
    """Install a marketplace skill into a local skills directory."""
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    install_dir = destination_root / skill.name

    if install_dir.exists() and not force:
        raise FileExistsError(f"Skill already exists: {install_dir}")

    if install_dir.exists():
        with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{install_dir.name}.install-") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            staged_dir = tmp_dir / install_dir.name
            _populate_install_dir(skill=skill, install_dir=staged_dir)
            _atomic_replace_directory(existing_dir=install_dir, staged_dir=staged_dir)
        return install_dir

    try:
        _populate_install_dir(skill=skill, install_dir=install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


def check_for_updates(
    roots: list[Path],
    selector: str | None = None,
) -> list[SkillUpdateInfo]:
    """Check managed skill installs for newer upstream revisions."""
    marketplace_skills = {skill.name.lower(): skill for skill in load_marketplace_skills()}
    updates = [_evaluate_update(skill_dir, marketplace_skills) for skill_dir in _iter_unique_skill_dirs(roots)]
    filtered = _filter_updates(updates, selector)
    if selector is not None and not filtered:
        raise CLIError(f"No installed skills match '{selector}'.")
    return filtered


def apply_updates(
    roots: list[Path],
    selector: str | None = None,
) -> list[SkillUpdateInfo]:
    """Upgrade managed skills in place when the upstream revision changes."""
    updates = check_for_updates(roots, selector)
    results: list[SkillUpdateInfo] = []
    for update in updates:
        results.append(_apply_single_update(update))
    return results


def read_installed_skill_manifest(skill_dir: Path) -> tuple[InstalledSkillManifest | None, str | None]:
    """Read local skill metadata written by `hf skills add`."""
    manifest_path = skill_dir / SKILL_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "metadata root must be an object"
    try:
        return _parse_installed_skill_manifest(payload), None
    except ValueError as exc:
        return None, str(exc)


def write_installed_skill_manifest(skill_dir: Path, manifest: InstalledSkillManifest) -> None:
    payload = {
        "schema_version": manifest.schema_version,
        "installed_revision": manifest.installed_revision,
    }
    (skill_dir / SKILL_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_marketplace_payload() -> dict[str, Any]:
    response = _fetch_from_skills_repo(
        f"contents/{MARKETPLACE_PATH}",
        params={"ref": DEFAULT_SKILLS_REF},
    )
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise CLIError(f"Failed to decode GitHub API response for 'contents/{MARKETPLACE_PATH}': {exc}") from exc
    if not isinstance(payload, dict):
        raise CLIError("Invalid marketplace response: expected a JSON object.")

    content = payload.get("content")
    encoding = payload.get("encoding")
    if not isinstance(content, str) or encoding != "base64":
        raise CLIError("Invalid marketplace payload: expected base64-encoded content.")

    try:
        decoded = base64.b64decode(content).decode("utf-8")
        parsed = json.loads(decoded)
    except Exception as exc:  # noqa: BLE001
        raise CLIError(f"Failed to decode marketplace payload: {exc}") from exc

    if not isinstance(parsed, dict):
        raise CLIError("Invalid marketplace payload: expected a JSON object.")
    return parsed


def _select_marketplace_skill(skills: list[MarketplaceSkill], selector: str) -> MarketplaceSkill | None:
    selector_lower = selector.strip().lower()
    for skill in skills:
        if skill.name.lower() == selector_lower:
            return skill
    return None


def _normalize_repo_path(path: str) -> str:
    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    if not normalized:
        raise CLIError("Invalid marketplace entry: empty source path.")
    return normalized


def _populate_install_dir(skill: MarketplaceSkill, install_dir: Path) -> None:
    installed_revision = _resolve_available_revision(skill)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_remote_github_path(
        revision=installed_revision,
        source_path=skill.repo_path,
        install_dir=install_dir,
    )
    _validate_installed_skill_dir(install_dir)
    write_installed_skill_manifest(
        install_dir,
        InstalledSkillManifest(
            schema_version=SKILL_MANIFEST_SCHEMA_VERSION,
            installed_revision=installed_revision,
        ),
    )


def _validate_installed_skill_dir(skill_dir: Path) -> None:
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise RuntimeError(f"Installed skill is missing SKILL.md: {skill_file}")


def _extract_remote_github_path(revision: str, source_path: str, install_dir: Path) -> None:
    tar_bytes = _fetch_from_skills_repo(f"tarball/{revision}").content
    _extract_tar_subpath(tar_bytes, source_path=source_path, install_dir=install_dir)


def _extract_tar_subpath(tar_bytes: bytes, source_path: str, install_dir: Path) -> None:
    """Extract a skill subdirectory from a tar archive.

    GitHub tarballs include a leading `<repo>-<revision>/` directory. The helper also
    accepts archives that start directly at `skills/<name>/...` to keep tests simple.
    """
    source_parts = PurePosixPath(source_path).parts
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
        members = archive.getmembers()
        matched = False
        for member in members:
            relative_parts = _member_relative_parts(member_name=member.name, source_parts=source_parts)
            if relative_parts is None:
                continue
            if not relative_parts:
                matched = True
                continue
            matched = True
            relative_path = Path(*relative_parts)
            if ".." in relative_path.parts:
                raise RuntimeError(f"Invalid path found in archive for {source_path}.")
            destination_path = install_dir / relative_path
            if member.isdir():
                destination_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {member.name}.")
            destination_path.write_bytes(extracted.read())
    if not matched:
        raise FileNotFoundError(f"Path '{source_path}' not found in source archive.")


def _member_relative_parts(member_name: str, source_parts: tuple[str, ...]) -> tuple[str, ...] | None:
    path_parts = PurePosixPath(member_name).parts
    if tuple(path_parts[: len(source_parts)]) == source_parts:
        return path_parts[len(source_parts) :]
    if len(path_parts) > len(source_parts) and tuple(path_parts[1 : 1 + len(source_parts)]) == source_parts:
        return path_parts[1 + len(source_parts) :]
    return None


def _atomic_replace_directory(existing_dir: Path, staged_dir: Path) -> None:
    backup_dir = staged_dir.parent / f"{existing_dir.name}.backup"
    try:
        existing_dir.rename(backup_dir)
        staged_dir.rename(existing_dir)
        shutil.rmtree(backup_dir)
    except Exception:
        if backup_dir.exists() and not existing_dir.exists():
            backup_dir.rename(existing_dir)
        raise


def _iter_unique_skill_dirs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    discovered: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name.startswith("."):
                continue
            if not child.is_dir() and not child.is_symlink():
                continue
            resolved = child.resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _evaluate_update(skill_dir: Path, marketplace_skills: dict[str, MarketplaceSkill]) -> SkillUpdateInfo:
    base = SkillUpdateInfo(name=skill_dir.name, skill_dir=skill_dir, status="unmanaged")

    manifest, error = read_installed_skill_manifest(skill_dir)
    if manifest is None:
        return replace(base, status="invalid_metadata" if error else "unmanaged", detail=error)

    skill = marketplace_skills.get(skill_dir.name.lower())
    if skill is None:
        return replace(
            base,
            status="source_unreachable",
            detail=f"Skill '{skill_dir.name}' is no longer available in {DEFAULT_SKILLS_REPO_ID}.",
            current_revision=manifest.installed_revision,
        )

    current_revision = manifest.installed_revision
    try:
        available_revision = _resolve_available_revision(skill)
    except Exception as exc:
        return replace(base, status="source_unreachable", detail=str(exc), current_revision=current_revision)

    status: SkillUpdateStatus = "up_to_date" if available_revision == current_revision else "update_available"
    return replace(
        base,
        status=status,
        detail="update available" if status == "update_available" else None,
        current_revision=current_revision,
        available_revision=available_revision,
    )


def _apply_single_update(update: SkillUpdateInfo) -> SkillUpdateInfo:
    if update.status != "update_available":
        return update

    try:
        skill = get_marketplace_skill(update.skill_dir.name)
        install_marketplace_skill(skill, update.skill_dir.parent, force=True)
    except Exception as exc:
        return replace(update, status="source_unreachable", detail=str(exc))

    return replace(update, status="updated", detail="updated")


def _filter_updates(updates: list[SkillUpdateInfo], selector: str | None) -> list[SkillUpdateInfo]:
    if selector is None:
        return updates
    selector_lower = selector.strip().lower()
    return [update for update in updates if update.name.lower() == selector_lower]


def _resolve_available_revision(skill: MarketplaceSkill) -> str:
    response = _fetch_from_skills_repo(
        "commits",
        params={"sha": DEFAULT_SKILLS_REF, "path": skill.repo_path, "per_page": 1},
    )
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise CLIError(f"Failed to decode GitHub API response for 'commits': {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise CLIError(f"Unable to resolve the current revision for skill '{skill.name}'.")

    latest = payload[0]
    if not isinstance(latest, dict):
        raise CLIError(f"Invalid commit response while resolving skill '{skill.name}'.")

    revision = latest.get("sha")
    if not isinstance(revision, str) or not revision:
        raise CLIError(f"Invalid commit response while resolving skill '{skill.name}'.")
    return revision


def _parse_installed_skill_manifest(payload: dict[str, Any]) -> InstalledSkillManifest:
    if payload.get("schema_version") != SKILL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {payload.get('schema_version')}")

    installed_revision = payload.get("installed_revision")
    if not isinstance(installed_revision, str) or not installed_revision:
        raise ValueError("missing installed_revision")

    return InstalledSkillManifest(
        schema_version=SKILL_MANIFEST_SCHEMA_VERSION,
        installed_revision=installed_revision,
    )


def _fetch_from_skills_repo(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    url = f"https://api.github.com/repos/{DEFAULT_SKILLS_REPO_OWNER}/{DEFAULT_SKILLS_REPO_NAME}/{endpoint.lstrip('/')}"
    try:
        response = get_session().get(
            url,
            params=params,
            headers={"Accept": "application/vnd.github+json"},
            follow_redirects=True,
            timeout=GITHUB_API_TIMEOUT,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise CLIError(f"Failed to fetch '{endpoint}' from {DEFAULT_SKILLS_REPO_ID}: {exc}") from exc
    return response
