# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import yaml
from packaging import version as package_version

hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "doc_builder")
DOC_BUILDER_CACHE = os.getenv("DOC_BUILDER_CACHE", default_cache_path)


def get_default_branch_name(repo_folder):
    config = get_doc_config()
    if config is not None and hasattr(config, "default_branch_name"):
        print(config.default_branch_name)
        return config.default_branch_name
    try:
        p = subprocess.run(
            "git symbolic-ref refs/remotes/origin/HEAD".split(),
            capture_output=True,
            check=True,
            encoding="utf-8",
            cwd=repo_folder,
        )
        branch = p.stdout.strip().split("/")[-1]
        return branch
    except Exception:
        # Just in case git is not installed, we need a default
        return "main"


def update_versions_file(build_path, version, doc_folder):
    """
    Insert new version into _versions.yml file of the library
    Assumes that _versions.yml exists and has its first entry as main version
    """
    main_branch = get_default_branch_name(doc_folder)
    if version == main_branch:
        return
    with open(os.path.join(build_path, "_versions.yml")) as versions_file:
        versions = yaml.load(versions_file, yaml.FullLoader)

        if versions[0]["version"] != main_branch:
            raise ValueError(f"{build_path}/_versions.yml does not contain a {main_branch} version")

        main_version, sem_versions = versions[0], versions[1:]
        new_version = {"version": version}
        did_insert = False
        for i, value in enumerate(sem_versions):
            if package_version.parse(new_version["version"]) == package_version.parse(value["version"]):
                # Nothing to do, the version is here already.
                return
            elif package_version.parse(new_version["version"]) > package_version.parse(value["version"]):
                sem_versions.insert(i, new_version)
                did_insert = True
                break
        if not did_insert:
            sem_versions.append(new_version)

    with open(os.path.join(build_path, "_versions.yml"), "w") as versions_file:
        versions_updated = [main_version] + sem_versions
        yaml.dump(versions_updated, versions_file)


doc_config = None


def read_doc_config(doc_folder):
    """
    Execute the `_config.py` file inside the doc source directory and executes it as a Python module.
    """
    global doc_config

    if os.path.isfile(os.path.join(doc_folder, "_config.py")):
        loader = importlib.machinery.SourceFileLoader("doc_config", os.path.join(doc_folder, "_config.py"))
        spec = importlib.util.spec_from_loader("doc_config", loader)
        doc_config = importlib.util.module_from_spec(spec)
        loader.exec_module(doc_config)


def get_doc_config():
    """
    Returns the `doc_config` if it has been loaded.
    """
    return doc_config


def is_watchdog_available():
    """
    Checks if soft dependency `watchdog` exists.
    """
    return importlib.util.find_spec("watchdog") is not None


def is_doc_builder_repo(path):
    """
    Detects whether a folder is the `doc_builder` or not.
    """
    setup_file = Path(path) / "setup.py"
    if not setup_file.exists():
        return False
    with open(os.path.join(path, "setup.py")) as f:
        first_line = f.readline()
    return first_line == "# Doc-builder package setup.\n"


def locate_kit_folder():
    """
    Returns the location of the `kit` folder of `doc-builder`.

    Will clone the doc-builder repo and cache it, if it's not found.
    """
    # First try: let's search where the module is.
    repo_root = Path(__file__).parent.parent.parent
    kit_folder = repo_root / "kit"
    if kit_folder.is_dir():
        return kit_folder

    # Second try, maybe we are inside the doc-builder repo
    current_dir = Path.cwd()
    while current_dir.parent != current_dir and not (current_dir / ".git").is_dir():
        current_dir = current_dir.parent
    kit_folder = current_dir / "kit"
    if kit_folder.is_dir() and is_doc_builder_repo(current_dir):
        return kit_folder

    # Otherwise, let's clone the repo and cache it.
    return Path(get_cached_repo()) / "kit"


def get_cached_repo():
    """
    Clone and cache the `doc-builder` repo.
    """
    os.makedirs(DOC_BUILDER_CACHE, exist_ok=True)
    cache_repo_path = Path(DOC_BUILDER_CACHE) / "doc-builder-repo"
    if not cache_repo_path.is_dir():
        print(
            "To build the HTML doc, we need the kit subfolder of the `doc-builder` repo. Cloning it and caching at "
            f"{cache_repo_path}."
        )
        _ = subprocess.run(
            "git clone https://github.com/huggingface/doc-builder.git".split(),
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            cwd=DOC_BUILDER_CACHE,
        )
        shutil.move(Path(DOC_BUILDER_CACHE) / "doc-builder", cache_repo_path)
    else:
        _ = subprocess.run(
            ["git", "pull"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            cwd=cache_repo_path,
        )
    return cache_repo_path


_SCRIPT_BLOCK_RE = re.compile(r"^\s*<script\b[^>]*>.*?</script>\s*", re.DOTALL)
_SCRIPT_MARKERS = ("HF_DOC_BODY_START", "HF_DOC_BODY_END")
_DOCBUILD_COMMENT_RE = re.compile(r"<!--\s*HF\s*DOCBUILD\s*BODY\s*(?:START|END)\s*-->", re.IGNORECASE)
_DOCBODY_LINE_RE = re.compile(r"^\s*HF_DOC_BODY_(?:START|END)\s*$", re.MULTILINE)


def sveltify_file_route(filename):
    """Convert an `.mdx` file path into the corresponding SvelteKit `+page.svelte` route."""
    filename = str(filename)
    if filename.endswith(".mdx"):
        return filename.rsplit(".", 1)[0] + "/+page.svelte"
    return filename


def markdownify_file_route(filename):
    """Return the `.md` companion file path for a given `.mdx` route."""
    filename = str(filename)
    if filename.endswith(".mdx"):
        return filename.rsplit(".", 1)[0] + ".md"
    return filename


def convert_mdx_to_markdown_text(content: str) -> str:
    """Reduce MDX content to Markdown-only text suitable for distribution."""

    content = _SCRIPT_BLOCK_RE.sub("", content, count=1)

    heading_match = re.search(r"^#", content, flags=re.MULTILINE)
    if heading_match:
        cleaned = content[heading_match.start() :]
    else:
        index_candidates = [content.find(marker) for marker in _SCRIPT_MARKERS if marker in content]
        index_candidates = [idx for idx in index_candidates if idx >= 0]
        cleaned = content[min(index_candidates) :] if index_candidates else content

    cleaned = _DOCBUILD_COMMENT_RE.sub("", cleaned)
    cleaned = _DOCBODY_LINE_RE.sub("", cleaned)
    cleaned = cleaned.replace("HF_DOC_BODY_START", "").replace("HF_DOC_BODY_END", "")
    return cleaned.lstrip().rstrip()


def write_markdown_route_file(source_file, destination_file):
    """
    Convert a generated `.mdx` file into the Markdown format expected by SvelteKit routes.

    The transformation removes a leading `<script>...</script>` block and ensures the
    resulting file starts directly with Markdown content.
    """

    with open(source_file, encoding="utf-8") as f:
        content = convert_mdx_to_markdown_text(f.read())

    # Strip HTML from the markdown content before writing
    content = strip_html_from_markdown(content)

    destination_path = Path(destination_file)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with open(destination_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    return content


def _collect_markdown_from_output(output_dir: Path) -> list[tuple[str, str]]:
    markdown_items: list[tuple[str, str]] = []
    for mdx_file in sorted(output_dir.glob("**/*.mdx")):
        relative_path = mdx_file.relative_to(output_dir).with_suffix(".md").as_posix()
        with open(mdx_file, encoding="utf-8") as f:
            markdown_text = convert_mdx_to_markdown_text(f.read())
        markdown_items.append((relative_path, markdown_text))
    return markdown_items


def write_llms_feeds(
    output_dir: Path,
    markdown_items: Sequence[tuple[str, str]] | None = None,
    base_url: str | None = None,
    package_name: str | None = None,
    version: str | None = None,
    language: str | None = None,
    is_python_module: bool = True,
):
    """Generate llms.txt and llms-full.txt files alongside the documentation output."""

    output_dir = Path(output_dir)
    if markdown_items is None:
        markdown_items = _collect_markdown_from_output(output_dir)
    else:
        markdown_items = [(str(path).replace(os.sep, "/"), text) for path, text in markdown_items]

    markdown_items = [item for item in markdown_items if item[0]]
    if not markdown_items:
        return

    parts = list(output_dir.parts)
    if len(parts) >= 1 and language is None:
        language = parts[-1]
    if len(parts) >= 2 and version is None:
        version = parts[-2]
    if len(parts) >= 3 and package_name is None:
        package_name = parts[-3]

    base_host = "https://huggingface.co/docs"
    normalized_package = (package_name or "").strip()
    if normalized_package.endswith("course") or normalized_package == "cookbook":
        base_host = "https://huggingface.co/learn"

    def should_include_language(lang: str | None) -> bool:
        return bool(lang and lang.lower() != "en")

    def should_include_version(is_module: bool, ver: str | None) -> bool:
        return is_module and ver is not None

    if base_url is None and package_name:
        url_parts = [base_host, quote(package_name, safe="")]
        if should_include_version(is_python_module, version):
            url_parts.append(quote(version, safe=""))
        if should_include_language(language):
            url_parts.append(quote(language, safe=""))
        base_url = "/".join(url_parts)
    elif base_url is not None and base_url.startswith("https://") and package_name:
        normalized_base = base_url.rstrip("/")
        # Ensure host respects learn/docs rules when caller passes a minimal base_url
        if normalized_base.endswith(f"/docs/{package_name}") and (
            normalized_package.endswith("course") or normalized_package == "cookbook"
        ):
            base_url = normalized_base.replace("/docs/", "/learn/", 1)

    header_title = normalized_package.title() if normalized_package else "Documentation"

    def build_url(relative_path: str) -> str:
        relative_path = relative_path.lstrip("/")
        if base_url:
            return f"{base_url.rstrip('/')}/{quote(relative_path, safe='/')}"
        return f"/{relative_path}"

    def extract_title(markdown_text: str, fallback: str) -> str:
        for line in markdown_text.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip() or fallback
        return fallback

    bullet_lines: list[str] = []
    sections: list[str] = []

    for relative_path, markdown_text in markdown_items:
        url = build_url(relative_path)
        fallback_title = Path(relative_path).name.replace(".md", "").replace("-", " ").replace("_", " ").title()
        title = extract_title(markdown_text, fallback_title)
        bullet_lines.append(f"- [{title}]({url})")
        markdown_body = markdown_text.strip()
        sections.extend([f"### {title}", url, "", markdown_body, ""] if markdown_body else [f"### {title}", url, ""])

    header_lines = [f"# {header_title}", "", "## Docs", ""]
    llms_lines = header_lines + bullet_lines + [""]
    llms_full_lines = header_lines + bullet_lines + [""] + sections

    output_dir.joinpath("llms.txt").write_text("\n".join(llms_lines).rstrip() + "\n", encoding="utf-8")
    output_dir.joinpath("llms-full.txt").write_text("\n".join(llms_full_lines).rstrip() + "\n", encoding="utf-8")


def chunk_list(lst, n):
    """
    Create a list of chunks
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]


class HTMLStripper(HTMLParser):
    """Helper class to strip HTML tags while preserving text content."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_data(self):
        return "".join(self.text)


def strip_html_tags(text: str) -> str:
    """Strip HTML tags from text while preserving content."""
    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_data()


def extract_docstring_info(docstring_block: str) -> dict:
    """Extract information from a docstring block."""
    info = {
        "name": None,
        "anchor": None,
        "source": None,
        "parameters": None,
        "paramsdesc": None,
        "rettype": None,
        "retdesc": None,
        "description": None,
    }

    # Extract name
    name_match = re.search(r"<name>(.*?)</name>", docstring_block, re.DOTALL)
    if name_match:
        raw_name = name_match.group(1).strip()
        # Remove "class " or "def " prefix if present
        cleaned_name = re.sub(r"^(class|def)\s+", "", raw_name)
        info["name"] = cleaned_name

    # Extract anchor
    anchor_match = re.search(r"<anchor>(.*?)</anchor>", docstring_block, re.DOTALL)
    if anchor_match:
        info["anchor"] = anchor_match.group(1).strip()

    # Extract source
    source_match = re.search(r"<source>(.*?)</source>", docstring_block, re.DOTALL)
    if source_match:
        info["source"] = source_match.group(1).strip()

    # Extract parameters description
    paramsdesc_match = re.search(r"<paramsdesc>(.*?)</paramsdesc>", docstring_block, re.DOTALL)
    if paramsdesc_match:
        info["paramsdesc"] = paramsdesc_match.group(1).strip()

    # Extract return type
    rettype_match = re.search(r"<rettype>(.*?)</rettype>", docstring_block, re.DOTALL)
    if rettype_match:
        info["rettype"] = rettype_match.group(1).strip()

    # Extract return description
    retdesc_match = re.search(r"<retdesc>(.*?)</retdesc>", docstring_block, re.DOTALL)
    if retdesc_match:
        info["retdesc"] = retdesc_match.group(1).strip()

    # Extract text outside docstring tags but inside the div
    # This is the description text
    description_match = re.search(r"</docstring>(.*?)(?:</div>|$)", docstring_block, re.DOTALL)
    if description_match:
        desc_text = description_match.group(1).strip()
        # Remove any remaining HTML tags
        desc_text = re.sub(r"<[^>]+>", "", desc_text)
        if desc_text:
            info["description"] = desc_text

    return info


def format_parameters(paramsdesc: str) -> str:
    """
    Format parameter descriptions by:
    - Removing bullets (-)
    - Removing bold formatting (**)
    - Changing -- to :
    - Adding blank lines between parameters
    """
    lines = paramsdesc.split("\n")
    formatted_params = []
    current_param = []

    for line in lines:
        # Check if this is a new parameter line (starts with "- **")
        if re.match(r"^\s*-\s+\*\*", line):
            # Save the previous parameter if exists
            if current_param:
                param_text = " ".join(current_param)
                # Remove - and ** formatting
                param_text = re.sub(r"^\s*-\s+\*\*([^*]+)\*\*", r"\1", param_text)
                # Change -- to :
                param_text = re.sub(r"\s+--\s+", " : ", param_text, count=1)
                formatted_params.append(param_text)
                formatted_params.append("")  # Add blank line between parameters
                current_param = []

            # Start new parameter
            current_param.append(line)
        elif current_param:
            # Continuation of current parameter description
            current_param.append(line.strip())

    # Don't forget the last parameter
    if current_param:
        param_text = " ".join(current_param)
        param_text = re.sub(r"^\s*-\s+\*\*([^*]+)\*\*", r"\1", param_text)
        param_text = re.sub(r"\s+--\s+", " : ", param_text, count=1)
        formatted_params.append(param_text)

    return "\n".join(formatted_params)


def process_docstring_block(docstring_block: str) -> str:
    """
    Process a docstring block by:
    1. Extracting the class/function name and relevant info
    2. Stripping all HTML tags
    3. Converting to clean markdown with level 4 header
    """
    # Extract structured information from the docstring
    info = extract_docstring_info(docstring_block)

    # Build the cleaned markdown
    parts = []

    # Add the name as level 4 header with anchor
    if info["name"]:
        if info["anchor"]:
            parts.append(f"#### {info['name']}[[{info['anchor']}]]")
        else:
            parts.append(f"#### {info['name']}")
        parts.append("")

    # Add source link if available
    if info["source"]:
        # Strip any HTML from source
        source_clean = strip_html_tags(info["source"])
        parts.append(f"[Source]({source_clean})")
        parts.append("")

    # Add description
    if info["description"]:
        parts.append(info["description"])
        parts.append("")

    # Add parameters description
    if info["paramsdesc"]:
        parts.append("**Parameters:**")
        parts.append("")
        # Format parameters: remove bullets and bold, change -- to :, add blank lines
        formatted_params = format_parameters(info["paramsdesc"])
        parts.append(formatted_params)
        parts.append("")

    # Add return type
    if info["rettype"]:
        parts.append("**Returns:**")
        parts.append("")
        # Strip HTML tags from return type
        rettype_clean = strip_html_tags(info["rettype"])
        parts.append(f"`{rettype_clean}`")
        parts.append("")

    # Add return description
    if info["retdesc"]:
        if not info["rettype"]:
            parts.append("**Returns:**")
            parts.append("")
        parts.append(info["retdesc"])
        parts.append("")

    result = "\n".join(parts)

    # Clean up excessive newlines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def strip_remaining_html(content: str) -> str:
    """
    Strip remaining HTML tags while preserving markdown structure.
    Handles tags like <Tip>, <ExampleCodeBlock>, etc.
    """
    # Remove HTML comments, but preserve special flags like <!-- WRAP CODE BLOCKS --> and <!-- STRETCH TABLES -->
    content = re.sub(r"<!--(?!\s*(WRAP CODE BLOCKS|STRETCH TABLES)\s*-->).*?-->", "", content, flags=re.DOTALL)

    # Remove common component tags while preserving their content
    # (Tip, TipEnd, ExampleCodeBlock, hfoptions, hfoption, etc.)
    tags_to_remove = [
        "Tip",
        "TipEnd",
        "ExampleCodeBlock",
        "hfoptions",
        "hfoption",
        "EditOnGithub",
        "div",
        "span",
        "anchor",
    ]

    for tag in tags_to_remove:
        # Remove opening tags with any attributes
        content = re.sub(rf"<{tag}[^>]*>", "", content, flags=re.IGNORECASE)
        # Remove closing tags
        content = re.sub(rf"</{tag}>", "", content, flags=re.IGNORECASE)

    # Remove any remaining HTML tags (generic cleanup)
    # This is more aggressive but preserves text content
    content = re.sub(r"<[^>]+>", "", content)

    # Clean up multiple consecutive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content


def strip_html_from_markdown(content: str) -> str:
    """
    Strip HTML from markdown content.

    Handles:
    - Docstring blocks wrapped in <div class="docstring...">...</div>
    - Other HTML tags throughout the document
    """
    result = content

    # Process docstring blocks with their wrapping divs
    # Pattern to match: <div class="docstring...">...<docstring>...</docstring>...</div>
    docstring_pattern = r'<div[^>]*class="docstring[^"]*"[^>]*>.*?<docstring>.*?</docstring>.*?</div>'

    def replace_docstring(match):
        block = match.group(0)
        return process_docstring_block(block)

    result = re.sub(docstring_pattern, replace_docstring, result, flags=re.DOTALL)

    # Strip remaining HTML tags (like <Tip>, </Tip>, <ExampleCodeBlock>, etc.)
    # But preserve markdown code blocks
    result = strip_remaining_html(result)

    return result
