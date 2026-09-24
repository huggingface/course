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


import concurrent
import importlib
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import meilisearch
from huggingface_hub import InferenceClient
from tqdm import tqdm

from .autodoc import autodoc_markdown, resolve_links_in_text
from .convert_md_to_mdx import process_md
from .convert_rst_to_mdx import find_indent, is_empty_line
from .meilisearch_helper import (
    add_embeddings_to_db,
    create_embedding_db,
    delete_embedding_db,
    swap_indexes,
    update_db_settings,
)
from .utils import chunk_list, read_doc_config

Chunk = namedtuple("Chunk", "text source_page_url source_page_title package_name headings page")
Embedding = namedtuple(
    "Embedding",
    "text source_page_url source_page_title library embedding heading1 heading2 heading3 heading4 heading5 page",
)

MEILI_INDEX = "docs-semantic-search"
MEILI_INDEX_TEMP = "docs-semantic-search-temp"

_re_md_anchor = re.compile(r"\[\[(.*)]]")
_re_non_alphaneumeric = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_re_congruent_whitespaces = re.compile(r"\s{2,}")


class MarkdownChunkNode:
    def __init__(self, heading):
        self.heading, self.anchor = self.create_local(heading)
        self.children = []
        self.content = ""

    def create_local(self, heading):
        search_anchor = _re_md_anchor.search(heading)
        if search_anchor:
            # id/local already exists
            local = search_anchor.group(1)
            heading = _re_md_anchor.sub("", heading)
            return heading, local
        else:
            # create id/local
            local = slugify(heading)
            return heading, local

    def add_child(self, child, header_level):
        if header_level < 2:
            raise ValueError(
                "MarkdownChunkNode.add_child method should only be called on root node with heading level of 1"
            )
        parent = self
        nested_level = header_level - 2
        current_level = 1
        while nested_level:
            if not parent.children:
                parent.children.append(MarkdownChunkNode("#" * current_level))
            parent = parent.children[-1]
            nested_level -= 1
            current_level += 1
        parent.children.append(child)

    def get_chunks(self, page_info, chunk_len_chars, headings=None):
        if headings is None:
            headings = []
        chunks = []
        headings = headings + [self.heading]
        split_content = self.split_markdown(self.content)
        if not len(split_content):
            return []
        chunk_str = ""
        for content in split_content:
            if len(chunk_str) > chunk_len_chars:
                chunks.append(
                    Chunk(
                        text=chunk_str.strip(),
                        source_page_url=f"https://huggingface.co/docs/{page_info['package_name']}/{page_info['page']}#{self.anchor}",
                        source_page_title=get_page_title(page_info["page"]),
                        package_name=page_info["package_name"],
                        headings=headings,
                        page=page_info["page"],
                    )
                )
                chunk_str = ""
            chunk_str += content + " "

        if len(chunk_str):
            chunks.append(
                Chunk(
                    text=chunk_str.strip(),
                    source_page_url=f"https://huggingface.co/docs/{page_info['package_name']}/{page_info['page']}#{self.anchor}",
                    source_page_title=get_page_title(page_info["page"]),
                    package_name=page_info["package_name"],
                    headings=headings,
                    page=page_info["page"],
                )
            )

        for child in self.children:
            child_chunks = child.get_chunks(page_info, chunk_len_chars, headings=headings)
            chunks.extend(child_chunks)

        return chunks

    def split_markdown(self, text):
        # split markdown on periods & code blocks
        code_block_pattern = re.compile(r"(```.*?```)", re.DOTALL)
        segments = code_block_pattern.split(text)

        result = []

        for segment in segments:
            if segment.startswith("```") and segment.endswith("```"):
                result.append(segment)
            else:
                subsegments = re.split(r"\.\s+|\.$", segment)
                result.extend([s.strip() for s in subsegments if s.strip()])

        return result


_re_autodoc = re.compile(r"^\s*\[\[autodoc\]\]\s+(\S+)\s*$")
_re_list_item = re.compile(r"^\s*-\s+(\S+)\s*$")


def create_autodoc_chunks(
    content, package, return_anchors=False, page_info=None, version_tag_suffix="src/", headings=None
):
    """
    Replaces [[autodoc]] special syntax by the corresponding generated documentation in some content.

    Args:
        content (`str`): The documentation to treat.
        package (`types.ModuleType`): The package where to look for objects to document.
        return_anchors (`bool`, *optional*, defaults to `False`):
            Whether or not to return the list of anchors generated.
        page_info (`Dict[str, str]`, *optional*): Some information about the page.
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
        headings (`List[str]`, *optional*, defaults to `[]`):
            List of headings to add to the chunks.
    """
    if headings is None:
        headings = []
    lines = content.split("\n")
    autodoc_chunks = []
    if return_anchors:
        anchors = []
        errors = []
    idx = 0
    while idx < len(lines):
        if _re_autodoc.search(lines[idx]) is not None:
            object_name = _re_autodoc.search(lines[idx]).groups()[0]
            autodoc_indent = find_indent(lines[idx])
            idx += 1
            while idx < len(lines) and is_empty_line(lines[idx]):
                idx += 1
            if (
                idx < len(lines)
                and find_indent(lines[idx]) > autodoc_indent
                and _re_list_item.search(lines[idx]) is not None
            ):
                methods = []
                methods_indent = find_indent(lines[idx])
                while is_empty_line(lines[idx]) or (
                    find_indent(lines[idx]) == methods_indent and _re_list_item.search(lines[idx]) is not None
                ):
                    if not is_empty_line(lines[idx]):
                        methods.append(_re_list_item.search(lines[idx]).groups()[0])
                    idx += 1
                    if idx >= len(lines):
                        break
            else:
                methods = None
            object_docs, new_anchors, new_errors = autodoc_markdown(
                object_name,
                package,
                methods=methods,
                return_anchors=return_anchors,
                page_info=page_info,
                version_tag_suffix=version_tag_suffix,
            )

            object_doc_chunks = [
                Chunk(
                    text=od["doc"],
                    source_page_url=f"https://huggingface.co/docs/{page_info['package_name']}/{page_info['page']}#{od['anchor_name']}",
                    source_page_title=get_page_title(page_info["page"]),
                    package_name=page_info["package_name"],
                    headings=headings,
                    page=page_info["page"],
                )
                for od in object_docs
            ]

            autodoc_chunks.extend(object_doc_chunks)

            if return_anchors:
                # if len(doc[1]) and idx_last_heading is not None:
                #     object_anchor = doc[1][0]
                #     autodoc_chunks[idx_last_heading] += f"[[{object_anchor}]]"
                #     idx_last_heading = None
                anchors.extend(new_anchors)
                errors.extend(new_errors)

        else:
            idx += 1

    return (autodoc_chunks, anchors, errors) if return_anchors else autodoc_chunks


def get_autodoc_sections(markdown_text: str) -> list[dict[str, any]]:
    """
    Extract text chunks and their corresponding heading paths.

    Returns:
        List of dictionaries, each containing 'text' and 'headings' keys
    """
    sections = get_autodoc_sections_helper(markdown_text)

    result = []

    for section in sections:
        if section["content"].strip():  # Only include sections with content
            # Create headings array including current heading
            headings_array = section["parent_headings"][:]  # Copy parent headings
            if section["heading"]:
                headings_array.append(section["heading"])

            result.append({"text": section["content"], "headings": headings_array if headings_array else []})

    return result


def get_autodoc_sections_helper(markdown_text: str) -> list[dict[str, any]]:
    """
    Split markdown text by headings and include parent headings for each section.

    Args:
        markdown_text (str): The input markdown text

    Returns:
        List of dictionaries, each containing:
        - 'heading': the current section heading
        - 'parent_headings': list of parent headings (from highest to lowest level)
        - 'full_heading_path': complete heading path as string
        - 'content': the text content of the section
        - 'level': heading level (1 for #, 2 for ##, etc.)
    """
    lines = markdown_text.split("\n")
    sections = []

    # Pattern to match headings (# ## ### etc.)
    heading_pattern = r"^(#+)\s+(.+)$"

    # Keep track of heading hierarchy
    heading_stack = []  # Stack to maintain parent headings
    current_content = []

    for line in lines:
        heading_match = re.match(heading_pattern, line)

        if heading_match:
            # Save previous section if it exists
            if heading_stack or current_content:
                if heading_stack:
                    sections.append(
                        {
                            "heading": heading_stack[-1]["text"],
                            "parent_headings": [h["text"] for h in heading_stack[:-1]],
                            "full_heading_path": " > ".join([h["text"] for h in heading_stack]),
                            "content": "\n".join(current_content).strip(),
                            "level": heading_stack[-1]["level"],
                        }
                    )
                else:
                    # Content before any heading
                    sections.append(
                        {
                            "heading": "",
                            "parent_headings": [],
                            "full_heading_path": "",
                            "content": "\n".join(current_content).strip(),
                            "level": 0,
                        }
                    )

            # Process new heading
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            full_heading = heading_match.group(1) + " " + heading_text

            # Update heading stack based on level
            # Remove headings at same or lower level
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()

            # Add current heading to stack
            heading_stack.append({"text": full_heading, "level": level})

            # Reset content for new section
            current_content = []
        else:
            # Add line to current section content
            current_content.append(line)

    # Don't forget the last section
    if heading_stack or current_content:
        if heading_stack:
            sections.append(
                {
                    "heading": heading_stack[-1]["text"],
                    "parent_headings": [h["text"] for h in heading_stack[:-1]],
                    "full_heading_path": " > ".join([h["text"] for h in heading_stack]),
                    "content": "\n".join(current_content).strip(),
                    "level": heading_stack[-1]["level"],
                }
            )
        else:
            # Content at the end without heading
            sections.append(
                {
                    "heading": "",
                    "parent_headings": [],
                    "full_heading_path": "",
                    "content": "\n".join(current_content).strip(),
                    "level": 0,
                }
            )

    return sections


def create_markdown_chunks(text, page_info=None):
    # todo: replace code blocks
    CODE_COMMENT_ESCAPE = "ESCAPE-PYTHON-CODE-COMMENT"
    _re_codeblock = re.compile(r"```.+?```", re.DOTALL)
    text = _re_codeblock.sub(lambda m: m[0].replace("#", CODE_COMMENT_ESCAPE), text)

    # Insert a newline at the start if not present to standardize the split process
    if not text.startswith("\n"):
        text = "\n" + text

    # Split by headers, keeping the headers as delimiters
    sections = re.split(r"(\n#+ [^\n]+)", text)

    # Organize the content under each heading
    root = None
    node = None

    # Loop through sections to associate text with headings
    for section in sections:
        if section.strip() and re.match(r"\n#+ [^\n]+", section):
            heading = section.strip()
            heading_level = heading.count("#")
            node = MarkdownChunkNode(heading)
            if heading_level == 1:
                root = node
            else:
                if root is None:
                    root = node
                else:
                    root.add_child(node, heading_level)
        elif node:
            section.replace(CODE_COMMENT_ESCAPE, "#")
            node.content += section.strip()

    if root is None:
        return []

    CHUNK_LEN_CHARS = 2000
    chunks = root.get_chunks(page_info, chunk_len_chars=CHUNK_LEN_CHARS)
    return chunks


_re_html_comment = re.compile(r"<!--.+?-->", re.DOTALL)
_re_framework_pytorch = re.compile(r"<frameworkcontent>.+?<pt>(.+?)<\/pt>.+?<\/frameworkcontent>", re.DOTALL)


def clean_md(text):
    text = text.replace("[[open-in-colab]]", "")
    text = _re_html_comment.sub("", text)
    text = _re_framework_pytorch.sub(lambda m: m.group(1).strip(), text)
    return text.strip()


def split_into_excerpts(text: str, max_length: int) -> list[str]:
    """
    Split text into excerpts of approximately max_length characters.

    Args:
        text: The text to split
        max_length: Maximum length of each excerpt in characters

    Returns:
        List of text excerpts
    """
    if not text:
        return []

    excerpts = []
    current_index = 0

    while current_index < len(text):
        end_index = current_index + max_length

        # If we're at the end of the text, just take the rest
        if end_index >= len(text):
            excerpts.append(text[current_index:].strip())
            break

        # Look for the next word boundary after "max_length" characters
        remaining_text = text[end_index:]
        word_boundary_match = re.search(r"\b", remaining_text)

        margin = 50

        if word_boundary_match and word_boundary_match.start() <= margin:
            # Found a word boundary within 50 characters, use it
            end_index = end_index + word_boundary_match.start()
        else:
            # No word boundary within 50 chars, fall back to finding good breaking points
            break_point = end_index
            for i in range(end_index, max(current_index + max_length - margin, current_index), -1):
                if i < len(text):
                    char = text[i]
                    if char in [" ", "\n", ".", ",", ";", "!", "?"]:
                        break_point = i + 1
                        break
            end_index = break_point

        excerpts.append(text[current_index:end_index].strip())
        current_index = end_index

    return [excerpt for excerpt in excerpts if len(excerpt) > 0]


def build_headings_object(heading_stack: list[str]) -> dict:
    """
    Build headings dictionary from heading stack.

    Args:
        heading_stack: List of heading strings in format "## Heading Text"

    Returns:
        Dictionary with heading1 through heading6 keys
    """
    headings = {}

    for heading in heading_stack:
        match = re.match(r"^(#{1,6})\s+(.+)$", heading)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings[f"heading{level}"] = text

    return headings


def split_markdown_by_headings(markdown_content: str, excerpts_max_length: int = 1000) -> list[dict]:
    """
    Split markdown content by headings and create sections with excerpts.
    Similar to the TypeScript implementation for consistent chunking.

    Args:
        markdown_content: The markdown text to split
        excerpts_max_length: Maximum length of each excerpt in characters (default: 1000)

    Returns:
        List of dictionaries with 'excerpts' (list of text chunks) and 'headings' (dict) keys
    """
    lines = markdown_content.split("\n")
    sections = []

    current_section = ""
    heading_stack = []
    line_index = 0

    while line_index < len(lines):
        line = lines[line_index]
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            # Save the previous section if it has content
            if current_section.strip():
                sections.append(
                    {
                        "excerpts": split_into_excerpts(current_section.strip(), excerpts_max_length),
                        "headings": build_headings_object(heading_stack),
                    }
                )

            # Parse the heading
            heading_level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            full_heading = f"{heading_match.group(1)} {heading_text}"

            # Update heading stack based on level
            # Keep only headings with lower level than current
            new_stack = []
            for h in heading_stack:
                h_match = re.match(r"^(#{1,6})", h)
                if h_match:
                    existing_level = len(h_match.group(1))
                    if existing_level < heading_level:
                        new_stack.append(h)
            heading_stack = new_stack

            # Add current heading
            heading_stack.append(full_heading)

            # Start new section with the heading
            current_section = line

            # Look ahead to include content after heading
            line_index += 1
            while line_index < len(lines):
                next_line = lines[line_index]
                next_heading_match = re.match(r"^(#{1,6})\s+(.+)$", next_line)

                if next_heading_match:
                    # Found next heading, break to process it
                    break
                else:
                    # Add line to current section
                    current_section += "\n" + next_line
                    line_index += 1

            # Don't increment line_index here since we either reached end or found next heading
            continue
        else:
            # Add line to current section
            current_section += ("\n" if current_section else "") + line
            line_index += 1

    # Add the last section
    if current_section.strip():
        sections.append(
            {
                "excerpts": split_into_excerpts(current_section.strip(), excerpts_max_length),
                "headings": build_headings_object(heading_stack),
            }
        )

    return sections


def get_page_title(path: str):
    """
    Given a path to doc page, generate doc page title.
    Example: "api/schedulers/lms_discrete" -> "Lms discrete"
    """
    # Split the string by '/' and take the last part
    last_part = path.split("/")[-1]
    # Replace underscores with spaces
    formatted_string = last_part.replace("_", " ")
    # Capitalize the first letter of the entire string
    return formatted_string.capitalize()


def slugify(string: str):
    local = _re_non_alphaneumeric.sub("", string)
    local = _re_congruent_whitespaces.sub(" ", local.strip()).replace(" ", "-")
    local = local.lower()
    return local


_re_autodoc_all = re.compile(r"(\[\[autodoc\]\]\s+[\w\.]+(?:\n\s+-\s+\w+)*\b)", re.DOTALL)


class ChunkingError(Exception):
    pass


def create_chunks(package, doc_folder, page_info, version_tag_suffix, is_python_module) -> list[Chunk]:
    """
    Build the MDX files for a given package.

    Args:
        package (`types.ModuleType`): The package where to look for objects to document.
        doc_folder (`str` or `os.PathLike`): The folder where the doc source files are.
        page_info (`Dict[str, str]`): Some information about the page.
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
    """
    doc_folder = Path(doc_folder)
    anchor_mapping = {}

    if "package_name" not in page_info:
        page_info["package_name"] = package.__name__

    chunks = []
    all_files = list(doc_folder.glob("**/*"))
    all_errors = []
    for file in tqdm(all_files, desc="Building the chunks to embed"):
        new_anchors = None
        errors = None
        page_info["path"] = file
        try:
            if file.suffix in [".md", ".mdx"]:
                page_info["page"] = file.with_suffix("").relative_to(doc_folder).as_posix()
                with open(file, encoding="utf-8-sig") as reader:
                    content = reader.read()
                content = clean_md(content)
                content = process_md(content, page_info)

                autodoc_chunks, new_anchors, errors = [], [], []
                if "[[autodoc]]" in content:
                    autodoc_sections = get_autodoc_sections(content)
                    for section in autodoc_sections:
                        autodoc_content = "\n\n".join(_re_autodoc_all.findall(section["text"]))
                        if not autodoc_content.strip():
                            continue
                        _autodoc_chunks, _new_anchors, _errors = create_autodoc_chunks(
                            autodoc_content,
                            package,
                            return_anchors=True,
                            page_info=page_info,
                            version_tag_suffix=version_tag_suffix,
                            headings=section["headings"],
                        )
                        autodoc_chunks.extend(_autodoc_chunks)
                        new_anchors.extend(_new_anchors)
                        errors.extend(_errors)

                content = _re_autodoc_all.sub("", content)

                markdown_chunks = create_markdown_chunks(
                    content,
                    page_info=page_info,
                )

                # Make sure we clean up for next page.
                del page_info["page"]

                page_chunks = markdown_chunks + autodoc_chunks

                if is_python_module:
                    page_chunks = [
                        chunk._replace(text=resolve_links_in_text(chunk.text, package, anchor_mapping, page_info))
                        for chunk in page_chunks
                    ]

                chunks.extend(page_chunks)

        except Exception as e:
            raise ChunkingError(f"There was an error when converting {file} to chunks to embed.\n" + e.args[0]) from e

        if new_anchors:
            page_name = str(file.with_suffix("").relative_to(doc_folder))
            for anchor in new_anchors:
                if isinstance(anchor, tuple):
                    anchor_mapping.update(
                        {a: f"{page_name}#{anchor[0]}" for a in anchor[1:] if a not in anchor_mapping}
                    )
                    anchor = anchor[0]
                anchor_mapping[anchor] = page_name

        if errors:
            all_errors.extend(errors)

    if len(all_errors) > 0:
        raise ValueError(
            "The deployment of the documentation will fail because of the following errors:\n" + "\n".join(all_errors)
        )

    return chunks


def chunks_to_embeddings(client, chunks, is_python_module) -> list[Embedding]:
    texts = []
    for c in chunks:
        prefix = f'Documentation of {"library" if is_python_module else "service"} "{c.package_name}" under section: {" > ".join(c.headings)}'
        texts.append(prefix + "\n\n" + c.text)

    inference_output = client.feature_extraction(texts, truncate=True)
    inference_output = inference_output.tolist()

    embeddings = []
    for c, embed in zip(chunks, inference_output, strict=False):
        headings = [None] * 5
        last_heading = None

        for heading_str in c.headings:
            level = heading_str.count("#")
            heading_text = heading_str.lstrip("# ").strip()
            if 1 <= level <= 5:
                headings[level - 1] = heading_text
                last_heading = heading_text

        # If the page does not have any heading, add the last heading to the page URL
        source_page_url = c.source_page_url
        if "#" not in c.source_page_url and last_heading is not None:
            source_page_url += "#" + slugify(last_heading)

        embeddings.append(
            Embedding(
                text=c.text,
                source_page_url=source_page_url,
                source_page_title=c.source_page_title,
                library=c.package_name,
                embedding=embed,
                heading1=headings[0],
                heading2=headings[1],
                heading3=headings[2],
                heading4=headings[3],
                heading5=headings[4],
                page=c.page,
            )
        )

    return embeddings


def call_embedding_inference(chunks: list[Chunk], hf_ie_url, hf_ie_token, is_python_module) -> list[Embedding]:
    """
    Using https://huggingface.co/inference-endpoints with a text embedding model
    """
    if not hf_ie_url:
        raise ValueError("hf_ie_url is required but got empty value")
    if not hf_ie_token:
        raise ValueError("hf_ie_token is required but got empty value")

    batch_size = 20
    embeddings = []

    client = InferenceClient(base_url=hf_ie_url, token=hf_ie_token)

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_chunk = {
            executor.submit(chunks_to_embeddings, client, chunks[i : i + batch_size], is_python_module): i
            for i in range(0, len(chunks), batch_size)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_chunk),
            total=len(future_to_chunk),
            desc="Calling Embedding Inference Endpoints",
        ):
            batch_embeddings = future.result()
            embeddings.extend(batch_embeddings)

    return embeddings


def build_embeddings(
    package_name,
    doc_folder,
    hf_ie_url,
    hf_ie_token,
    meilisearch_key,
    meilisearch_url,
    version="main",
    version_tag="main",
    language="en",
    is_python_module=False,
    version_tag_suffix="src/",
    repo_owner="huggingface",
    repo_name=None,
):
    """
    Build the documentation of a package.

    Args:
        package_name (`str`): The name of the package.
        doc_folder (`str` or `os.PathLike`): The folder in which the source documentation of the package is.
        version (`str`, *optional*, defaults to `"main"`): The name of the version of the doc.
        version_tag (`str`, *optional*, defaults to `"main"`): The name of the version tag (on GitHub) of the doc.
        language (`str`, *optional*, defaults to `"en"`): The language of the doc.
        is_python_module (`bool`, *optional*, defaults to `False`):
            Whether the docs being built are for python module. (For example, HF Course is not a python module).
        version_tag_suffix (`str`, *optional*, defaults to `"src/"`):
            Suffix to add after the version tag (e.g. 1.3.0 or main) in the documentation links.
            For example, the default `"src/"` suffix will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/src/`.
            For example, `version_tag_suffix=""` will result in a base link as `https://github.com/huggingface/{package_name}/blob/{version_tag}/`.
        repo_owner (`str`, *optional*, defaults to `"huggingface"`):
            The owner of the repository on GitHub. In most cases, this is `"huggingface"`. However, for the `timm` library, the owner is `"rwightman"`.
        repo_name (`str`, *optional*, defaults to `package_name`):
            The name of the repository on GitHub. In most cases, this is the same as `package_name`. However, for the `timm` library, the name is `"pytorch-image-models"` instead of `"timm"`.
    """
    page_info = {
        "version": version,
        "version_tag": version_tag,
        "language": language,
        "package_name": package_name,
        "repo_owner": repo_owner,
        "repo_name": repo_name if repo_name is not None else package_name,
    }

    read_doc_config(doc_folder)

    package = importlib.import_module(package_name) if is_python_module else None

    # Step 1: create chunks
    chunks = create_chunks(
        package,
        doc_folder,
        page_info,
        version_tag_suffix=version_tag_suffix,
        is_python_module=is_python_module,
    )

    # Step 2: create embeddings
    embeddings = call_embedding_inference(chunks, hf_ie_url, hf_ie_token, is_python_module)

    # Step 3: push embeddings to vector database (meilisearch)
    client = meilisearch.Client(meilisearch_url, meilisearch_key)
    ITEMS_PER_CHUNK = 5000  # a value that was found experimentally
    for chunk_embeddings in tqdm(chunk_list(embeddings, ITEMS_PER_CHUNK), desc="Uploading data to meilisearch"):
        add_embeddings_to_db(client, MEILI_INDEX_TEMP, chunk_embeddings)


def clean_meilisearch(meilisearch_key: str, swap: bool, meilisearch_url: str):
    """
    Swap & delete temp index.
    """
    client = meilisearch.Client(meilisearch_url, meilisearch_key)
    if swap:
        swap_indexes(client, MEILI_INDEX, MEILI_INDEX_TEMP)
    delete_embedding_db(client, MEILI_INDEX_TEMP)
    create_embedding_db(client, MEILI_INDEX_TEMP)
    update_db_settings(client, MEILI_INDEX_TEMP)
    print("[meilisearch] successfully swapped & deleted temp index.")


def add_gradio_docs(hf_ie_url: str, hf_ie_token: str, meilisearch_key: str, meilisearch_url: str):
    """Add Gradio documentation to embeddings."""
    # Step 1: download the documentation
    url = "https://huggingface.co/datasets/gradio/docs/resolve/main/docs.json"

    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()  # Raises an HTTPError for bad responses

    data = response.json()

    chunks = [
        Chunk(
            doc["text"],
            doc["source_page_url"],
            doc["source_page_title"],
            "gradio",
            [doc["source_page_title"]] + ([doc["heading1"]] if doc.get("heading1") else []),
            doc["source_page_url"],  # use URL as page identifier for gradio docs
        )
        for doc in data
    ]

    # Step 2: create embeddings
    batch_size = 20
    embeddings = []

    client = InferenceClient(base_url=hf_ie_url, token=hf_ie_token)

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_chunk = {
            executor.submit(chunks_to_embeddings, client, chunks[i : i + batch_size], True): i
            for i in range(0, len(chunks), batch_size)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_chunk),
            total=len(future_to_chunk),
            desc="Calling Embedding Inference Endpoints",
        ):
            batch_embeddings = future.result()
            embeddings.extend(batch_embeddings)

    # Step 3: push embeddings to vector database (meilisearch)
    client = meilisearch.Client(meilisearch_url, meilisearch_key)
    ITEMS_PER_CHUNK = 5000  # a value that was found experimentally
    for chunk_embeddings in tqdm(chunk_list(embeddings, ITEMS_PER_CHUNK), desc="Uploading gradio docs to meilisearch"):
        add_embeddings_to_db(client, MEILI_INDEX_TEMP, chunk_embeddings)
