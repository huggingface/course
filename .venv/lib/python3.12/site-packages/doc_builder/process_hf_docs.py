#!/usr/bin/env python3
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

"""
Process documentation from HuggingFace doc-build dataset.
Downloads and processes pre-built documentation markdown files.
"""

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import httpx
from packaging import version as package_version
from tqdm import tqdm

from .build_embeddings import Chunk, split_markdown_by_headings

HF_DATASET_REPO = "hf-doc-build/doc-build"
HF_DATASET_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/tree/main"
HF_DATASET_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"


def get_latest_version_zip(library_name: str) -> str | None:
    """
    Get the latest version zip filename for a library by querying the API.

    Args:
        library_name: Name of the library (e.g., 'reachy_mini')

    Returns:
        The filename of the latest version zip (e.g., 'v1.2.13.zip'), or None if not found
    """
    api_url = f"{HF_DATASET_API_URL}/{library_name}"
    print(f"  Querying API for available versions: {api_url}")

    try:
        response = httpx.get(api_url, timeout=60, follow_redirects=True)
        response.raise_for_status()
        files = response.json()

        # Filter for zip files (exclude _versions.yml and main.zip)
        zip_files = [
            f["path"].split("/")[-1]  # Get just the filename
            for f in files
            if f["type"] == "file" and f["path"].endswith(".zip") and "main.zip" not in f["path"]
        ]

        if not zip_files:
            print(f"  No version zips found for {library_name}")
            return None

        # Sort by version (highest first) using packaging.version
        # Filenames are like "v1.2.13.zip" -> extract "1.2.13" (strip 'v' prefix)
        def version_key(filename):
            version_str = filename.replace(".zip", "").lstrip("v")
            try:
                return package_version.parse(version_str)
            except Exception:
                return package_version.parse("0")

        zip_files_sorted = sorted(zip_files, key=version_key, reverse=True)
        latest = zip_files_sorted[0]

        print(f"  Found {len(zip_files)} versions, latest: {latest}")
        return latest

    except Exception as e:
        print(f"  Error querying API: {e}")
        return None


def fetch_library_directories() -> list[dict]:
    """
    Fetch the list of library directories from the HF doc-build dataset.

    Returns:
        List of directory metadata dictionaries with 'path' and 'oid' keys
    """
    print(f"Fetching library directories from {HF_DATASET_API_URL}...")
    response = httpx.get(HF_DATASET_API_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()

    data = response.json()

    # Filter only directories
    directories = [item for item in data if item.get("type") == "directory"]

    print(f"Found {len(directories)} library directories")
    return directories


def download_and_extract_zip(library_name: str, output_dir: Path, zip_filename: str = "main.zip") -> Path | None:
    """
    Download and extract a zip file for a library.

    Args:
        library_name: Name of the library (e.g., 'accelerate')
        output_dir: Directory to extract files to
        zip_filename: Name of the zip file to download (default: 'main.zip')

    Returns:
        Path to extracted directory, or None if download failed
    """
    zip_url = f"{HF_DATASET_BASE_URL}/{library_name}/{zip_filename}"

    try:
        print(f"  Downloading {zip_url}...")
        with httpx.stream("GET", zip_url, follow_redirects=True) as response:
            response.raise_for_status()

            # Get total size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download to memory
            zip_content = io.BytesIO()
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"  {library_name}") as pbar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    zip_content.write(chunk)
                    pbar.update(len(chunk))

        # Extract zip
        zip_content.seek(0)
        extract_path = output_dir / library_name
        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_content) as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"  Extracted to {extract_path}")
        return extract_path

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            if zip_filename == "main.zip":
                # Try to find and download the latest version instead
                print(f"  ⚠️  No main.zip found for {library_name}, looking for latest version...")
                latest_zip = get_latest_version_zip(library_name)
                if latest_zip:
                    return download_and_extract_zip(library_name, output_dir, zip_filename=latest_zip)
                else:
                    print(f"  ⚠️  No versions found for {library_name}, skipping...")
                    return None
            else:
                print(f"  ⚠️  {zip_filename} not found for {library_name}, skipping...")
                return None
        raise
    except Exception as e:
        print(f"  ❌ Error processing {library_name}: {e}")
        return None


def find_markdown_files(directory: Path) -> list[Path]:
    """
    Recursively find all markdown files in a directory.

    Args:
        directory: Root directory to search

    Returns:
        List of paths to markdown files
    """
    markdown_files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix in [".md", ".mdx"]:
            markdown_files.append(file_path)
    return markdown_files


def markdown_file_to_url(file_path: Path, library_name: str, base_dir: Path) -> str:
    """
    Convert a file path to a HuggingFace docs URL.

    Args:
        file_path: Path to the markdown file
        library_name: Name of the library
        base_dir: Base directory (the extracted library folder)

    Returns:
        URL string
    """
    # Get relative path from base_dir
    relative_path = file_path.relative_to(base_dir)

    # Remove file extension
    path_without_ext = relative_path.with_suffix("")

    # Convert to URL format
    url_path = str(path_without_ext).replace(os.sep, "/")

    # Build URL
    url = f"https://huggingface.co/docs/{library_name}/{url_path}"

    return url


def get_page_title(file_path: Path) -> str:
    """
    Generate a page title from file path.

    Args:
        file_path: Path to the file

    Returns:
        Formatted page title
    """
    # Use the filename without extension
    name = file_path.stem
    # Replace underscores and hyphens with spaces
    formatted = name.replace("_", " ").replace("-", " ")
    # Capitalize
    return formatted.title()


def process_markdown_file(
    file_path: Path, library_name: str, base_dir: Path, excerpts_max_length: int = 1000
) -> list[Chunk]:
    """
    Process a single markdown file into chunks.

    Args:
        file_path: Path to the markdown file
        library_name: Name of the library
        base_dir: Base directory for URL generation
        excerpts_max_length: Maximum length of each excerpt

    Returns:
        List of Chunk objects
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Split markdown by headings
        sections = split_markdown_by_headings(content, excerpts_max_length)

        # Generate base URL for this file
        base_url = markdown_file_to_url(file_path, library_name, base_dir)
        page_title = get_page_title(file_path)

        # Convert sections to Chunks
        chunks = []
        for section in sections:
            headings_dict = section["headings"]

            # Create heading list from the dictionary
            heading_list = []
            for i in range(1, 7):
                heading_key = f"heading{i}"
                if heading_key in headings_dict:
                    # Reconstruct the heading with # marks
                    heading_text = headings_dict[heading_key]
                    heading_list.append("#" * i + " " + heading_text)

            # Generate URL with anchor (use first heading as anchor)
            url = base_url
            if headings_dict:
                # Use the deepest heading for anchor
                last_heading = None
                for i in range(6, 0, -1):
                    if f"heading{i}" in headings_dict:
                        last_heading = headings_dict[f"heading{i}"]
                        break

                if last_heading:
                    # Create anchor from heading (lowercase, replace spaces with hyphens)
                    anchor = last_heading.lower().replace(" ", "-")
                    # Remove special characters
                    anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
                    url = f"{base_url}#{anchor}"

            # Create a chunk for each excerpt
            # Get the page path (relative path without extension)
            page_path = str(file_path.relative_to(base_dir).with_suffix("")).replace(os.sep, "/")
            for excerpt in section["excerpts"]:
                chunk = Chunk(
                    text=excerpt,
                    source_page_url=url,
                    source_page_title=page_title,
                    package_name=library_name,
                    headings=heading_list,
                    page=page_path,
                )
                chunks.append(chunk)

        return chunks

    except Exception as e:
        print(f"    ⚠️  Error processing {file_path.name}: {e}")
        return []


def process_library(
    library_name: str, output_dir: Path, excerpts_max_length: int = 1000, skip_download: bool = False
) -> list[Chunk]:
    """
    Process a single library: download, extract, and chunk all markdown files.

    Args:
        library_name: Name of the library
        output_dir: Directory for temporary files
        excerpts_max_length: Maximum length of each excerpt
        skip_download: Skip download if files already exist

    Returns:
        List of all chunks for this library
    """
    print(f"\n📚 Processing library: {library_name}")

    # Check if already extracted
    extract_path = output_dir / library_name

    if skip_download and extract_path.exists():
        print(f"  ℹ️  Using existing files at {extract_path}")
    else:
        # Download and extract
        extract_path = download_and_extract_zip(library_name, output_dir)
        if extract_path is None:
            return []

    # The zip extracts to: extract_path/library_name/{version}/en/
    # where {version} can be "main" or a version like "v1.2.13"
    # We only process the 'en' (English) folder
    library_dir = extract_path / library_name

    # Find the version folder (main, v1.2.13, etc.)
    version_folders = [d for d in library_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not version_folders:
        print(f"  ⚠️  No version folder found for {library_name}")
        return []

    # Use the first (and typically only) version folder
    version_folder = version_folders[0]
    print(f"  Using version folder: {version_folder.name}")

    base_dir = version_folder / "en"
    if not base_dir.exists():
        print(f"  ⚠️  No 'en' folder found in {version_folder.name} for {library_name}")
        return []
    print(f"  Using English docs at {base_dir}")

    # Find all markdown files
    markdown_files = find_markdown_files(base_dir)
    print(f"  Found {len(markdown_files)} markdown files")

    if not markdown_files:
        print(f"  ⚠️  No markdown files found for {library_name}")
        return []

    # Process each markdown file
    all_chunks = []
    print("  Processing markdown files...")
    for md_file in tqdm(markdown_files, desc=f"  {library_name}", unit="file"):
        # Skip model_doc pages for transformers library and api/models for diffusers
        page_path = str(md_file.relative_to(base_dir)).replace(os.sep, "/")
        if library_name == "transformers" and "model_doc" in page_path:
            continue
        if library_name == "diffusers" and ("api/models" in page_path or "api/pipelines" in page_path):
            continue
        chunks = process_markdown_file(md_file, library_name, base_dir, excerpts_max_length)
        all_chunks.extend(chunks)

    print(f"  ✅ Generated {len(all_chunks)} chunks from {len(markdown_files)} files")

    return all_chunks


def process_all_libraries(
    output_dir: Path | None = None,
    excerpts_max_length: int = 1000,
    libraries: list[str] | None = None,
    skip_download: bool = False,
) -> dict:
    """
    Process all libraries from the HF doc-build dataset.

    Args:
        output_dir: Directory for temporary files (uses temp dir if None)
        excerpts_max_length: Maximum length of each excerpt
        libraries: List of specific libraries to process (None = all)
        skip_download: Skip download if files already exist

    Returns:
        Dictionary mapping library names to their chunks
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="hf_docs_"))
        print(f"Using temporary directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch library directories
    directories = fetch_library_directories()

    # Filter if specific libraries requested
    if libraries:
        directories = [d for d in directories if d["path"] in libraries]
        print(f"Processing {len(directories)} requested libraries: {libraries}")

    # Skip libraries containing "course" or "cookbook" (case-insensitive)
    skipped_libraries = []
    filtered_directories = []
    for directory in directories:
        library_name = directory["path"]
        library_name_lower = library_name.lower()
        if "course" in library_name_lower or "cookbook" in library_name_lower:
            skipped_libraries.append(library_name)
        else:
            filtered_directories.append(directory)

    if skipped_libraries:
        print(f"Skipping {len(skipped_libraries)} libraries: {skipped_libraries}")

    directories = filtered_directories

    # Process each library
    results = {}
    for directory in directories:
        library_name = directory["path"]
        chunks = process_library(library_name, output_dir, excerpts_max_length, skip_download)
        results[library_name] = chunks

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    total_chunks = 0
    for library_name, chunks in results.items():
        print(f"  {library_name}: {len(chunks)} chunks")
        total_chunks += len(chunks)
    print(f"\n  Total: {total_chunks} chunks across {len(results)} libraries")
    print("=" * 80)

    return results


def save_chunks_to_json(chunks: list[Chunk], output_file: Path):
    """
    Save chunks to a JSON file.

    Args:
        chunks: List of Chunk objects
        output_file: Path to output JSON file
    """
    # Convert chunks to dictionaries
    chunks_data = [
        {
            "text": chunk.text,
            "source_page_url": chunk.source_page_url,
            "source_page_title": chunk.source_page_title,
            "package_name": chunk.package_name,
            "headings": chunk.headings,
            "page": chunk.page,
        }
        for chunk in chunks
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Process HuggingFace documentation from doc-build dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for downloaded/extracted files (uses temp dir if not specified)",
    )
    parser.add_argument(
        "--libraries",
        type=str,
        nargs="+",
        default=None,
        help="Specific libraries to process (e.g., accelerate diffusers)",
    )
    parser.add_argument(
        "--excerpt-length", type=int, default=1000, help="Maximum length of each excerpt in characters (default: 1000)"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download if files already exist in output-dir"
    )
    parser.add_argument("--save-json", type=str, default=None, help="Save all chunks to a JSON file")

    args = parser.parse_args()

    # Process libraries
    results = process_all_libraries(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        excerpts_max_length=args.excerpt_length,
        libraries=args.libraries,
        skip_download=args.skip_download,
    )

    # Save to JSON if requested
    if args.save_json:
        all_chunks = []
        for chunks in results.values():
            all_chunks.extend(chunks)
        save_chunks_to_json(all_chunks, Path(args.save_json))
