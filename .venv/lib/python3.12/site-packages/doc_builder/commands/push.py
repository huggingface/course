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

import argparse
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from time import sleep, time

from huggingface_hub import HfApi, hf_hub_download

REPO_TYPE = "dataset"
SEPARATOR = "/"


def create_zip_name(library_name, version, with_ext=True):
    file_name = f"{library_name}{SEPARATOR}{version}"
    if with_ext:
        file_name += ".zip"
    return file_name


def merge_with_existing_docs(api, doc_build_repo_id, zip_file_path, path_docs_built, library_name, doc_version_folder):
    """
    Download existing zip (if any) and merge with new docs to preserve all languages.
    This prevents race conditions when multiple language builds run in parallel.

    Args:
        api: HfApi instance
        doc_build_repo_id: The HF dataset repo ID (e.g., "hf-doc-build/doc-build")
        zip_file_path: Path to the zip file in the repo (e.g., "transformers/main.zip")
        path_docs_built: Local path to newly built docs
        library_name: Name of the library (e.g., "transformers")
        doc_version_folder: Version folder name (e.g., "main" or "v4.57.0")

    Returns:
        Path to the merged docs folder, or original path if no existing docs
    """
    try:
        # Try to download existing zip
        print(f"Checking for existing docs at {zip_file_path}...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Download existing zip
            existing_zip_path = hf_hub_download(
                repo_id=doc_build_repo_id,
                repo_type=REPO_TYPE,
                filename=zip_file_path,
                local_dir=temp_dir,
            )

            print("Found existing docs, merging with new docs...")

            # Extract existing zip to a temporary location
            existing_docs_dir = temp_dir / "existing_docs"
            existing_docs_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(existing_zip_path, "r") as zf:
                zf.extractall(existing_docs_dir)

            # The zip structure is: library_name/version/language/...
            # e.g., transformers/main/en/..., transformers/main/ja/...
            existing_version_dir = existing_docs_dir / library_name / doc_version_folder
            new_version_dir = path_docs_built / doc_version_folder

            if existing_version_dir.exists() and new_version_dir.exists():
                # Get existing languages
                existing_langs = {d.name for d in existing_version_dir.iterdir() if d.is_dir()}
                new_langs = {d.name for d in new_version_dir.iterdir() if d.is_dir()}

                print(f"Existing languages: {sorted(existing_langs)}")
                print(f"New languages: {sorted(new_langs)}")

                # Copy missing languages from existing to new
                langs_to_copy = existing_langs - new_langs
                for lang in langs_to_copy:
                    src = existing_version_dir / lang
                    dst = new_version_dir / lang
                    print(f"Preserving existing language: {lang}")
                    shutil.copytree(src, dst)

                print(f"Final languages: {sorted(existing_langs | new_langs)}")

    except Exception as e:
        # If download fails (e.g., zip doesn't exist yet), just continue with new docs
        print(f"No existing docs found or error downloading ({e}), uploading new docs only")


def push_command(args):
    """
    Commit file doc builds changes using: 1. zip doc build artifacts 2. hf_hub client to upload/delete zip file
    Usage: doc-builder push $args
    """
    if args.n_retries < 1:
        raise ValueError(f"CLI arg `n_retries` MUST be positive & non-zero; supplied value was {args.n_retries}")
    if args.is_remove:
        push_command_remove(args)
    else:
        push_command_add(args)


def push_command_add(args):
    """
    Commit file changes using: 1. zip doc build artifacts 2. hf_hub client to upload zip file
    Used in: build_main_documentation.yml & build_pr_documentation.yml

    This function merges new docs with existing docs to preserve all languages when
    multiple language builds run in parallel (e.g., English and other languages).
    """
    max_n_retries = args.n_retries + 1
    number_of_retries = args.n_retries
    n_seconds_sleep = 5

    library_name = args.library_name
    path_docs_built = Path(library_name)
    doc_version_folder = next(filter(lambda x: not x.is_file(), path_docs_built.glob("*")), None).relative_to(
        path_docs_built
    )
    doc_version_folder = str(doc_version_folder)

    zip_file_path = create_zip_name(library_name, doc_version_folder)

    api = HfApi()

    # Merge with existing docs to preserve all languages (handles parallel builds)
    merge_with_existing_docs(
        api=api,
        doc_build_repo_id=args.doc_build_repo_id,
        zip_file_path=zip_file_path,
        path_docs_built=path_docs_built,
        library_name=library_name,
        doc_version_folder=doc_version_folder,
    )

    # eg create ./transformers/v4.0.zip with '/transformers/v4.0/*' file architecture inside
    # Use subprocess.run instead of shutil.make_archive to avoid corrupted files, see https://github.com/huggingface/doc-builder/issues/348
    print(f"Running zip command: zip -r {zip_file_path} {path_docs_built}")
    subprocess.run(["zip", "-r", zip_file_path, path_docs_built], check=True)

    time_start = time()

    while number_of_retries:
        try:
            if args.upload_version_yml:
                # removing doc artifact folder to upload 2 files using `upload_folder`: _version.yml and zipped doc artifact file
                shutil.rmtree(f"{library_name}/{doc_version_folder}")
                api.upload_folder(
                    repo_id=args.doc_build_repo_id,
                    repo_type=REPO_TYPE,
                    folder_path=library_name,
                    path_in_repo=library_name,
                    commit_message=args.commit_msg,
                    token=args.token,
                )
            else:
                api.upload_file(
                    repo_id=args.doc_build_repo_id,
                    repo_type=REPO_TYPE,
                    path_or_fileobj=zip_file_path,
                    path_in_repo=zip_file_path,
                    commit_message=args.commit_msg,
                    token=args.token,
                )
            break
        except Exception as e:
            number_of_retries -= 1
            print(f"push_command_add error occurred: {e}")
            if number_of_retries:
                print(
                    f"Failed on try #{max_n_retries - number_of_retries}, pushing again in {n_seconds_sleep} seconds"
                )
                sleep(n_seconds_sleep)
            else:
                raise RuntimeError("push_command_add failed") from e

    time_end = time()
    logging.debug(
        f"push_command_add took {time_end - time_start:.4f} seconds or {(time_end - time_start) / 60.0:.2f} mins"
    )


def push_command_remove(args):
    """
    Commit file deletions using hf_hub client to delete zip file
    Used in: delete_doc_comment.yml
    """
    max_n_retries = args.n_retries + 1
    number_of_retries = args.n_retries
    n_seconds_sleep = 5

    library_name = args.library_name
    doc_version_folder = args.doc_version
    doc_build_repo_id = args.doc_build_repo_id
    commit_msg = args.commit_msg

    api = HfApi()
    zip_file_path = create_zip_name(library_name, doc_version_folder)

    while number_of_retries:
        try:
            api.delete_file(
                zip_file_path, doc_build_repo_id, token=args.token, repo_type=REPO_TYPE, commit_message=commit_msg
            )
            break
        except Exception as e:
            number_of_retries -= 1
            print(f"push_command_remove error occurred: {e}")
            if number_of_retries:
                print(
                    f"Failed on try #{max_n_retries - number_of_retries}, pushing again in {n_seconds_sleep} seconds"
                )
                sleep(n_seconds_sleep)
            else:
                raise RuntimeError("push_command_remove failed") from e


def push_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("push")
    else:
        parser = argparse.ArgumentParser("Doc Builder push command")

    parser.add_argument(
        "library_name",
        type=str,
        help="The name of the library, which also acts as a path where built doc artifacts reside in",
    )
    parser.add_argument(
        "--doc_build_repo_id",
        type=str,
        help="Repo to which doc artifacts will be committed (e.g. `huggingface/doc-build-dev`)",
    )
    parser.add_argument("--token", type=str, help="Github token that has write/push permission to `doc_build_repo_id`")
    parser.add_argument(
        "--commit_msg",
        type=str,
        help="Git commit message",
        default="Github GraphQL createcommitonbranch commit",
    )
    parser.add_argument("--n_retries", type=int, help="Number of push retries in the event of conflict", default=1)
    parser.add_argument(
        "--doc_version",
        type=str,
        default=None,
        help="Version of the generated documentation.",
    )
    parser.add_argument(
        "--is_remove",
        action="store_true",
        help="Whether or not to remove entire folder ('--doc_version') from git tree",
    )
    parser.add_argument(
        "--upload_version_yml",
        action="store_true",
        help="Whether or not to push _version.yml file to git repo",
    )

    if subparsers is not None:
        parser.set_defaults(func=push_command)
    return parser
