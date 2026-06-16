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


import argparse
import sys

from doc_builder.check_links import check_links


def check_links_command(args):
    """Run the link checker command."""
    print(f"Checking links in {args.path_to_docs}...")
    result = check_links(args.path_to_docs, max_workers=args.max_workers, show_progress=not args.no_progress)

    # Choose output format
    if args.format == "list":
        print(result.get_list_output())
    else:
        print(result.get_summary())

    if result.has_broken_links():
        sys.exit(1)


def check_links_command_parser(subparsers=None):
    """Create the argument parser for the check-links command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "check-links", help="Check internal links in documentation files for broken references"
        )
    else:
        parser = argparse.ArgumentParser("Doc Builder check-links command")

    parser.add_argument("path_to_docs", type=str, help="Path to the documentation folder to check")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto-detect based on CPU count)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["detailed", "list"],
        default="detailed",
        help="Output format: 'detailed' (default, shows link text and URL) or 'list' (compact, file:line - URL)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (useful for CI/CD or when tqdm is not available)",
    )

    if subparsers is not None:
        parser.set_defaults(func=check_links_command)
    return parser
