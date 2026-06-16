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

import argparse
import base64
import io
from pathlib import Path

import nbformat
from PIL import Image
from tqdm import tqdm

from ..style_doc import format_code_example


def png_to_jpeg(png_base64):
    png_bytes = base64.b64decode(png_base64)
    png_image = Image.open(io.BytesIO(png_bytes))
    png_image = png_image.convert("RGB")
    jpeg_bytes = io.BytesIO()
    png_image.save(jpeg_bytes, format="JPEG")
    jpeg_base64 = base64.b64encode(jpeg_bytes.getvalue()).decode("utf-8")
    return jpeg_base64


def notebook_to_mdx(notebook, max_len):
    content = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code = cell["source"]
            outputs = [
                o
                for o in cell["outputs"]
                if ("text" in o and o.get("name", None) == "stdout")
                or "text/plain" in o
                or ("data" in o and ("image/png" in o["data"] or "image/jpeg" in o["data"]))
            ]

            if len(outputs) > 0:
                code_lines = code.split("\n")
                # We can add >>> everywhere without worrying as format_code_example will replace them by ...
                # when needed.
                code_lines = [f">>> {line}" if not len(line) == 0 or line.isspace() else line for line in code_lines]
                code = "\n".join(code_lines)
                code = format_code_example(code, max_len=max_len)[0]
                content.append(f"```python\n{code}\n```")

                first_output = outputs[0]
                if "text" in first_output:
                    output = first_output["text"]
                    output = f"<pre>\n{output.strip()}\n</pre>"
                elif "text/plain" in first_output:
                    output = first_output["text/plain"]
                    output = f"<pre>\n{output.strip()}\n</pre>"
                elif "data" in first_output and "image/png" in first_output["data"]:
                    jpeg_base64 = png_to_jpeg(first_output["data"]["image/png"])
                    output = f'<img src="data:image/jpeg;base64,{jpeg_base64}">\n'
                elif "data" in first_output and "image/jpeg" in first_output["data"]:
                    output = f'<img src="data:image/jpeg;base64,{first_output["data"]["image/jpeg"]}">\n'

                content.append(output)
            else:
                code = format_code_example(code, max_len=max_len)[0]
                content.append(f"```python\n{code}\n```")
        elif cell["cell_type"] == "markdown":
            content.append(cell["source"])
        else:
            content.append(f"```\n{cell['source']}\n```")

    mdx_content = "\n\n".join(content)
    return mdx_content


def notebook_to_mdx_command(args):
    src_path = Path(args.notebook_src).resolve()
    src_dir = src_path.parent if src_path.is_file() else src_path
    notebook_paths = [src_path] if src_path.is_file() else [*src_dir.glob("**/*.ipynb")]

    for notebook_path in tqdm(notebook_paths, desc="Converting .ipynb files to .md files"):
        notebook = nbformat.read(notebook_path, as_version=4)
        mdx_content = notebook_to_mdx(notebook, args.max_len)
        mdx_file_name = notebook_path.name[: -len(".ipynb")] + ".md"
        output_dir = notebook_path.parent if args.output_dir is None else Path(args.output_dir).resolve()
        dest_file_path = output_dir / mdx_file_name

        if src_path.is_dir() and args.open_notebook_prefix is not None:
            relative_path = notebook_path.relative_to(src_path)
            colab_link = f"{args.open_notebook_prefix}/{str(relative_path)}"
            colab_link_component = f'<DocNotebookDropdown classNames="absolute z-10 right-0 top-0" options={{[{{label: "Google Colab", value: "{colab_link}"}}]}} />'
            mdx_content = f"{colab_link_component}\n\n" + mdx_content

        with open(dest_file_path, "w", encoding="utf-8") as f:
            f.write(mdx_content)


def notebook_to_mdx_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("notebook-to-mdx")
    else:
        parser = argparse.ArgumentParser("Doc Builder convert notebook to MD command")

    parser.add_argument("notebook_src", type=str, help="The notebook or directory containing notebook(s) to convert.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where the markdown files will be. Defaults to notebook source dir.",
        default=None,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=119,
        help="The number of maximum characters per line.",
    )
    parser.add_argument(
        "--open_notebook_prefix",
        type=str,
        default=None,
        help="Example: https://colab.research.google.com/github/{user}/{repo}/blob/{branch}",
    )

    if subparsers is not None:
        parser.set_defaults(func=notebook_to_mdx_command)
    return parser
