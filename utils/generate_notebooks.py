import argparse
import os
import re
import nbformat
import shutil
import yaml

from pathlib import Path

PATH_TO_COURSE = "chapters/en/"

re_framework_test = re.compile(r"^{#if\s+fw\s+===\s+'([^']+)'}\s*$")
re_framework_else = re.compile(r"^{:else}\s*$")
re_framework_end = re.compile(r"^{/if}\s*$")

re_html_line = re.compile(r"^<[^>]*/>\s*$")
re_html_tag = re.compile(r"<([^/>]*)>\s*$")

re_python_code = re.compile(r"^```(?:py|python|py no\-format|python no\-format)\s*$")
re_output_code = re.compile(r"^```(?:py|python)\s+out\s*$")
re_end_code = re.compile(r"^```\s*$")

frameworks = {"pt": "PyTorch", "tf": "TensorFlow"}

def read_and_split_frameworks(fname):
    """
    Read the MDX in fname and creates two versions (if necessary) for each framework.
    """
    with open(fname, "r") as f:
        content = f.readlines()
    
    contents = {"pt": [], "tf": []}
    
    differences = False
    current_content = []
    line_idx = 0
    for line in content:
        if re_framework_test.search(line) is not None:
            differences = True
            framework = re_framework_test.search(line).groups()[0]
            for key in contents:
                contents[key].extend(current_content)
            current_content = []
        elif re_framework_else.search(line) is not None:
            contents[framework].extend(current_content)
            current_content = []
            framework = "pt" if framework == "tf" else "tf"
        elif re_framework_end.search(line) is not None:
            contents[framework].extend(current_content)
            current_content = []
        else:
            current_content.append(line)

    if len(current_content) > 0:
        for key in contents:
            contents[key].extend(current_content)
    
    if differences:
        return {k: "".join(content) for k, content in contents.items()}
    else:
        return "".join(content)

def extract_cells(content):
    """
    Extract the code/output cells from content.
    """
    cells = []
    current_cell = None
    is_output = False
    for line in content.split("\n"):
        if re_python_code.search(line) is not None:
            is_output = False
            current_cell = []
        elif re_output_code.search(line) is not None:
            is_output = True
            current_cell = []
        elif re_end_code.search(line) is not None and current_cell is not None:
            cell = "\n".join(current_cell)
            if is_output:
                if not isinstance(cells[-1], tuple):
                    cells[-1] = (cells[-1], cell)
            else:
                cells.append(cell)
            current_cell = None
            current_md = []
        elif current_cell is not None:
            current_cell.append(line)

    return cells


def convert_to_nb_cell(cell):
    """
    Convert some cell (either just code or tuple (code, output)) to a proper notebook cell.
    """
    nb_cell = {"cell_type": "code", "execution_count": None, "metadata": {}}
    if isinstance(cell, tuple):
        nb_cell["source"] = cell[0]
        nb_cell["outputs"] = [nbformat.notebooknode.NotebookNode({
            'data': {'text/plain': cell[1]},
            'execution_count': None,
            'metadata': {},
            'output_type': 'execute_result',
        })]
    else:
        nb_cell["source"] = cell
        nb_cell["outputs"] = []
    return nbformat.notebooknode.NotebookNode(nb_cell)


def nb_cell(source, code=True):
    if not code:
        return nbformat.notebooknode.NotebookNode(
            {"cell_type": "markdown", "source": source, "metadata": {}}
        )
    return nbformat.notebooknode.NotebookNode(
        {"cell_type": "code", "metadata": {}, "source": source, "execution_count": None, "outputs": []}
    )


def build_notebook(fname, title, output_dir="."):
    """
    Build the notebook for fname with a given title in output_dir.
    """
    sections = read_and_split_frameworks(fname)
    sections_with_accelerate = [
        "A full training",
        "Token classification (PyTorch)",
        "Fine-tuning a masked language model (PyTorch)",
        "Translation (PyTorch)",
        "Summarization (PyTorch)",
        "Training a causal language model from scratch (PyTorch)",
        "Question answering (PyTorch)",
    ]
    sections_with_hf_hub = [
        "Sharing pretrained models (PyTorch)",
        "Sharing pretrained models (TensorFlow)",
        "Creating your own dataset",
        "Token classification (PyTorch)",
        "Token classification (TensorFlow)",
        "Training a new tokenizer from an old one",
        "Fine-tuning a masked language model (PyTorch)",
        "Fine-tuning a masked language model (TensorFlow)",
        "Translation (PyTorch)",
        "Translation (TensorFlow)",
        "Summarization (PyTorch)",
        "Summarization (TensorFlow)",
        "Training a causal language model from scratch (PyTorch)",
        "Training a causal language model from scratch (TensorFlow)",
        "Question answering (PyTorch)",
        "Question answering (TensorFlow)",
        "What to do when you get an error",
    ]
    sections_with_faiss = ["Semantic search with FAISS (PyTorch)", "Semantic search with FAISS (TensorFlow)"]
    stem = Path(fname).stem
    if not isinstance(sections, dict):
        contents = [sections]
        titles = [title]
        fnames = [f"{stem}.ipynb"]
    else:
        contents = []
        titles = []
        fnames = []
        for key, section in sections.items():
            contents.append(section)
            titles.append(f"{title} ({frameworks[key]})")
            fnames.append(f"{stem}_{key}.ipynb")
    
    for title, content, fname in zip(titles, contents, fnames):
        cells = extract_cells(content)
        if len(cells) == 0:
            continue
        
        nb_cells = [
            nb_cell(f"# {title}", code=False),
            nb_cell("Install the Transformers and Datasets libraries to run this notebook.", code=False)
        ]

        # Install cell
        installs = ["!pip install datasets transformers[sentencepiece]"]
        if title in sections_with_accelerate:
            installs.append("!pip install accelerate")
            installs.append("# To run the training on TPU, you will need to uncomment the followin line:")
            installs.append("# !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl")
        if title in sections_with_hf_hub:
            installs.append("!apt install git-lfs")
        if title in sections_with_faiss:
            installs.append("!pip install faiss-gpu")
        
        nb_cells.append(nb_cell("\n".join(installs)))

        if title in sections_with_hf_hub:
            nb_cells.extend([
                nb_cell("You will need to setup git, adapt your email and name in the following cell.", code=False),
                nb_cell("!git config --global user.email \"you@example.com\"\n!git config --global user.name \"Your Name\""),
                nb_cell("You will also need to be logged in to the Hugging Face Hub. Execute the following and enter your credentials.", code=False),
                nb_cell("from huggingface_hub import notebook_login\n\nnotebook_login()"),
            ])
        nb_cells += [convert_to_nb_cell(cell) for cell in cells]
        metadata = {"colab": {"name": title, "provenance": []}}
        nb_dict = {"cells": nb_cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}
        notebook = nbformat.notebooknode.NotebookNode(nb_dict)
        os.makedirs(output_dir, exist_ok=True)
        nbformat.write(notebook, os.path.join(output_dir, fname), version=4)


def get_titles():
    """
    Parse the yaml _chapters.yml to get the correspondence filename to title
    """
    table = yaml.safe_load(open(os.path.join(PATH_TO_COURSE, "_chapters.yml"), "r"))
    result = {}
    for entry in table:
        chapter_name = entry["local"]
        sections = []
        for i, section in enumerate(entry["sections"]):
            if isinstance(section, str):
                result[os.path.join(chapter_name, f"section{i+1}")] = section
            else:
                section_name = section["local"]
                section_title = section["title"]
                if isinstance(section_name, str):
                    result[os.path.join(chapter_name, section_name)] = section_title
                else:
                    if isinstance(section_title, str):
                        section_title = {key: section_title for key in section_name.keys()}
                    for key in section_name.keys():
                        result[os.path.join(chapter_name, section_name[key])] = section_title[key]
    return {k: v for k, v in result.items() if "quiz" not in v}


def create_notebooks(output_dir):
    for folder in os.listdir(output_dir):
        if folder.startswith("chapter"):
            shutil.rmtree(os.path.join(output_dir, folder))
    titles = get_titles()
    for fname, title in titles.items():
        build_notebook(
            os.path.join(PATH_TO_COURSE, f"{fname}.mdx"),
            title,
            os.path.join(output_dir, Path(fname).parent),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Where to output the notebooks")
    args = parser.parse_args()

    create_notebooks(args.output_dir)
