import argparse
import black
import os
import re
from pathlib import Path


def blackify(filename, check_only=False):
    # Read the content of the file
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")

    # Split the content into code samples in py or python blocks.
    code_samples = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if line.strip() in ["```py", "```python"]:
            line_index += 1
            start_index = line_index
            while line_index < len(lines) and lines[line_index].strip() != "```":
                line_index += 1

            code = "\n".join(lines[start_index:line_index])
            # Deal with ! instructions
            code = re.sub(r"^!", r"## !", code, flags=re.MULTILINE)

            code_samples.append({"start_index": start_index, "end_index": line_index - 1, "code": code})
            line_index += 1
        else:
            line_index += 1

    # Let's blackify the code! We put everything in one big text to go faster.
    delimiter = "\n\n### New cell ###\n"
    full_code = delimiter.join([sample["code"] for sample in code_samples])
    formatted_code = full_code.replace("\t", "    ")
    formatted_code = black.format_str(formatted_code, mode=black.FileMode({black.TargetVersion.PY37}, line_length=90))

    # Black adds last new lines we don't want, so we strip individual code samples.
    cells = formatted_code.split(delimiter)
    cells = [cell.strip() for cell in cells]
    formatted_code = delimiter.join(cells)

    if check_only:
        return full_code == formatted_code
    elif full_code == formatted_code:
        # Nothing to do, all is good
        return

    formatted_code = re.sub(r"^## !", r"!", formatted_code, flags=re.MULTILINE)
    print(f"Formatting {filename}")
    # Re-build the content with formatted code
    new_lines = []
    start_index = 0
    for sample, code in zip(code_samples, formatted_code.split(delimiter)):
        new_lines.extend(lines[start_index : sample["start_index"]])
        new_lines.append(code)
        start_index = sample["end_index"] + 1
    new_lines.extend(lines[start_index:])

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


def format_all_files(check_only=False):
    failures = []
    for filename in Path("chapters").glob("**/*.mdx"):
        try:
            same = blackify(filename, check_only=check_only)
            if check_only and not same:
                failures.append(filename)
        except Exception:
            print(f"Failed to format {filename}.")
            raise

    if check_only and len(failures) > 0:
        raise ValueError(f"{len(failures)} files need to be formatted, run `make style`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Just check files are properly formatted.",
    )
    args = parser.parse_args()

    format_all_files(check_only=args.check_only)
