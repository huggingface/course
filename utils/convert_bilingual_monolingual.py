import re
import argparse
from pathlib import Path

PATTERN_TIMESTAMP = re.compile(
    "^[0-9][0-9]:[0-9][0-9]:[0-9][0-9],[0-9][0-9][0-9] --> [0-9][0-9]:[0-9][0-9]:[0-9][0-9],[0-9][0-9][0-9]"
)
PATTERN_NUM = re.compile("\\d+")


def convert(input_file, output_file):
    """
    Convert bilingual caption file to monolingual caption. Supported caption file type is SRT.
    """
    line_count = 0
    with open(input_file) as file:
        with open(output_file, "w") as output_file:
            for line in file:
                if line_count == 0:
                    line_count += 1
                    output_file.write(line)
                elif PATTERN_TIMESTAMP.match(line):
                    line_count += 1
                    output_file.write(line)
                elif line == "\n":
                    line_count = 0
                    output_file.write(line)
                else:
                    if line_count == 2:
                        output_file.write(line)
                    line_count += 1
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_language_folder", type=str, help="Folder with input bilingual SRT files to be converted"
    )
    parser.add_argument(
        "--output_language_folder",
        type=str,
        default="tmp-subtitles",
        help="Folder to store converted monolingual SRT files",
    )
    args = parser.parse_args()

    output_path = Path(args.output_language_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_files = Path(args.input_language_folder).glob("*.srt")
    for input_file in input_files:
        convert(input_file, output_path / input_file.name)
    print(f"Succesfully converted {len(list(input_files))} files to {args.output_language_folder} folder")
