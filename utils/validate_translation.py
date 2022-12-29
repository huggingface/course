import argparse
import os
import yaml

from pathlib import Path

PATH_TO_COURSE = Path("chapters/")


def load_sections(language: str):
    toc = yaml.safe_load(open(os.path.join(PATH_TO_COURSE / language, "_toctree.yml"), "r"))
    sections = []
    for chapter in toc:
        for section in chapter["sections"]:
            sections.append(section["local"])
    return set(sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Translation language to validate")
    args = parser.parse_args()

    english_sections = load_sections("en")
    translation_sections = load_sections(args.language)
    missing_sections = sorted(english_sections.difference(translation_sections))

    if len(missing_sections) > 0:
        print("Completed sesions:\n")
        for section in sorted(translation_sections):
            print(section)

        print("\nMissing sections:\n")
        for section in missing_sections:
            print(section)
    else:
        print("✅ No missing sections - translation complete!")
