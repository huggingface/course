"""Validate the state of a course translation against the English source.

This reports, for a given language:

* a progress summary (how many sections are translated vs. the English total),
* sections referenced in the language's ``_toctree.yml`` whose ``.mdx`` file is
  missing on disk -- these are hard errors that break the ``doc-builder`` build,
* sections that still need to be translated.

Example:

    python utils/validate_translation.py --language fr
"""

import argparse
import os
from pathlib import Path

import yaml

PATH_TO_COURSE = Path("chapters/")


def load_sections(language: str):
    """Return the set of section paths declared in a language's ``_toctree.yml``.

    A section may declare a single ``local`` path or framework-specific
    ``local_fw`` paths (one each for the PyTorch and TensorFlow variants). We
    collect all of them so validation works regardless of which style a chapter
    uses, and so framework-specific files are not silently ignored.
    """
    toc = yaml.safe_load(
        open(os.path.join(PATH_TO_COURSE / language, "_toctree.yml"), "r")
    )
    sections = []
    for chapter in toc:
        for section in chapter["sections"]:
            if "local" in section:
                sections.append(section["local"])
            if "local_fw" in section:
                sections.extend(section["local_fw"].values())
    return set(sections)


def find_missing_files(language: str, sections):
    """Return sections declared in ``_toctree.yml`` whose ``.mdx`` file is absent.

    The course can only be built when every section referenced by the table of
    contents has a matching file on disk, so a missing file is a hard error
    rather than simply untranslated content.
    """
    missing_files = []
    for section in sorted(sections):
        mdx_path = PATH_TO_COURSE / language / f"{section}.mdx"
        if not mdx_path.exists():
            missing_files.append(section)
    return missing_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="Translation language to validate")
    args = parser.parse_args()

    english_sections = load_sections("en")
    translation_sections = load_sections(args.language)
    missing_sections = sorted(english_sections.difference(translation_sections))
    missing_files = find_missing_files(args.language, translation_sections)

    total = len(english_sections)
    completed = total - len(missing_sections)
    percentage = (completed / total * 100) if total else 0
    print(
        f"📊 '{args.language}' translation progress: "
        f"{completed}/{total} sections ({percentage:.1f}%)\n"
    )

    if missing_files:
        print("❌ Sections listed in _toctree.yml but missing their .mdx file")
        print("   (these break the course build):\n")
        for section in missing_files:
            print(f"  - {section}.mdx")
        print()

    if missing_sections:
        print("📝 Sections not yet translated:\n")
        for section in missing_sections:
            print(f"  - {section}")
    else:
        print("✅ No missing sections - translation complete!")

    # Exit with a non-zero status when the table of contents references files
    # that do not exist, so the script can double as a CI gate.
    if missing_files:
        raise SystemExit(1)
