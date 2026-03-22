import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PATH_TO_COURSE = Path("chapters")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ANCHOR_RE = re.compile(r"\[\[([^\]]+)\]\]")
CODE_FENCE_RE = re.compile(r"^```")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass
class SectionStats:
    level: int
    title: str
    anchor: str
    word_count: int = 0
    paragraph_count: int = 0
    list_item_count: int = 0
    code_block_count: int = 0
    iframe_count: int = 0
    image_count: int = 0

    def score(self) -> int:
        return (
            self.word_count
            + self.paragraph_count * 20
            + self.list_item_count * 12
            + self.code_block_count * 30
            + self.iframe_count * 10
            + self.image_count * 8
        )


def strip_anchor(title: str) -> tuple[str, str]:
    match = ANCHOR_RE.search(title)
    anchor = match.group(1) if match else title.strip().lower()
    clean_title = ANCHOR_RE.sub("", title).strip()
    return clean_title, anchor


def normalize_text(line: str) -> str:
    return HTML_TAG_RE.sub(" ", line).strip()


def is_list_item(line: str) -> bool:
    stripped = line.lstrip()
    return bool(re.match(r"^([-*+]\s|\d+\.\s)", stripped))


def should_track_heading(level: int) -> bool:
    return level >= 2


def parse_sections(file_path: Path) -> list[SectionStats]:
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    sections: list[SectionStats] = []
    current: SectionStats | None = None
    in_code_block = False
    paragraph_has_text = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if CODE_FENCE_RE.match(line.strip()):
            in_code_block = not in_code_block
            if current is not None:
                current.code_block_count += 1
            paragraph_has_text = False
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match and not in_code_block:
            level = len(heading_match.group(1))
            title, anchor = strip_anchor(heading_match.group(2))
            if should_track_heading(level):
                current = SectionStats(level=level, title=title, anchor=anchor)
                sections.append(current)
                paragraph_has_text = False
            continue

        if current is None:
            continue

        stripped = line.strip()
        if not stripped:
            paragraph_has_text = False
            continue

        if "<iframe" in stripped:
            current.iframe_count += 1
        if "<img" in stripped or stripped.startswith("!["):
            current.image_count += 1

        if in_code_block:
            continue

        if is_list_item(stripped):
            current.list_item_count += 1
            current.word_count += len(WORD_RE.findall(normalize_text(stripped)))
            paragraph_has_text = False
            continue

        normalized = normalize_text(stripped)
        if not normalized:
            paragraph_has_text = False
            continue

        current.word_count += len(WORD_RE.findall(normalized))
        if not paragraph_has_text:
            current.paragraph_count += 1
            paragraph_has_text = True

    return sections


def align_sections(
    en_sections: list[SectionStats], tr_sections: list[SectionStats]
) -> tuple[list[tuple[SectionStats, SectionStats]], list[SectionStats], list[SectionStats]]:
    pairs: list[tuple[SectionStats, SectionStats]] = []
    used_translation_indexes: set[int] = set()
    english_level_ordinals: dict[int, int] = {}
    translation_level_indexes: dict[int, list[int]] = {}

    for index, tr_section in enumerate(tr_sections):
        translation_level_indexes.setdefault(tr_section.level, []).append(index)

    translation_anchor_index = {
        section.anchor: index
        for index, section in enumerate(tr_sections)
        if section.anchor and section.anchor not in {s.anchor for s in tr_sections[:index]}
    }

    for en_section in en_sections:
        tr_index = translation_anchor_index.get(en_section.anchor)
        if tr_index is not None and tr_index not in used_translation_indexes:
            pairs.append((en_section, tr_sections[tr_index]))
            used_translation_indexes.add(tr_index)
            continue

        ordinal = english_level_ordinals.get(en_section.level, 0)
        english_level_ordinals[en_section.level] = ordinal + 1
        candidates = [
            index
            for index in translation_level_indexes.get(en_section.level, [])
            if index not in used_translation_indexes
        ]
        if ordinal < len(candidates):
            tr_index = candidates[ordinal]
            pairs.append((en_section, tr_sections[tr_index]))
            used_translation_indexes.add(tr_index)

    unmatched_english = [
        en_section
        for en_section in en_sections
        if not any(pair[0] is en_section for pair in pairs)
    ]
    unmatched_translation = [
        tr_section
        for index, tr_section in enumerate(tr_sections)
        if index not in used_translation_indexes
    ]
    return pairs, unmatched_english, unmatched_translation


def load_sections_from_toc(language: str) -> list[str]:
    import yaml

    toc_path = PATH_TO_COURSE / language / "_toctree.yml"
    toc = yaml.safe_load(toc_path.read_text(encoding="utf-8"))
    sections: list[str] = []
    for chapter in toc:
        for section in chapter["sections"]:
            sections.append(section["local"])
            local_fw = section.get("local_fw", {})
            sections.extend(local_fw.values())
    return sections


def compare_file(relative_path: str, language: str, warn_ratio: float, severe_ratio: float) -> list[str]:
    english_path = PATH_TO_COURSE / "en" / f"{relative_path}.mdx"
    translation_path = PATH_TO_COURSE / language / f"{relative_path}.mdx"
    messages: list[str] = []

    if not english_path.exists() or not translation_path.exists():
        return messages

    en_sections = parse_sections(english_path)
    tr_sections = parse_sections(translation_path)
    pairs, unmatched_english, unmatched_translation = align_sections(en_sections, tr_sections)

    for section in unmatched_english:
        messages.append(
            f"SEVERE {relative_path}: missing subsection at level {section.level} matching English heading '{section.title}'"
        )

    for section in unmatched_translation:
        messages.append(
            f"WARN   {relative_path}: extra translated subsection at level {section.level} titled '{section.title}'"
        )

    for en_stats, tr_stats in pairs:
        anchor = en_stats.anchor

        en_score = en_stats.score()
        tr_score = tr_stats.score()
        if en_score == 0:
            continue

        ratio = tr_score / en_score
        paragraph_gap = en_stats.paragraph_count - tr_stats.paragraph_count
        list_gap = en_stats.list_item_count - tr_stats.list_item_count

        if ratio < severe_ratio:
            messages.append(
                f"SEVERE {relative_path}#{anchor}: translated content score is {ratio:.2f} of English "
                f"(en={en_score}, {language}={tr_score}, paragraphs {tr_stats.paragraph_count}/{en_stats.paragraph_count}, "
                f"lists {tr_stats.list_item_count}/{en_stats.list_item_count})"
            )
        elif ratio < warn_ratio or paragraph_gap >= 2 or list_gap >= 3:
            messages.append(
                f"WARN   {relative_path}#{anchor}: translated content score is {ratio:.2f} of English "
                f"(en={en_score}, {language}={tr_score}, paragraphs {tr_stats.paragraph_count}/{en_stats.paragraph_count}, "
                f"lists {tr_stats.list_item_count}/{en_stats.list_item_count})"
            )

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare English and translated MDX files section by section to detect overly compressed translations."
    )
    parser.add_argument("--language", required=True, help="Translation language, for example 'es'.")
    parser.add_argument(
        "--section",
        action="append",
        help="Specific section path to validate, for example chapter1/8. Can be passed multiple times.",
    )
    parser.add_argument(
        "--warn-ratio",
        type=float,
        default=0.60,
        help="Warn when translated section score falls below this ratio of the English score.",
    )
    parser.add_argument(
        "--severe-ratio",
        type=float,
        default=0.42,
        help="Mark as severe when translated section score falls below this ratio of the English score.",
    )
    args = parser.parse_args()

    relative_paths = args.section if args.section else load_sections_from_toc(args.language)
    relative_paths = sorted(dict.fromkeys(relative_paths))

    findings: list[str] = []
    for relative_path in relative_paths:
        findings.extend(
            compare_file(
                relative_path,
                args.language,
                warn_ratio=args.warn_ratio,
                severe_ratio=args.severe_ratio,
            )
        )

    if findings:
        print("Translation depth findings:\n")
        for finding in findings:
            print(finding)
    else:
        print("✅ No subsection depth mismatches detected with the current thresholds.")


if __name__ == "__main__":
    main()