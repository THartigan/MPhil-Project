#!/usr/bin/env python3
"""Count a report while excluding narrative author attributions.

The source tree is copied to a temporary directory before author names are
removed, so the LaTeX sources themselves are never modified.  TeXcount then
applies the report's existing inclusion and weighting rules to that copy.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IGNORED_DIRECTORIES = {".git", "output", "tmp"}

# Match normal capitalised surnames without also treating acronyms such as
# LION, ASTRA or CT as author names.
SURNAME = r"[A-Z\u00c0-\u00d6\u00d8-\u00de][a-z\u00df-\u00f6\u00f8-\u00ff]+(?:[-'\u2019][A-Z\u00c0-\u00d6\u00d8-\u00de]?[a-z\u00df-\u00f6\u00f8-\u00ff]+)*"

NAMED_REFERENCE_PATTERNS = (
    # Hu et al., Ozdenizci et al. and possessive forms such as Hu et al.'s.
    re.compile(rf"(?<![\w]){SURNAME}\s+et(?:\s|~)+al\.(?:['\u2019]s)?"),
    # Narrative two- or three-author forms immediately followed by a citation,
    # for example Anderson and Song \cite{...}.
    re.compile(
        rf"(?<![\w]){SURNAME}(?:\s*,\s*{SURNAME})*"
        rf"\s+(?:and|\\&)\s+{SURNAME}(?:['\u2019]s)?"
        rf"(?=\s*~?\\(?:cite|parencite|textcite|autocite)\b)"
    ),
)


def remove_named_references(source: str) -> tuple[str, list[str]]:
    """Remove narrative author names and return the removed phrases."""

    removed: list[str] = []
    for pattern in NAMED_REFERENCE_PATTERNS:
        source = pattern.sub(lambda match: removed.append(match.group(0)) or "", source)
    return source, removed


def copy_preprocessed_sources(destination: Path) -> list[tuple[Path, str]]:
    """Mirror the project's TeX sources, excluding named references."""

    exclusions: list[tuple[Path, str]] = []
    for source_path in PROJECT_ROOT.rglob("*.tex"):
        relative_path = source_path.relative_to(PROJECT_ROOT)
        if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
            continue

        processed, removed = remove_named_references(
            source_path.read_text(encoding="utf-8")
        )
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(processed, encoding="utf-8")
        exclusions.extend((relative_path, phrase) for phrase in removed)

    return exclusions


def run_texcount(main_file: Path, working_directory: Path) -> int:
    """Run TeXcount using the report's established weighting policy."""

    texcount = shutil.which("texcount")
    if texcount is None:
        raise RuntimeError("texcount is not installed")

    result = subprocess.run(
        [
            texcount,
            "-inc",
            "-sum=1,1,1,0,0,0,0",
            "-nobib",
            "-1",
            str(main_file),
        ],
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )

    for line in reversed(result.stdout.splitlines()):
        if re.fullmatch(r"\s*\d+\s*", line):
            return int(line)
    raise RuntimeError("texcount did not return a numeric total")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count report words without narrative author attributions."
    )
    parser.add_argument("document", type=Path, help="main LaTeX document")
    parser.add_argument(
        "--show-exclusions",
        action="store_true",
        help="list excluded author phrases on standard error",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_document = arguments.document.resolve()
    try:
        relative_document = source_document.relative_to(PROJECT_ROOT)
    except ValueError:
        print("document must be inside the report source tree", file=sys.stderr)
        return 2

    if not source_document.is_file():
        print(f"document not found: {source_document}", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="report-wordcount-") as temporary:
            temporary_root = Path(temporary)
            exclusions = copy_preprocessed_sources(temporary_root)
            count = run_texcount(relative_document, temporary_root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1

    if arguments.show_exclusions:
        for path, phrase in exclusions:
            print(f"{path}: {phrase}", file=sys.stderr)
        print(f"Excluded {len(exclusions)} named references.", file=sys.stderr)

    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
