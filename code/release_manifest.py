#!/usr/bin/env python3
"""Create or verify the frozen SHA-256 release manifest.

The manifest is intentionally not rewritten during an ordinary scientific
rerun: regenerated floating-point and PDF files are compared by the
tolerance-based validator instead.  Run this helper only after the curated
release tree is final.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "outputs_local",
    "texmf",
    "tmp",
    "venv",
}
EXCLUDED_NAMES = {
    "MANIFEST.sha256",
    ".DS_Store",
    # Superseded by data/locked_branch_declared_metric.csv.
    "locked_branch_physical_metric.csv",
    "mainNotes.bib",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
    ".synctex.gz",
}


def is_release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name in EXCLUDED_NAMES:
        return False
    curated_hidden_files = {".gitattributes", ".gitignore"}
    if relative.name.startswith(".") and relative.name not in curated_hidden_files:
        return False
    return not any(relative.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def current_entries() -> list[tuple[str, str]]:
    files = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and is_release_file(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    return [
        (digest(path), f"./{path.relative_to(ROOT).as_posix()}") for path in files
    ]


def write_manifest() -> None:
    entries = current_entries()
    text = "".join(f"{checksum}  {relative}\n" for checksum, relative in entries)
    temporary = MANIFEST.with_suffix(".sha256.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(MANIFEST)
    print(f"Wrote {MANIFEST} with {len(entries)} files.")


def read_manifest() -> list[tuple[str, str]]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"missing manifest: {MANIFEST}")
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"malformed manifest line {line_number}: {line!r}"
            ) from exc
        entries.append((checksum, relative))
    return entries


def verify_manifest() -> None:
    expected = read_manifest()
    actual = current_entries()
    if expected == actual:
        print(f"Verified {len(actual)} files against {MANIFEST}.")
        return

    expected_map = {relative: checksum for checksum, relative in expected}
    actual_map = {relative: checksum for checksum, relative in actual}
    missing = sorted(set(expected_map) - set(actual_map))
    unexpected = sorted(set(actual_map) - set(expected_map))
    changed = sorted(
        relative
        for relative in set(expected_map) & set(actual_map)
        if expected_map[relative] != actual_map[relative]
    )
    lines = ["release manifest verification failed"]
    if missing:
        lines.append(f"missing ({len(missing)}): {', '.join(missing)}")
    if unexpected:
        lines.append(f"unexpected ({len(unexpected)}): {', '.join(unexpected)}")
    if changed:
        lines.append(f"changed ({len(changed)}): {', '.join(changed)}")
    raise RuntimeError("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "verify"))
    args = parser.parse_args()
    if args.action == "write":
        write_manifest()
    else:
        verify_manifest()


if __name__ == "__main__":
    main()
