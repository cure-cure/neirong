"""Repair UTF-8 ZIP member names that Linux unzip decoded as CP866.

The uploaded archives do not mark their non-ASCII member names as UTF-8.
On this AutoDL image, ``unzip`` therefore produced reversible mojibake such as
``хЫ╛хГП`` instead of ``图像``.  This utility renames path components
only; it never changes file contents.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def repaired_name(name: str) -> str:
    try:
        return name.encode("cp866").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def repair_tree(root: Path, *, apply: bool) -> list[tuple[Path, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    changes: list[tuple[Path, Path]] = []
    for source in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        new_name = repaired_name(source.name)
        if new_name == source.name:
            continue
        target = source.with_name(new_name)
        if target.exists():
            raise FileExistsError(f"Refusing filename collision: {source} -> {target}")
        changes.append((source, target))
        if apply:
            source.rename(target)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changes = repair_tree(args.root, apply=args.apply)
    mode = "renamed" if args.apply else "would rename"
    for source, target in changes:
        print(f"{source} -> {target}")
    print(f"{mode}: {len(changes)} path components")


if __name__ == "__main__":
    main()
