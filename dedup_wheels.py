"""
Split wheels by set of architectures and delete redundant ones.
"""

import argparse
import os
import pathlib
import re
import sys

from collections import defaultdict
from typing import Iterable


def list_wheels(wheel_dir: pathlib.Path):
    return (wheel.name for wheel in wheel_dir.glob("*.whl"))


def dedup_wheels(wheel_names: Iterable[str]):
    split_wheelname = re.compile(r"(?P<project>\w+-[0-9a-z.]+)-(?P<python>[^-]+-[^-]+)-(?P<archs>(?:[^.]+[.])+)whl").match

    keep: set = set()
    seen: dict[tuple[str,str],list[set[str]]] = defaultdict(list)
    best: dict[tuple[str,str],list[str]] = defaultdict(list)

    for wheel in sorted(wheel_names, key=len, reverse=True):
        parts = split_wheelname(wheel)
        if not parts:
            keep.add(wheel)
            continue

        archs = set(parts['archs'].rstrip('.').split('.'))
        key = (parts['project'], parts['python'])
        for archs_seen in seen[key]:
            if not (archs - archs_seen):
                break
        else:
            seen[key].append(archs)
            best[key].append(wheel)

    keep.update(wheel for wheel_list in best.values() for wheel in wheel_list)
    return sorted(keep)


def print_wheels(wheel_names: Iterable[str]):
    for wheel in sorted(wheel_names):
        print(f"  {wheel}")


def main(wheel_dir: str, delete=True):
    wheel_path = pathlib.Path(wheel_dir)
    all_wheels = list(list_wheels(wheel_path))
    wheels = dedup_wheels(all_wheels)
    redundant = set(all_wheels).difference(wheels)

    if delete:
        for wheel in sorted(redundant):
            print(f"deleting {wheel}")
            os.unlink(wheel_path / wheel)
    elif redundant:
        print("Redundant wheels found:")
        print_wheels(redundant)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-d", "--delete", action="store_true")

    return parser.parse_args(args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args(sys.argv[1:])
        main(args.directory, delete=args.delete)
    else:
        print(f"Usage: {sys.argv[0]} WHEEL_DIR", file=sys.stderr)
