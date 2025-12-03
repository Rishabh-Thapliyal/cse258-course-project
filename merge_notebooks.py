#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional


def read_notebook(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_notebook(nb: Dict[str, Any], path: str) -> None:
    # Preserve outputs by writing the combined JSON unchanged
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def make_separator_cell(title: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": f"# {title}\n"
    }


def merge_two_notebooks(nb_a: Dict[str, Any], nb_b: Dict[str, Any], title_b: Optional[str] = None) -> Dict[str, Any]:
    # Choose nbformat and minor versions conservatively
    nbformat = max(nb_a.get("nbformat", 4), nb_b.get("nbformat", 4))
    nbformat_minor = max(nb_a.get("nbformat_minor", 0), nb_b.get("nbformat_minor", 0))

    merged: Dict[str, Any] = {
        "nbformat": nbformat,
        "nbformat_minor": nbformat_minor,
        "metadata": nb_a.get("metadata", {}),
        "cells": []
    }

    cells_a: List[Dict[str, Any]] = nb_a.get("cells", [])
    cells_b: List[Dict[str, Any]] = nb_b.get("cells", [])

    merged["cells"].extend(cells_a)
    if title_b:
        merged["cells"].append(make_separator_cell(title_b))
    merged["cells"].extend(cells_b)

    return merged


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Merge two Jupyter notebooks (including outputs) into one.")
    parser.add_argument("first", help="Path to first .ipynb (will appear first)")
    parser.add_argument("second", help="Path to second .ipynb (will appear after the first)")
    parser.add_argument("-o", "--output", default="Combined_3.ipynb", help="Output .ipynb path")
    parser.add_argument("--no-separator", action="store_true", help="Do not insert a markdown separator between notebooks")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.first):
        print(f"Error: file not found: {args.first}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.second):
        print(f"Error: file not found: {args.second}", file=sys.stderr)
        return 1

    nb_a = read_notebook(args.first)
    nb_b = read_notebook(args.second)

    title_b_val: Optional[str] = None if args.no_separator else f"Merged from: {os.path.basename(args.second)}"
    merged = merge_two_notebooks(nb_a, nb_b, title_b=title_b_val)

    write_notebook(merged, args.output)
    print(f"Wrote merged notebook to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


