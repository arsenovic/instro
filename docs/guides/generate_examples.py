"""Generate Mintlify example pages and per-category index pages from ../examples/.

Walks every ``*.py`` under ``examples/`` (relative to the repo root), writes a
matching ``.mdx`` page under ``docs/guides/examples/``, and writes one
``index.mdx`` per category folder listing links to that category's pages.

Also walks ``examples/`` directories inside ``packages/instro-unstable/`` and
emits pages under ``examples/unstable/<submodule>/``, each with a warning
callout that the API is not stable. All unstable examples share a single
``examples/unstable/index.mdx``, with one heading per submodule, so a new
unstable submodule needs no ``docs.json`` edit.

Unlike earlier versions of this script, it does not touch ``docs.json``.
``docs.json``'s Examples tab has one static entry per category pointing at
that category's ``index.mdx``; it doesn't change when individual example
scripts are added, removed, or renamed, so there's nothing for this script to
regenerate there. Adding or removing a whole category is the one case that
still needs a manual ``docs.json`` edit (for non-unstable categories).

Run via ``just gen-examples``.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EXAMPLES_SRC = REPO_ROOT / "examples"
UNSTABLE_SRC = REPO_ROOT / "packages" / "instro-unstable" / "instro" / "unstable"
EXAMPLES_OUT = SCRIPT_DIR / "examples"

CATEGORY_TITLES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("daq", "DAQ"),
        ("dmm", "DMM"),
        ("psu", "PSU"),
        ("eload", "Electronic Load"),
        ("i2c", "I2C"),
        ("publishers", "Publishers"),
        ("modbus", "Modbus"),
        ("ethernetip", "EtherNet/IP"),
        ("test_rack_example", "Test Rack"),
        ("vna", "VNA"),
        ("flowcontroller", "Flow Controller"),
        ("motorcontroller", "Motor Controller"),
    ]
)

ROOT_CATEGORY = "general"
ROOT_CATEGORY_TITLE = "General"

_UNSTABLE_WARNING = """\
<Warning>
  This example uses `instro-unstable`. This code is new and may change without notice.
</Warning>

"""


def category_title(folder: str) -> str:
    return CATEGORY_TITLES.get(folder, folder.replace("_", " ").title())


def extract_title(py_path: Path) -> str:
    docstring = ast.get_docstring(ast.parse(py_path.read_text()))
    if not docstring:
        return py_path.stem
    first = docstring.strip().splitlines()[0].strip()
    if first.lower().startswith("example:"):
        first = first[len("example:") :].strip()
    return first.rstrip(".") or py_path.stem


def write_mdx(py_path: Path, out_path: Path, *, unstable: bool = False) -> None:
    title = extract_title(py_path)
    body = py_path.read_text()
    if not body.endswith("\n"):
        body += "\n"
    warning = _UNSTABLE_WARNING if unstable else ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f'---\ntitle: "{title}"\n---\n\n{warning}```python {py_path.name}\n{body}```\n')


def write_index(index_path: Path, title: str, entries: list[tuple[str, str]]) -> None:
    """entries: (page_title, nav_path) pairs, already in display order."""
    lines = [f'---\ntitle: "{title}"\n---\n\n']
    lines += [f"- [{entry_title}](/{nav_path})\n" for entry_title, nav_path in entries]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(lines))


def write_unstable_index(index_path: Path, sections: "OrderedDict[str, list[tuple[str, str]]]") -> None:
    """sections: submodule folder -> (page_title, nav_path) pairs, already in display order."""
    lines = ['---\ntitle: "Unstable"\n---\n\n', _UNSTABLE_WARNING.replace("This example uses", "These examples use")]
    for folder, entries in sections.items():
        lines.append(f"## {category_title(folder)}\n\n")
        lines += [f"- [{entry_title}](/{nav_path})\n" for entry_title, nav_path in entries]
        lines.append("\n")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("".join(lines).rstrip("\n") + "\n")


def clean_output_dir(output_path: Path) -> None:
    if not output_path.exists():
        return
    for mdx in output_path.rglob("*.mdx"):
        mdx.unlink()
    for d in sorted(
        (p for p in output_path.rglob("*") if p.is_dir()),
        key=lambda p: -len(p.parts),
    ):
        if not any(d.iterdir()):
            d.rmdir()


def main(output_path: Path) -> None:
    clean_output_dir(output_path)

    categories: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    root_entries: list[tuple[str, str]] = []

    for py_path in sorted(EXAMPLES_SRC.rglob("*.py")):
        rel = py_path.relative_to(EXAMPLES_SRC)
        nav_path = f"examples/{rel.with_suffix('').as_posix()}"
        out_path = (output_path / rel).with_suffix(".mdx")
        write_mdx(py_path, out_path)
        print(f"wrote {out_path.relative_to(SCRIPT_DIR)}")

        title = extract_title(py_path)
        if len(rel.parts) == 1:
            root_entries.append((title, nav_path))
        else:
            categories.setdefault(rel.parts[0], []).append((title, nav_path))

    unstable_categories: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for py_path in sorted(UNSTABLE_SRC.rglob("examples/*.py")):
        category = py_path.parent.parent.name
        nav_path = f"examples/unstable/{category}/{py_path.stem}"
        out_path = output_path / "unstable" / category / py_path.with_suffix(".mdx").name
        write_mdx(py_path, out_path, unstable=True)
        print(f"wrote {out_path.relative_to(SCRIPT_DIR)}")

        title = extract_title(py_path)
        unstable_categories.setdefault(category, []).append((title, nav_path))

    for folder, entries in categories.items():
        index_path = output_path / folder / "index.mdx"
        write_index(index_path, category_title(folder), entries)
        print(f"wrote {index_path.relative_to(SCRIPT_DIR)}")

    if root_entries:
        index_path = output_path / ROOT_CATEGORY / "index.mdx"
        write_index(index_path, ROOT_CATEGORY_TITLE, root_entries)
        print(f"wrote {index_path.relative_to(SCRIPT_DIR)}")

    if unstable_categories:
        index_path = output_path / "unstable" / "index.mdx"
        write_unstable_index(index_path, OrderedDict(sorted(unstable_categories.items())))
        print(f"wrote {index_path.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main(EXAMPLES_OUT)
