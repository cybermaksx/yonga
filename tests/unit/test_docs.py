"""The user documentation agrees with the code it describes.

Each check here guards a statement a real install run found to be wrong: a
command missing from the README, CPU-torch advice that broke the exporter, and
diagnostic codes a user could hit with no entry in ``TROUBLESHOOTING.md``. They
read the documents from the repository checkout, which is where the unit suite
runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from yonga.cli import app

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

#: Spelled-out counts the README may use for its command overview.
NUMBER_WORDS: dict[str, int] = {
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty-four": 24,
    "twenty-five": 25,
}

#: PyTorch's CPU wheel index, as the install documents name it.
CPU_INDEX = "download.pytorch.org/whl/cpu"


def top_level_commands() -> set[str]:
    """Every command ``yonga --help`` lists, groups included."""
    group = typer.main.get_command(app)
    commands = getattr(group, "commands", {})
    return {name for name, command in commands.items() if not command.hidden}


def what_works_today() -> str:
    """The README section with the command overview."""
    text = README.read_text(encoding="utf-8")
    start = text.index("## What works today")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_the_readme_overview_has_a_row_for_every_command() -> None:
    """A command that ships has a row; a row names a command that ships."""
    rows = re.findall(r"^\| `yonga ([a-z-]+)", what_works_today(), flags=re.MULTILINE)
    assert set(rows) == top_level_commands()


def test_the_readme_counts_the_commands_it_lists() -> None:
    """``Sixteen commands`` went stale when ``self`` arrived."""
    match = re.search(r"^([A-Z][a-z-]+) commands;", what_works_today(), flags=re.MULTILINE)
    assert match is not None
    assert NUMBER_WORDS[match.group(1).lower()] == len(top_level_commands())


def test_cpu_torch_advice_takes_torchvision_from_the_same_index() -> None:
    """``optimum-intel`` requires torchvision; PyPI's build cannot load against a CPU torch."""
    documents = [README, *sorted(DOCS.glob("*.md"))]
    advice = [
        (document.name, line.strip())
        for document in documents
        for line in document.read_text(encoding="utf-8").splitlines()
        if CPU_INDEX in line and re.search(r"\bpip install\b.*\btorch\b", line)
    ]
    assert advice, "no CPU-torch install command found; update this test"
    for name, line in advice:
        assert "torchvision" in line, f"{name}: {line}"


def test_cpu_torchvision_repair_replaces_the_pypi_build() -> None:
    """``pip install torchvision`` again keeps the PyPI build: it already satisfies pip."""
    text = (DOCS / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    assert "pip uninstall -y torchvision" in text
    assert f"pip install torchvision --index-url https://{CPU_INDEX}" in text


def test_troubleshooting_lists_every_self_install_code() -> None:
    """``self install`` is the first command an AppImage user runs; its refusals are documented."""
    source = (ROOT / "src" / "yonga" / "application" / "self_install.py").read_text(
        encoding="utf-8"
    ) + (ROOT / "src" / "yonga" / "cli.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'"(self\.[a-z_]+(?:\.[a-z_]+)*)"', source))
    actions = set(re.findall(r'_io_failure\("([a-z_]+)"', source))
    codes |= {f"self.io.{action}" for action in actions}
    assert {"self.not_appimage", "self.io.write", "self.install.declined"} <= codes
    text = (DOCS / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    missing = sorted(code for code in codes if f"`{code}`" not in text)
    assert not missing
