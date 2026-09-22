"""Placeholder command-line entry point.

This module only exists so that the ``yonga`` console script and
``python -m yonga`` resolve while the real CLI is being written. It implements
``--version`` and nothing else; the CLI agent replaces this file wholesale.
"""

from __future__ import annotations

import typer

from yonga import APP_NAME, __version__

app = typer.Typer(
    name=APP_NAME,
    help="Local AI model manager for Intel AI PCs (placeholder CLI).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` was passed."""
    if value:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Placeholder root command; real subcommands land with the CLI agent."""


__all__ = ["app", "main"]
