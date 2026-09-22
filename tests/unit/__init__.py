"""Test package marker.

Present so mypy names these modules ``tests.unit.test_x`` and the
``[[tool.mypy.overrides]] module = ["tests.*"]`` relaxation in ``pyproject.toml``
actually applies. pytest imports them the same way either side of this file.
"""

from __future__ import annotations
