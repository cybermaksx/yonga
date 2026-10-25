"""Small host-name helpers shared by the commands that bind or print an address."""

from __future__ import annotations

import ipaddress

__all__ = ["is_loopback", "url_host"]


def is_loopback(host: str) -> bool:
    """True for ``localhost`` and any loopback IP address (brackets allowed)."""
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def url_host(host: str) -> str:
    """``host`` as it appears in a URL: an IPv6 literal gets its brackets."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host
