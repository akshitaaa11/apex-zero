"""Shared pytest configuration.

Unit tests must not touch the network (AGENTS.md §6.7): every test not marked ``network`` runs
with socket connections disabled, so an accidental live call fails loudly instead of passing.
"""

from __future__ import annotations

import socket

import pytest


class NetworkAccessError(RuntimeError):
    """Raised when a non-network test opens a connection."""


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network") is not None:
        return

    def guard(*args: object, **kwargs: object) -> None:
        raise NetworkAccessError(f"network access in a non-network test: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
