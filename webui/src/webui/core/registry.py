"""Load + validate the box definitions (``boxes/*.yaml``) into a Registry."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml
from pydantic import ValidationError

from .schema import BoxDef


class RegistryError(RuntimeError):
    """A box definition file is invalid; ``detail`` carries the reason."""


class Registry:
    """Validated, unique-id set of :class:`BoxDef`."""

    def __init__(self, defs: list[BoxDef]):
        self.defs = defs

    def __iter__(self) -> Iterator[BoxDef]:
        return iter(self.defs)

    def __len__(self) -> int:
        return len(self.defs)

    def get(self, box_id: str) -> BoxDef:
        for d in self.defs:
            if d.id == box_id:
                return d
        known = ", ".join(d.id for d in self.defs) or "(none)"
        raise KeyError(f"unknown box definition {box_id!r} (known: {known})")

    def match(self, hint: Optional[str]) -> Optional[BoxDef]:
        """Best-effort match of a fleet-entry name/id against a def id or
        name (case-insensitive, also matches the box_key).  ``None`` when no
        def matches — the caller then reports what's available."""
        if not hint:
            return None
        h = hint.strip().lower()
        for d in self.defs:
            if h in (d.id.lower(), d.name.lower(), (d.box_key or "").lower()):
                return d
        return None

    def to_list(self) -> list[dict]:
        return [d.model_dump(mode="json") for d in self.defs]


def load_def(path: Path) -> BoxDef:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BoxDef.model_validate(raw)
    except (ValidationError, yaml.YAMLError, ValueError) as e:
        raise RegistryError(f"{path}: {e}") from e


def load_registry(boxes_dir: str | Path, only: Optional[Iterable[str]] = None) -> Registry:
    """Load every ``*.yaml`` in ``boxes_dir`` (only the given ids if set).

    Fails fast with :class:`RegistryError` on the first bad file — the box
    definitions are the contract, a typo must not ship.
    """
    d = Path(boxes_dir)
    if not d.is_dir():
        raise RegistryError(f"boxes dir not found: {d}")
    paths = sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))
    if not paths:
        raise RegistryError(f"no box definitions in {d}")
    defs: list[BoxDef] = []
    seen: set[str] = set()
    for p in paths:
        defn = load_def(p)
        if only is not None and defn.id not in set(only):
            continue
        if defn.id in seen:
            raise RegistryError(f"duplicate box id {defn.id!r} (in {p})")
        seen.add(defn.id)
        defs.append(defn)
    return Registry(defs)


__all__ = ["Registry", "RegistryError", "load_def", "load_registry"]
