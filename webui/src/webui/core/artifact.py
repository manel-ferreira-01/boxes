"""In-memory artifact store: uploaded files + heavy result payloads.

Tokens are capability strings (like tapnext's ``session_id``): anyone who
knows the token can fetch the bytes.  The store is deliberately simple —
TTL eviction + a byte cap with FIFO eviction — because this is a trusted
office tool, not a public multi-tenant service.  (Documented in README.)
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


class ArtifactMissing(KeyError):
    pass


@dataclass
class Artifact:
    token: str
    data: bytes
    content_type: str = "application/octet-stream"
    extra: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class ArtifactStore:
    def __init__(self, ttl: float = 3600.0, max_bytes: int = 2_000_000_000):
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._d: dict[str, Artifact] = {}
        self._total = 0
        self._lock = threading.Lock()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def new_token() -> str:
        return "upl_" + secrets.token_urlsafe(10)

    @staticmethod
    def url(token: str) -> str:
        return f"/api/file/{token}"

    # ------------------------------------------------------------ interface
    def add(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        extra: Optional[dict[str, Any]] = None,
        token: Optional[str] = None,
    ) -> str:
        token = token or self.new_token()
        with self._lock:
            self._expire_locked()
            room = self.max_bytes - self._total
            if len(data) > room:
                # FIFO-evict oldest artifacts until it fits (never block a call).
                for old in sorted(self._d, key=lambda t: self._d[t].created):
                    self._remove_locked(old)
                    if len(data) <= self.max_bytes - self._total:
                        break
            else:
                # opportunistically make room if near the cap
                pass
            art = Artifact(token=token, data=data, content_type=content_type,
                           extra=extra or {})
            self._d[token] = art
            self._total += len(data)
            return token

    def get(self, token: str) -> Artifact:
        with self._lock:
            self._expire_locked()
            art = self._d.get(token)
        if art is None:
            raise ArtifactMissing(token)
        return art

    def tokens(self) -> list[str]:
        with self._lock:
            self._expire_locked()
            return list(self._d)

    def public_view(self, token: str) -> dict:
        art = self.get(token)
        return {
            "token": token,
            "ref": "@" + token,
            "url": self.url(token),
            "content_type": art.content_type,
            "size": len(art.data),
            "extra": {k: v for k, v in art.extra.items()
                      if k in ("dtype", "shape", "pytype", "note", "filename")},
        }

    def stats(self) -> dict:
        with self._lock:
            return {"count": len(self._d), "bytes": self._total}

    # ------------------------------------------------------------------ misc
    def _expire_locked(self) -> None:
        now = time.time()
        for tok in [t for t, a in self._d.items() if now - a.created > self.ttl]:
            self._remove_locked(tok)

    def _remove_locked(self, token: str) -> None:
        art = self._d.pop(token, None)
        if art is not None:
            self._total -= len(art.data)


__all__ = ["Artifact", "ArtifactMissing", "ArtifactStore"]
