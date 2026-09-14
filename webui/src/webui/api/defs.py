"""Box-definition routes (the contract the SPA renders from)."""

from __future__ import annotations

from fastapi import APIRouter

from ..core import Registry
from ..core.schema import WIDGETS, VISUALIZERS, OVERLAY_LAYERS, VALUE_KINDS
from .errors import WebUIError


def create_defs_router(registry: Registry) -> APIRouter:
    r = APIRouter(prefix="/api/defs", tags=["defs"])

    @r.get("")
    def list_defs():
        return {
            "defs": registry.to_list(),
            "vocabulary": {
                "widgets": sorted(WIDGETS),
                "visualizers": sorted(VISUALIZERS),
                "overlay_layers": sorted(OVERLAY_LAYERS),
                "value_kinds": list(VALUE_KINDS),
            },
        }

    @r.get("/{box_id}")
    def get_def(box_id: str):
        try:
            defn = registry.get(box_id)
        except KeyError as e:
            raise WebUIError(404, {"message": str(e.args[0])}) from e
        return defn.model_dump(mode="json")

    return r


__all__ = ["create_defs_router"]
