"""Fleet routes: which boxes exist where + reachability probes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, ConfigDict

from ..core import Fleet, Registry
from .errors import WebUIError


class FleetAdd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    addr: str
    def_id: Optional[str] = None
    note: Optional[str] = None


class FleetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = None
    addr: Optional[str] = None
    def_id: Optional[str] = None
    note: Optional[str] = None


def create_fleet_router(fleet: Fleet, registry: Registry) -> APIRouter:
    r = APIRouter(prefix="/api/fleet", tags=["fleet"])

    @r.get("")
    def list_fleet():
        return {"entries": [e.model_dump() for e in fleet.list()]}

    @r.post("", status_code=201)
    def add_fleet(body: FleetAdd):
        if body.def_id is not None:
            try:
                registry.get(body.def_id)
            except KeyError as e:
                raise WebUIError(400, {"def_id": str(e.args[0])}) from e
        entry = fleet.add(body.name, body.addr, def_id=body.def_id, note=body.note)
        return entry.model_dump()

    @r.get("/{entry_id}")
    def get_fleet(entry_id: str):
        return fleet.get(entry_id).model_dump()

    @r.patch("/{entry_id}")
    def update_fleet(entry_id: str, body: FleetUpdate):
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if body.def_id is not None:
            try:
                registry.get(body.def_id)
            except KeyError as e:
                raise WebUIError(400, {"def_id": str(e.args[0])}) from e
        try:
            return fleet.update(entry_id, **fields).model_dump()
        except ValueError as e:
            raise WebUIError(400, {"message": str(e)}) from e

    @r.delete("/{entry_id}", status_code=204)
    def remove_fleet(entry_id: str):
        fleet.remove(entry_id)
        return None

    @r.post("/{entry_id}/probe")
    def probe_fleet(entry_id: str, timeout: float = Query(5.0, ge=0.5, le=60)):
        return fleet.probe(entry_id, timeout=timeout).model_dump()

    return r


__all__ = ["create_fleet_router"]
