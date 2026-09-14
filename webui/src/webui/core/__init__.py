"""webui.core — the box-agnostic core (schema, registry, caller, ...).

Only this package — and the YAML files in ``webui/boxes/`` — may name a box.
"""

from .schema import (
    BoxDef, InputField, ParamDef, ActionDef, CommandSpec, SessionDef,
    LayerDef, ResultDef,
    WIDGETS, VISUALIZERS, OVERLAY_LAYERS, VALUE_KINDS,
)
from .registry import Registry, RegistryError, load_def, load_registry
from .artifact import Artifact, ArtifactMissing, ArtifactStore
from .caller import (
    TOKEN_PREFIX, CallRequest, CallSpec, CallBuildError,
    build_call, execute, resolve_data,
)
from .serialize import serialize_result, sniff_mime
from .fleet import Fleet, FleetEntry, probe_box

__all__ = [
    "BoxDef", "InputField", "ParamDef", "ActionDef", "CommandSpec",
    "SessionDef", "LayerDef", "ResultDef",
    "WIDGETS", "VISUALIZERS", "OVERLAY_LAYERS", "VALUE_KINDS",
    "Registry", "RegistryError", "load_def", "load_registry",
    "Artifact", "ArtifactMissing", "ArtifactStore",
    "TOKEN_PREFIX", "CallRequest", "CallSpec", "CallBuildError",
    "build_call", "execute", "resolve_data",
    "serialize_result", "sniff_mime",
    "Fleet", "FleetEntry", "probe_box",
]
