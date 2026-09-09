"""blocks_sdk: Lightweight, package-agnostic Python SDK for AI image & vision microservice boxes."""
from blocks_sdk.client import Client
from blocks_sdk.helpers import to_bytes, wrap_value, unwrap_value, wrap_envelope, unwrap_envelope
from blocks_sdk.services.yolo import YOLO
from blocks_sdk.services.tapnext import TAPNext
from blocks_sdk.services.cotracker import CoTracker
from blocks_sdk.services.text_embedding import TextEmbedding
from blocks_sdk.services.vggt import VGGT

__version__ = "0.2.0"

__all__ = [
    "Client",
    "YOLO",
    "TAPNext",
    "CoTracker",
    "TextEmbedding",
    "VGGT",
    "to_bytes",
    "wrap_value",
    "unwrap_value",
    "wrap_envelope",
    "unwrap_envelope",
]
