"""Clean, package-agnostic gRPC client wrapper for AI box microservices using universal Envelopes."""
import logging
from typing import Optional, Dict, Any, Union
import grpc
from blocks_sdk.protos import pipeline_pb2
from blocks_sdk import helpers

logger = logging.getLogger("blocks_sdk")


class Client:
    """
    Package-agnostic gRPC client to invoke any AI image/service container box.
    Uses universal Envelope protobuf serialization (`send envelope, receive envelope`)
    without requiring pre-compiled stubs or complex gRPC reflection.
    """

    def __init__(
        self,
        address: str = "localhost:8061",
        timeout: float = 30.0,
        max_message_length: int = -1,
        default_service: str = "pipeline.PipelineService"
    ):
        """
        Initialize gRPC Client box wrapper.

        Args:
            address: Target box address (host:port)
            timeout: Request timeout in seconds
            max_message_length: Max message size (-1 for unlimited)
            default_service: Fully qualified gRPC service name
        """
        self.address = address
        self.timeout = timeout
        self.default_service = default_service

        channel_options = [
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length),
        ]

        self.channel = grpc.insecure_channel(address, options=channel_options)

    def call(
        self,
        method: str = "Process",
        data: Optional[Dict[str, Any]] = None,
        config: Optional[Union[Dict[str, Any], str]] = None,
        service_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send an Envelope to the target box method and return the unwrapped response.

        Args:
            method: Name of the RPC method to call (e.g. 'Process', 'DetectSequence', 'TrackSequence')
            data: Payload dictionary (images, frames, vectors, text)
            config: Config metadata dictionary or JSON string
            service_name: Custom gRPC service name override
            **kwargs: Extra keyword arguments automatically assigned to data or config

        Returns:
            Decoded response dictionary with unwrapped Python data structures.
        """
        if data is None:
            data = {}
        else:
            data = dict(data)

        if config is None:
            config = {}
        elif isinstance(config, dict):
            config = dict(config)

        # Parse kwargs into data or config
        for key, val in kwargs.items():
            if key in ("config_json", "config_meta"):
                if isinstance(val, dict) and isinstance(config, dict):
                    config.update(val)
                else:
                    config = val
            elif isinstance(config, dict) and (isinstance(val, (int, float, bool)) or key in ("threshold", "stream", "grid_size", "source")):
                config[key] = val
            else:
                data[key] = val

        svc_name = service_name or self.default_service
        full_rpc_path = f"/{svc_name}/{method}"

        # Construct universal Envelope request
        request_envelope = helpers.wrap_envelope(data=data, config=config)

        # Create unary gRPC caller for Envelope
        invoker = self.channel.unary_unary(
            full_rpc_path,
            request_serializer=pipeline_pb2.Envelope.SerializeToString,
            response_deserializer=pipeline_pb2.Envelope.FromString
        )

        try:
            response_envelope = invoker(request_envelope, timeout=self.timeout)
            return helpers.unwrap_envelope(response_envelope)
        except grpc.RpcError as e:
            logger.error(f"gRPC Error on {full_rpc_path} @ {self.address}: {e.code()} - {e.details()}")
            raise

    def list_methods(self) -> list:
        """Return common pipeline service methods supported by standard boxes."""
        return ["Process", "DetectSequence", "TrackSequence", "AllProcessing", "similarity_check", "Forward"]

    def __getattr__(self, name: str) -> Any:
        """
        Allow invoking RPC methods directly as client attributes.
        Example: client.DetectSequence(images=[img1, img2], threshold=0.5)
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def dynamic_rpc_caller(*args, **kwargs):
            # Handle positional data argument if provided
            data = kwargs.pop('data', {})
            if args:
                if isinstance(args[0], dict):
                    data.update(args[0])
                else:
                    data['images'] = args[0]
            config = kwargs.pop('config', None)
            return self.call(method=name, data=data, config=config, **kwargs)

        return dynamic_rpc_caller

    def close(self):
        """Close gRPC channel connection."""
        if hasattr(self, "channel") and self.channel:
            self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
