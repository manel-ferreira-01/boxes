"""gRPC client wrapper with dynamic, zero-knowledge reflection service discovery."""
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
import grpc
from google.protobuf import descriptor_pool, message_factory, descriptor_pb2

logger = logging.getLogger("blocks_sdk")


class Client:
    """
    Dynamic gRPC client capable of calling any AI service endpoint on-the-fly.
    Uses gRPC Server Reflection to discover services, RPC methods, and message types
    without requiring pre-compiled stubs or prior knowledge of the target service.
    """

    # Class-level cache: address -> {
    #    "services": {service_name: [method_names]},
    #    "methods": {method_name: (service_full_name, input_class, output_class)},
    # }
    _reflection_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        address: str = "localhost:8061",
        timeout: float = 30.0,
        max_message_length: int = -1
    ):
        """
        Initialize gRPC client.
        
        Args:
            address: Service address (host:port or hostname:port)
            timeout: Request timeout in seconds
            max_message_length: Max message size (-1 for unlimited)
        """
        self.address = address
        self.timeout = timeout
        
        channel_options = [
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length),
        ]
        
        self.channel = grpc.insecure_channel(address, options=channel_options)
        self.stub = None  # Backward compatibility stub
        
        # Internal storage for reflection-discovered metadata
        self._discovered_services: Dict[str, List[str]] = {}
        self._discovered_methods: Dict[str, Tuple[str, Any, Any]] = {}
        
        # Discover methods on initialization if not disabled
        if not os.getenv('SKIP_DISCOVERY'):
            self.refresh_reflection()

    def refresh_reflection(self) -> bool:
        """Query gRPC server reflection to discover all available services and methods."""
        if os.getenv('SKIP_DISCOVERY'):
            return False

        try:
            from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc
            
            ref_stub = reflection_pb2_grpc.ServerReflectionStub(self.channel)
            
            # List all services
            request = reflection_pb2.ServerReflectionRequest(list_services="")
            responses = ref_stub.ServerReflectionInfo(iter([request]))
            
            service_names = []
            for resp in responses:
                if resp.HasField('list_services_response'):
                    for svc in resp.list_services_response.service:
                        name = svc.name
                        # Exclude internal reflection service
                        if not name.startswith("grpc.reflection"):
                            service_names.append(name)
            
            if not service_names:
                return False

            pool = descriptor_pool.DescriptorPool()
            all_fds = []
            seen_names = set()

            # Fetch file descriptor protos for all services
            for svc_name in service_names:
                req = reflection_pb2.ServerReflectionRequest(file_containing_symbol=svc_name)
                resps = ref_stub.ServerReflectionInfo(iter([req]))
                for resp in resps:
                    if resp.HasField('file_descriptor_response'):
                        for fd_bytes in resp.file_descriptor_response.file_descriptor_proto:
                            fd = descriptor_pb2.FileDescriptorProto()
                            fd.ParseFromString(fd_bytes)
                            if fd.name not in seen_names:
                                all_fds.append(fd)
                                seen_names.add(fd.name)

            # Load file descriptors into pool handling dependency resolution order
            loaded_files = set()
            unloaded = list(all_fds)
            while unloaded:
                progress = False
                next_unloaded = []
                for fd in unloaded:
                    if fd.name in loaded_files:
                        continue
                    try:
                        pool.Add(fd)
                        loaded_files.add(fd.name)
                        progress = True
                    except Exception:
                        next_unloaded.append(fd)
                if not progress:
                    break
                unloaded = next_unloaded

            factory = message_factory.MessageFactory(pool)
            
            discovered_services = {}
            discovered_methods = {}

            for svc_name in service_names:
                svc_desc = None
                try:
                    svc_desc = pool.FindServiceByName(svc_name)
                except KeyError:
                    try:
                        svc_desc = descriptor_pool.Default().FindServiceByName(svc_name)
                    except KeyError:
                        pass
                
                if svc_desc is None:
                    continue

                method_list = []
                for method in svc_desc.methods:
                    method_list.append(method.name)
                    try:
                        input_cls = factory.GetPrototype(method.input_type)
                        output_cls = factory.GetPrototype(method.output_type)
                    except Exception:
                        input_cls = message_factory.GetMessageClass(method.input_type)
                        output_cls = message_factory.GetMessageClass(method.output_type)
                    discovered_methods[method.name] = (svc_name, input_cls, output_cls)
                discovered_services[svc_name] = method_list

            self._discovered_services = discovered_services
            self._discovered_methods = discovered_methods
            
            Client._reflection_cache[self.address] = {
                "services": discovered_services,
                "methods": discovered_methods
            }
            return len(discovered_methods) > 0
        except Exception as e:
            # Silent fallback if service is offline or doesn't support reflection
            return False

    def list_methods(self) -> List[str]:
        """Return list of all discovered method names available on this service."""
        return list(self._discovered_methods.keys())

    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """Get schema details for a discovered RPC method."""
        if method_name in self._discovered_methods:
            svc_name, input_cls, output_cls = self._discovered_methods[method_name]
            return {
                "service": svc_name,
                "input_type": input_cls.DESCRIPTOR.full_name,
                "output_type": output_cls.DESCRIPTOR.full_name,
                "input_fields": [f.name for f in input_cls.DESCRIPTOR.fields],
                "output_fields": [f.name for f in output_cls.DESCRIPTOR.fields]
            }
        return None

    def call(
        self,
        method: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        config_json: str = '{"source": "blocks_sdk"}',
        **kwargs
    ) -> Any:
        """
        Make a gRPC call using dynamic reflection or fallback stubs.
        
        Args:
            method: RPC method name to call (if None, selects first available method)
            data: Dictionary of data to send
            config_json: JSON config string (for Envelope-based services)
            **kwargs: Direct field values to populate request message
            
        Returns:
            Decoded gRPC response object
        """
        if data is None:
            data = {}
            
        # Combine kwargs into data if provided
        combined_data = {**data, **kwargs}

        # If method not specified, pick first discovered or default method
        if method is None:
            if self._discovered_methods:
                method = next(iter(self._discovered_methods.keys()))
            else:
                method = "Process"

        # 1. Try Dynamic Invocation via Reflection
        if method in self._discovered_methods:
            svc_name, input_cls, output_cls = self._discovered_methods[method]
            request_msg = self._build_request_message(input_cls, combined_data, config_json)
            
            full_rpc_path = f"/{svc_name}/{method}"
            invoker = self.channel.unary_unary(
                full_rpc_path,
                request_serializer=input_cls.SerializeToString,
                response_deserializer=output_cls.FromString
            )
            try:
                return invoker(request_msg, timeout=self.timeout)
            except grpc.RpcError as e:
                logger.error(f"gRPC Error on {full_rpc_path}: {e.code()} - {e.details()}")
                raise

        # 2. Fallback to pre-compiled stub if available
        if self.stub is not None:
            rpc_method = getattr(self.stub, method, None)
            if rpc_method is not None:
                from blocks_sdk import helpers
                from blocks_sdk.protos import pipeline_pb2
                
                wrapped_data = {k: helpers.wrap_value(v) for k, v in combined_data.items()}
                req = pipeline_pb2.Envelope(
                    config_json=config_json,
                    data=wrapped_data
                )
                return rpc_method(req, timeout=self.timeout)

        # 3. Fallback to standard pipeline.proto Envelope if method is known (e.g. Process)
        try:
            from blocks_sdk import helpers
            from blocks_sdk.protos import pipeline_pb2, pipeline_pb2_grpc
            
            stub = pipeline_pb2_grpc.PipelineServiceStub(self.channel)
            rpc_method = getattr(stub, method, None)
            if rpc_method is not None:
                wrapped_data = {k: helpers.wrap_value(v) for k, v in combined_data.items()}
                req = pipeline_pb2.Envelope(
                    config_json=config_json,
                    data=wrapped_data
                )
                return rpc_method(req, timeout=self.timeout)
        except Exception:
            pass

        raise ValueError(
            f"Method '{method}' not found on service at {self.address}. "
            f"Available discovered methods: {self.list_methods()}"
        )

    def _build_request_message(self, input_cls: Any, data: Dict[str, Any], config_json: str) -> Any:
        """Intelligently construct and populate a protobuf request message from input data."""
        field_names = set(f.name for f in input_cls.DESCRIPTOR.fields)
        
        # Check if message is Envelope (has config_json and data fields)
        if "config_json" in field_names and "data" in field_names:
            from blocks_sdk import helpers
            wrapped_data = {
                k: helpers.wrap_value(v) for k, v in data.items()
            }
            return input_cls(config_json=config_json, data=wrapped_data)
        
        # Otherwise populate native fields on the message
        kwargs = {}
        for field in input_cls.DESCRIPTOR.fields:
            if field.name in data:
                val = data[field.name]
                kwargs[field.name] = val
        return input_cls(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Allow invoking discovered RPC methods directly as client methods (e.g. client.DetectSequence(...))."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        if name in self._discovered_methods:
            def dynamic_rpc_caller(**kwargs):
                data = kwargs.pop('data', {})
                config_json = kwargs.pop('config_json', '{"source": "blocks_sdk"}')
                return self.call(method=name, data=data, config_json=config_json, **kwargs)
            return dynamic_rpc_caller

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
