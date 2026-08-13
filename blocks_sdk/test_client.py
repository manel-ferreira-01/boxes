#!/usr/bin/env python3
"""Comprehensive test suite and usage demonstration for dynamic blocks_sdk."""
import os
import sys
import unittest
from concurrent import futures

# Add local path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocks_sdk import Client, YOLO, CoTracker, TextEmbedding, TAPNext, helpers


class TestBlocksSDK(unittest.TestCase):

    def setUp(self):
        os.environ.pop('SKIP_DISCOVERY', None)

    def test_client_reflection_fallback(self):
        """Verify client initializes gracefully when no reflection server is running."""
        os.environ['SKIP_DISCOVERY'] = '1'
        client = Client(address="localhost:9999", timeout=1.0)
        self.assertEqual(client.address, "localhost:9999")
        self.assertEqual(client.list_methods(), [])
        client.close()
        os.environ.pop('SKIP_DISCOVERY', None)

    def test_helpers_wrap_unwrap(self):
        """Verify wrapping and unwrapping of Python primitives to Protobuf Values."""
        val_bytes = helpers.wrap_value(b"test_data")
        self.assertEqual(helpers.unwrap_value(val_bytes), b"test_data")

        val_str = helpers.wrap_value("hello")
        self.assertEqual(helpers.unwrap_value(val_str), "hello")

        val_float = helpers.wrap_value(3.14)
        self.assertAlmostEqual(helpers.unwrap_value(val_float), 3.14, places=2)

        val_list_bytes = helpers.wrap_value([b"frame1", b"frame2"])
        self.assertEqual(helpers.unwrap_value(val_list_bytes), [b"frame1", b"frame2"])

        val_list_str = helpers.wrap_value(["a", "b"])
        self.assertEqual(helpers.unwrap_value(val_list_str), ["a", "b"])

    def test_mock_grpc_reflection(self):
        """Verify dynamic discovery and execution using a mock gRPC server with reflection."""
        os.environ.pop('SKIP_DISCOVERY', None)
        try:
            import grpc
            from grpc_reflection.v1alpha import reflection
            from blocks_sdk.protos import pipeline_pb2, pipeline_pb2_grpc

            class MockPipelineServicer(pipeline_pb2_grpc.PipelineServiceServicer):
                def Process(self, request, context):
                    return pipeline_pb2.Envelope(
                        config_json='{"status": "ok"}',
                        data=request.data
                    )

            server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
            pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(MockPipelineServicer(), server)
            
            SERVICE_NAMES = (
                pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(SERVICE_NAMES, server)
            port = server.add_insecure_port('127.0.0.1:0')
            server.start()

            address = f"127.0.0.1:{port}"
            client = Client(address=address, timeout=5.0)

            # Test 1: Dynamic discovery of methods
            methods = client.list_methods()
            self.assertIn("Process", methods)

            # Test 2: Dynamic attribute calling (client.Process(...))
            resp = client.Process(data={"img": b"fake_bytes"}, config_json='{"test": 1}')
            self.assertEqual(resp.config_json, '{"status": "ok"}')
            self.assertEqual(helpers.unwrap_value(resp.data["img"]), b"fake_bytes")

            client.close()
            server.stop(None)
        except ImportError:
            self.skipTest("grpc_reflection or grpc not installed in environment")


def main():
    print("==================================================")
    print("         blocks_sdk Dynamic Reflection Test       ")
    print("==================================================")
    
    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBlocksSDK)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n==================================================")
    print("          Dynamic Zero-Knowledge Demo             ")
    print("==================================================")
    print("""
Now any service can be connected to without knowing its methods in advance:

    from blocks_sdk import Client

    # Connect to ANY microservice box on demand
    client = Client("localhost:8062")

    # Discover what methods this box offers:
    methods = client.list_methods()
    print("Discovered RPC methods:", methods)

    # Inspect input/output schema:
    if methods:
        info = client.get_method_info(methods[0])
        print("Schema info:", info)

    # Invoke any discovered method on demand:
    response = client.call(
        method=methods[0] if methods else "Process",
        data={"images": [b"..."]},
        config_json='{"threshold": 0.5}'
    )
    """)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
