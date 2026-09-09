#!/usr/bin/env python3
"""Comprehensive unit test suite for package-agnostic blocks_sdk."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from concurrent import futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocks_sdk import Client, YOLO, CoTracker, TextEmbedding, TAPNext, VGGT, helpers


class TestBlocksSDK(unittest.TestCase):

    def test_image_to_bytes_conversion(self):
        """Test package-agnostic to_bytes conversion for paths, bytes, PIL, duck-typing."""
        # 1. Raw bytes
        raw = b"fake_image_bytes"
        self.assertEqual(helpers.to_bytes(raw), raw)

        # 2. File path (Path and str)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"file_bytes_data")
            tmp_path = tmp.name

        try:
            self.assertEqual(helpers.to_bytes(tmp_path), b"file_bytes_data")
            self.assertEqual(helpers.to_bytes(Path(tmp_path)), b"file_bytes_data")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 3. Duck-typed PIL Image mock
        class MockPILImage:
            def save(self, buf, format="JPEG"):
                buf.write(b"pil_encoded_image")

        pil_mock = MockPILImage()
        self.assertEqual(helpers.to_bytes(pil_mock), b"pil_encoded_image")

    def test_helpers_wrap_unwrap_envelope(self):
        """Verify wrapping/unwrapping of values and universal Envelopes."""
        # Scalar wrapping
        val_str = helpers.wrap_value("hello")
        self.assertEqual(helpers.unwrap_value(val_str), "hello")

        val_float = helpers.wrap_value(3.14)
        self.assertAlmostEqual(helpers.unwrap_value(val_float), 3.14, places=2)

        # List wrapping
        val_list_bytes = helpers.wrap_value([b"frame1", b"frame2"])
        self.assertEqual(helpers.unwrap_value(val_list_bytes), [b"frame1", b"frame2"])

        # Envelope wrap & unwrap
        envelope = helpers.wrap_envelope(
            data={"images": [b"img1", b"img2"], "threshold": 0.8},
            config={"model": "yolo"}
        )

        unwrapped = helpers.unwrap_envelope(envelope)
        self.assertEqual(unwrapped["images"], [b"img1", b"img2"])
        self.assertAlmostEqual(unwrapped["threshold"], 0.8, places=2)
        self.assertEqual(unwrapped["_config"], {"model": "yolo"})

    def test_mock_grpc_envelope_communication(self):
        """Verify sending and receiving Envelopes over gRPC."""
        try:
            import grpc
            from blocks_sdk.protos import pipeline_pb2, pipeline_pb2_grpc

            class MockPipelineServicer(pipeline_pb2_grpc.PipelineServiceServicer):
                def DetectSequence(self, request, context):
                    # Echo data back with result status
                    return pipeline_pb2.Envelope(
                        config_json='{"status": "detected", "count": 2}',
                        data=request.data
                    )

            server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
            pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(MockPipelineServicer(), server)
            port = server.add_insecure_port('127.0.0.1:0')
            server.start()

            address = f"127.0.0.1:{port}"
            client = Client(address=address, timeout=5.0)

            # Test 1: Envelope call using dynamic method attribute
            resp = client.DetectSequence(images=[b"frame1", b"frame2"], threshold=0.5)

            # Unwrapped dictionary assertions
            self.assertEqual(resp["images"], [b"frame1", b"frame2"])
            self.assertEqual(resp["_config"], {"status": "detected", "count": 2})

            # Test 2: Standard call method
            resp_generic = client.call("DetectSequence", data={"images": [b"test"]}, threshold=0.9)
            self.assertEqual(resp_generic["images"], [b"test"])

            client.close()
            server.stop(None)
        except ImportError:
            self.skipTest("grpc or dependencies not installed")

    def test_service_wrappers(self):
        """Verify high-level service wrappers initialize and generate expected envelope payloads."""
        yolo = YOLO("localhost:9999")
        self.assertEqual(yolo.address, "localhost:9999")

        cotracker = CoTracker("localhost:9998")
        self.assertEqual(cotracker.address, "localhost:9998")

        vggt = VGGT("localhost:9997")
        self.assertEqual(vggt.address, "localhost:9997")


def main():
    print("==================================================")
    print("      blocks_sdk Package-Agnostic Test Suite      ")
    print("==================================================")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBlocksSDK)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
