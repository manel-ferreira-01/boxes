from concurrent import futures
import grpc
import display_pb2
import display_pb2_grpc
from display_interface import GradioDisplay
import threading

class DisplayService(display_pb2_grpc.DisplayServiceServicer):
    def __init__(self, gradio_display: GradioDisplay):
        self.gradio_display = gradio_display

    def display(self, request, context):
        self.gradio_display.update(request.image, request.label)
        return display_pb2.DisplayResponse()

def serve(gradio_display: GradioDisplay, port: int = 8161):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    display_pb2_grpc.add_DisplayServiceServicer_to_server(DisplayService(gradio_display), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server running on port {port}")
    server.wait_for_termination()

def main():
    gradio_display = GradioDisplay()

    grpc_thread = threading.Thread(target=serve, args=(gradio_display,))
    grpc_thread.start()

    gradio_display.launch()

if __name__ == "__main__":
    main()
