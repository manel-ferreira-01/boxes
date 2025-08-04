import gradio as gr
import queue
import time
import os
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
from concurrent import futures
import threading
from PIL import Image
import io
import json
import numpy as np

import acquisition_pb2
import acquisition_pb2_grpc

# --- Configuration ---
ACQUISITION_PORT = int(os.getenv('PORT', 8061))
DISPLAY_PORT = 8161
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# --- Global Queues ---
data_acq_queue = queue.Queue(maxsize=13)
data_display_queue = queue.Queue(maxsize=13)
lastimage = None
lastinfo = "empty"

# --- Helper Functions ---
def image_to_bytes(img: Image.Image, format='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

# --- gRPC Service Implementations ---
class AcquisitionServiceServicer(acquisition_pb2_grpc.AcquisitionServiceServicer):
    def __init__(self):
        logging.info("AcquisitionService initialized.")
        app = acq_img()
        logging.info("Launching Gradio UI...")
        app.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=True)
        self.gradio = app

    def acquire(self, request, context):
        try:
            label, image_bytes = data_acq_queue.get()
            logging.info(f"GRPC Acquire: Got label length {len(label)}, image bytes {len(image_bytes)}")
            return acquisition_pb2.AcquireResponse(label=label, image=image_bytes)
        except Exception as e:
            logging.exception(f"Acquire error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal error: {e}')
            return acquisition_pb2.AcquireResponse()

class DisplayServiceServicer(acquisition_pb2_grpc.DisplayServiceServicer):
    def display(self, request, context):
        try:
            if not request.image:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('No image provided.')
                return acquisition_pb2.DisplayResponse()
            frame = np.asarray(Image.open(io.BytesIO(request.image)))
            data_display_queue.put([request.label, frame])
            logging.info("Display: Image added to display queue.")
            return acquisition_pb2.DisplayResponse()
        except json.JSONDecodeError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f'Invalid JSON: {e}')
            return acquisition_pb2.DisplayResponse()
        except Exception as e:
            logging.exception(f"Display error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal error: {e}')
            return acquisition_pb2.DisplayResponse()

# --- Gradio Logic ---
def handle_input(image: Image.Image, session_id: str):
    if image:
        try:
            label_data = {
                "source": "gradio_input",
                "session_id": session_id or f"gradio_session_{int(time.time())}",
                "timestamp": time.time(),
                "image_format": "PNG"
            }
            label_json = json.dumps(label_data)
            image_bytes = image_to_bytes(image)
            data_acq_queue.put((label_json, image_bytes))
            logging.info(f"Gradio input: Queued image.")
        except Exception as e:
            logging.exception(f"Gradio input error: {e}")
    else:
        logging.info("Gradio input: No image.")

def display_img(state):
    global lastimage, lastinfo
    try:
        info, img_np = data_display_queue.get(timeout=0.1)
        lastimage, lastinfo = img_np, info
        return img_np, info, state + 1
    except queue.Empty:
        return lastimage, lastinfo, state
    except Exception as e:
        logging.exception(f"Gradio display error: {e}")
        return None, f"Error: {e}", state

def acq_img():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Acquisition & Display Service")
        state = gr.State(value=0)
        timer = gr.Timer(0.6)
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload Image", type="pil")
                session_id_input = gr.Textbox(label="Session ID")
            with gr.Column():
                img_output = gr.Image(label="Processed Image", type="numpy")
                info_output = gr.Textbox(label="Processing Info")

        img_input.change(fn=handle_input, inputs=[img_input, session_id_input])
        timer.tick(fn=display_img, inputs=[state], outputs=[img_output, info_output, state])
    return demo

# --- gRPC Server Launchers ---
def start_acquisition_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 512 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
    acquisition_pb2_grpc.add_AcquisitionServiceServicer_to_server(
        AcquisitionServiceServicer(), server)
    grpc_reflection.enable_server_reflection([
        acquisition_pb2.DESCRIPTOR.services_by_name['AcquisitionService'].full_name,
        grpc_reflection.SERVICE_NAME
    ], server)
    server.add_insecure_port(f'[::]:{ACQUISITION_PORT}')
    server.start()
    logging.info(f"AcquisitionService running on port {ACQUISITION_PORT}")
    server.wait_for_termination()

def start_display_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 512 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
    acquisition_pb2_grpc.add_DisplayServiceServicer_to_server(DisplayServiceServicer(), server)
    grpc_reflection.enable_server_reflection([
        acquisition_pb2.DESCRIPTOR.services_by_name['DisplayService'].full_name,
        grpc_reflection.SERVICE_NAME
    ], server)
    server.add_insecure_port(f'[::]:{DISPLAY_PORT}')
    server.start()
    logging.info(f"DisplayService running on port {DISPLAY_PORT}")
    server.wait_for_termination()

# --- Main ---
if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s (%(module)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    t1 = threading.Thread(target=start_acquisition_server, daemon=True)
    t2 = threading.Thread(target=start_display_server, daemon=True)

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logging.info("Shutting down both services.")
