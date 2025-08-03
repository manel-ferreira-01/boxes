import gradio as gr
import queue
import time
import os
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
from concurrent import futures
from PIL import Image, ImageOps
import io
import json # Import json for handling JSON strings
import numpy as np # Import numpy
#import cv2 # Import cv2 (OpenCV)

import acquisition_pb2
import acquisition_pb2_grpc
import base64

# --- Configuration ---
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# --- Global Queues ---
# Use tuples (label_json_string, image_bytes) for consistency
data_acq_queue = queue.Queue(maxsize=13)
data_display_queue = queue.Queue(maxsize=13) # Stores [info (str), frame (np.array)] for Gradio output
lastimage=None
lastinfo="empty"
# --- Helper Functions ---

def image_to_bytes(img: Image.Image, format='PNG'):
    """Converts a PIL Image to a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def bytes_to_pil_image(image_bytes: bytes):
    """Converts bytes to a PIL Image."""
    return Image.open(io.BytesIO(image_bytes))

# --- gRPC Server Implementation ---
class AcquisitionServiceServicer(acquisition_pb2_grpc.AcquisitionServiceServicer):

    def __init__(self):
        """
        Initializes the AcquisitionServiceServicer.
        """
        logging.info("AcquisitionServiceServicer initialized.")
        #Launch the gradio app
        app = acq_img()
        # Launch Gradio app. prevent_thread_lock=True is important when running with other threads/services.
        logging.info("Launching Gradio UI...")
        app.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=True)
        self.gradio=app
     
    def acquire(self, request, context):
        """
        Implements the acquire RPC method.
        Pulls a label (JSON string) and image bytes from data_acq_queue.
        """
        try:
            # Data from queue is expected to be (label_str, image_bytes)
            label, image_bytes = data_acq_queue.get() # Add timeout for robustness
            logging.info(f"Acquire: Retrieved data from queue. Label length: {len(label)}, Image bytes: {len(image_bytes)}")
            return acquisition_pb2.AcquireResponse(label=label, image=image_bytes)
        
        except Exception as e:
            logging.exception(f"Acquire: An unexpected error occurred: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error during acquisition: {e}')
            return acquisition_pb2.AcquireResponse()

    def display(self, request, context):
        """
        Implements the display RPC method.
        Receives label (JSON string) and image bytes, processes them,
        and puts them into data_display_queue for Gradio to pick up.
        """
        try:
            # Convert bytes to OpenCV image (numpy array)
            #np_array = np.frombuffer(request.image, np.uint8)
            #frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            frame=Image.open(io.BytesIO(request.image))
            frame= np.asarray(frame)
            if frame is None:
                logging.error("Display: Failed to decode image bytes into an OpenCV frame.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Failed to decode image bytes.')
                return acquisition_pb2.DisplayResponse()

            info = request.label # label is already a string (hopefully JSON)

            logging.info(f"Display: Received image (shape: {frame.shape}) with label: {info}. Going to put in queue")

            data_display_queue.put([info, frame])
            return acquisition_pb2.DisplayResponse()
        
        except json.JSONDecodeError as e:
            logging.error(f"Display: Error decoding label JSON: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f'Invalid JSON format for label: {e}')
            return acquisition_pb2.DisplayResponse()
        
        except Exception as e:
            logging.exception(f"Display: An unexpected error occurred: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Internal server error during display: {e}')
            return acquisition_pb2.DisplayResponse()

# --- Gradio Interface Logic ---

# Input handler for Gradio: converts PIL image to bytes and puts into acquisition queue
def handle_input(image: Image.Image, session_id: str):
    """
    Handles input from Gradio UI. Converts PIL image to bytes and puts it
    into data_acq_queue along with a generated label.
    """
    if image is not None:
        try:
            # Create a simple JSON label for demonstration
            label_data = {
                "source": "gradio_input",
                "session_id": session_id if session_id else f"gradio_session_{int(time.time())}",
                "timestamp": time.time(),
                "image_format": "PNG"
            }
            label_json = json.dumps(label_data)
            
            image_bytes = image_to_bytes(image, format='PNG')
            data_acq_queue.put((label_json, image_bytes)) # Put tuple (label, image_bytes)
            logging.info(f"HANDLE_INPUT: Put image (bytes: {len(image_bytes)}) with label to acquisition queue.")
            
            # Immediately try to display something from the display queue if available
            return 
        except Exception as e:
            logging.exception(f"Gradio handle_input: Error processing image: {e}")
            return 
    else:
        logging.info("HANDLE_INPUT: No image provided.")
        #img = Image.new('RGB', (200, 200), (100,100,100))
        #image_bytes=image_to_bytes(img)
        #data_acq_queue.put(("merda", image_bytes)) # Put tuple (label, image_bytes)
        #return display_img() # Still attempt to display existing data
        return

# Output handler for Gradio: displays image from display queue
def display_img(state):
    """
    Retrieves image and info from data_display_queue for Gradio output.
    """
    global lastimage, lastinfo

    if not data_display_queue.empty():
#    if True : #fila com suspensão
        try:
            info, img_np = data_display_queue.get(timeout=0.1) # img_np is expected to be a numpy array from cv2.imdecode
            # Gradio gr.Image(type="numpy") expects a numpy array
            lastimage=img_np
            lastinfo=info
            return img_np, info,state+1
        except queue.Empty:
            # Should not happen with empty check, but for safety
            #logging.warning("Gradio display_img: Queue became empty before getting data.")
            return lastimage, lastinfo ,state
        except Exception as e:
            logging.exception(f"Gradio display_img: Error getting data from display queue: {e}")
            return None, f"Error displaying image: {e}",state
    else:
        #logging.info("Gradio display_img: Display queue is empty. Waiting for image...")
        return lastimage, lastinfo,state

# Gradio Interface Definition -------------------------------------------------------
def acq_img():
    """Defines and returns the Gradio Blocks interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Image Acquisition & Display Service")
        display_trigger_state = gr.State(value=0) 
        timer = gr.Timer(0.6)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input to Acquisition Queue (for `acquire` method)")
                img_input = gr.Image(label="Upload Image (sends to acquisition queue)", type="pil")
                session_id_input = gr.Textbox(label="Input labels/commands (TBD)", placeholder="write something")

            with gr.Column():
                gr.Markdown("### Output from Display Queue (from `display` method)")
                img_output = gr.Image(label="Processed Image (from display queue)", type="numpy") # Expect numpy from cv2
                info_output = gr.Textbox(label="Processing Info (from display queue)")

        # Link input to handler
        # Using .change instead of .input for better reactivity if image is dropped/changed
        img_input.change(fn=handle_input, inputs=[img_input, session_id_input])
        #session_id_input.change(fn=handle_input, inputs=[img_input, session_id_input], outputs=[img_output, info_output], show_progress=False)

        # To continuously poll for new images to display, use gr.Live
        # Note: gr.Live / gr.Blocks with .load and queue=True can be complex with gRPC.
        # For simplicity, we'll rely on the input trigger or manual refresh.
        # A more advanced setup might use websockets or server-sent events for continuous updates.
        timer.tick(
            fn=display_img,
            inputs=[display_trigger_state], # Pass the state variable as input
            outputs=[img_output, info_output, display_trigger_state]) # Update output components and the state Poll every 1 second       
    return demo

# --- Server Utilities ---

def get_port():
    """
    Parses the port where the server should listen.
    Exits the program if the environment variable is not an int or the value is not positive.
    Returns: The port where the server should listen or None if an error occurred.
    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
        return server_port
    except ValueError:
        logging.exception('Unable to parse port from environment variable.')
        return None

def run_server(server):
    """
    Runs the gRPC server and launches the Gradio interface.
    """
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'gRPC Server started at {target}')

    try:
        
        # Keep the main thread alive indefinitely for the gRPC server
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logging.info("Shutting down gRPC server and Gradio UI.")
        server.stop(0)
    except Exception as e:
        logging.exception(f"An unexpected error occurred during server execution: {e}")
    finally:
        logging.info("Server application terminated.")

# --- Main Execution Block ---
if __name__ == '__main__':
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    
    # Ensure protobuf generated files are available
    try:
        import acquisition_pb2
        import acquisition_pb2_grpc
    except ImportError:
        logging.error("Could not import protobuf generated files. Please run 'python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. acquisition.proto'")
        exit(1)

    # Create gRPC Server and add service
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 512 * 1024 * 1024), 
                                  ('grpc.max_receive_message_length', 512 * 1024 * 1024)])

    acquisition_pb2_grpc.add_AcquisitionServiceServicer_to_server(
        AcquisitionServiceServicer(), server)
    
    # Add reflection for gRPCurl and other tools
    service_names = (
        acquisition_pb2.DESCRIPTOR.services_by_name['AcquisitionService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)