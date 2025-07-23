import gradio as gr
import threading
import queue
import time
from PIL import Image, ImageOps

# Queues
data_acq_queue = queue.Queue()
data_display_queue = queue.Queue()

# Dummy processing function
def doprocess(image: Image.Image):
    # Example processing: convert to grayscale and invert colors
    grayscale = ImageOps.grayscale(image)
    inverted = ImageOps.invert(grayscale)
    return inverted, "Processed: Grayscale + Inverted"

# Background image processing thread
def process_acq_image():
    while True:
        if not data_acq_queue.empty():
            img = data_acq_queue.get()
            processed_img, info = doprocess(img)
            data_display_queue.put((processed_img, info))
        time.sleep(0.1)  # Avoid busy-waiting

# Output handler
def display_img():
    if not data_display_queue.empty():
        img, info = data_display_queue.get()
        return img, info
    else:
        return None, "Waiting for image..."

# Input handler
def handle_input(image):
    if image is not None:
        data_acq_queue.put(image)
    return display_img()

# Launch interface
def acq_img():
    with gr.Blocks() as demo:
        with gr.Row():
            img_input = gr.Image(label="Upload Image", type="pil")
            img_output = gr.Image(label="Processed Image")
        info_output = gr.Textbox(label="Processing Info")

        img_input.input(fn=handle_input, inputs=img_input, outputs=[img_output])
    
    return demo

# Start the processing thread
threading.Thread(target=process_acq_image, daemon=True).start()

# Run the app
if __name__ == "__main__":
    app = acq_img()
    app.launch(share=True)
