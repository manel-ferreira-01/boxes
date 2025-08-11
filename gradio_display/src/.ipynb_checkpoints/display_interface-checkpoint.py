import gradio as gr
import numpy as np
import cv2
from threading import Lock

class GradioDisplay:
    def __init__(self):
        self.image = None
        self.label = ""
        self.lock = Lock()
        self.interface = self._create_interface()

    def _create_interface(self):

        with gr.Blocks() as demo:
            gr.Markdown("## gRPC Image Display")
            with gr.Column():
                self.image_input = gr.Image(label="Input Image")
                self.label_input = gr.Textbox(label="Input,Labels")
                refresh_btn = gr.Button("🔄 Run ", elem_id="refresh-btn")
            with gr.Column():
                self.image_output = gr.Image(label="Received Image", interactive=False)
                self.label_output = gr.Textbox(label="Label", interactive=False)
#   Metodos
            refresh_btn.click(
                fn=self._update_display,
                inputs=[self.image_output,self.label_input],
                outputs=[self.image_output, self.label_output]
            )

        return demo

    def _update_display(self):
        while True:
            if os.path.exists(path):
                return True
            else:
                time.sleep(1)
        with self.lock:
            return [self.image, self.label]

    def update(self, image_bytes: bytes, label: str):
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        with self.lock:
            self.image = image_rgb
            self.label = label

    def launch(self):
        self.interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
        