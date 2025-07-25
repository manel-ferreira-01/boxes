# AcquisitionService Server - Usage Manual

This document describes the functionality and usage of the `AcquisitionService` gRPC server, which integrates with a Gradio web interface for image acquisition and display.

## 1. Introduction

The `AcquisitionService` server provides two primary functionalities:
* **Image Acquisition (`acquire`):** Allows a gRPC client to request an image and associated metadata (label) that has been previously made available to the server (e.g., uploaded via the Gradio UI).
* **Image Display (`display`):** Enables a gRPC client to send an image and metadata to the server for display on its integrated Gradio web interface.

The server leverages Protocol Buffers for efficient inter-process communication via gRPC, and Gradio for a user-friendly web-based visualization and input mechanism.

## 2. Server Architecture Overview

The server's architecture can be conceptualized as follows:

* **gRPC Server:** Listens for incoming gRPC requests (`acquire` and `display`).
* **Gradio Web Interface:** Provides a local web UI for users to:
    * Upload images, which are then placed into an internal queue for the `acquire` gRPC method to pick up.
    * View images and associated information sent from a gRPC client via the `display` method.
* **Internal Queues (`data_acq_queue`, `data_display_queue`):** These Python `queue.Queue` objects act as communication channels between the gRPC service implementation and the Gradio UI handlers.
    * `data_acq_queue`: Stores `(label_json_string, image_bytes)` tuples. Data is put here by the Gradio input handler and retrieved by the `acquire` gRPC method.
    * `data_display_queue`: Stores `(info_string, opencv_numpy_array)` tuples. Data is put here by the `display` gRPC method and retrieved by the Gradio output handler.

## 3. Getting Started (Deployment & Execution)

### 3.1. Prerequisites

Before running the server, ensure you have the following installed:

* Python 3.x
* `pip` (Python package installer)
* Required Python libraries:
    * `grpcio`
    * `grpcio-tools`
    * `Pillow` (PIL)
    * `gradio`
    * `numpy`
    * `opencv-python` (or `opencv-contrib-python`)

You can install these using pip:

```bash
pip install grpcio grpcio-tools Pillow gradio numpy opencv-python
```

### **3.2. Protocol Buffer Compilation**

The server uses a acquisition.proto file to define its service and message structures. You need to compile this .proto file into Python source code:

Bash

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. acquisition.proto

This command will generate two files in your current directory: acquisition_pb2.py and acquisition_pb2_grpc.py. These files are essential for both the server and any gRPC clients.

### **3.3. Running the Server**

To start the server, execute the Python script:

Bash

python server.py

Upon successful startup, you will see logging messages indicating:

* The gRPC server starting (default port: 8061).  
* The Gradio UI launching (default port: 7860).

You can access the Gradio web interface by navigating to the address provided in the console output (e.g., http://0.0.0.0:7860 or a public share link if share=True is enabled).  
Environment Variable for gRPC Port:  
You can optionally specify the gRPC server's port using the PORT environment variable:

Bash

PORT=8080 python server.py

## **4. Server Functionality & Interaction**

### **4.1. acquire Method (gRPC Client Initiated)**

The acquire gRPC method allows a client to retrieve an image and its label from the server's internal acquisition queue (data_acq_queue).  
**Workflow:**

1. **User uploads an image via the Gradio UI:** When a user uploads an image through the "Upload Image" component on the Gradio interface and optionally provides a "Session ID", the handle_input function is triggered.  
2. **Image processed and queued:** handle_input converts the uploaded PIL image to bytes, creates a JSON string label (including the provided session ID and other metadata), and places this (label_json, image_bytes) tuple into data_acq_queue.  
3. **gRPC Client Request:** A gRPC client sends an empty AcquireRequest to the server's acquire method.  
4. **Data Retrieval:** The acquire method attempts to get (label, image_bytes) from data_acq_queue.  
   * **Success:** If data is available, it's returned to the client in an AcquireResponse.  
   * **Failure:** If the queue is empty after a timeout, the server responds with a NOT_FOUND gRPC status code, indicating no data is currently available for acquisition.

**Usage Notes:**

* This method is designed to "pull" data that has been *made available* to the server, typically through the Gradio UI.  
* Clients should handle the NOT_FOUND status code gracefully.

### **4.2. display Method (gRPC Client Initiated)**

The display gRPC method allows a client to send an image and a label to the server for visual display on its Gradio web interface.  
**Workflow:**

1. **gRPC Client Request:** A gRPC client constructs a DisplayRequest message, populating its label field (expected to be a JSON string) and image field (raw image bytes, e.g., PNG or JPEG).  
2. **Server Receives and Processes:** The display method on the server receives the request.  
   * It decodes the incoming image bytes into an OpenCV (NumPy) image array.  
   * It uses the label string directly.  
3. **Data Queued for Display:** The processed (info_string, opencv_numpy_array) tuple is then placed into data_display_queue.  
4. **Gradio Displays Image:** The display_img function, which is linked to the output components of the Gradio UI, periodically checks data_display_queue. When new data arrives, it retrieves the info and img_np and updates the "Processed Image" and "Processing Info" displays on the web interface.  
5. **Empty Response:** The display method returns an empty DisplayResponse upon successful processing, indicating that the request was received and the data was queued for display.

**Usage Notes:**

* The label field is expected to be a JSON string, allowing for structured metadata to be sent alongside the image.  
* The server expects image bytes that can be decoded by OpenCV (cv2.imdecode).

### **4.3. Gradio Web Interface**

The Gradio interface provides a convenient way to interact with the server's queues.  
**Components:**

* **"Upload Image (sends to acquisition queue)":** An Image component where you can drag-and-drop or upload image files. This triggers the handle_input function, placing the image and an associated label into data_acq_queue.  
* **"Optional Session ID":** A Textbox to provide a custom string that will be included in the JSON label when an image is uploaded.  
* **"Processed Image (from display queue)":** An Image component that displays images sent by a gRPC client via the display method.  
* **"Processing Info (from display queue)":** A Textbox that displays the label string sent alongside the image by the gRPC client's display method.

**Interaction:**

* **To provide an image for the acquire gRPC method:** Upload an image in the "Upload Image" section of the Gradio UI.  
* **To view an image sent by the display gRPC method:** Send an image from your gRPC client. The image will appear in the "Processed Image" section, and its associated label in the "Processing Info" section.

## **5. Logging**

The server provides informative logging messages to the console, including:

* Server startup and shutdown events.  
* Status of gRPC calls (acquire and display).  
* Queue operations (putting/getting data).  
* Error messages with traceback for debugging.

Monitor the console output for insights into server operation and to diagnose any issues.

## **6. Error Handling**

The gRPC methods include basic error handling:

* acquire: Returns grpc.StatusCode.NOT_FOUND if the acquisition queue is empty.  
* display: Returns grpc.StatusCode.INVALID_ARGUMENT if image decoding fails or label is invalid JSON.  
* General grpc.StatusCode.INTERNAL is used for unexpected server-side errors.

Clients should be designed to handle these gRPC status codes gracefully.

## **7. Development & Debugging**

* **Python Logging:** Adjust the logging.basicConfig level (e.g., logging.DEBUG) at the beginning of server.py for more verbose output during development.  
* **grpcurl:** Since gRPC reflection is enabled, you can use tools like grpcurl to inspect the service and make calls from the command line, which is useful for quick tests:  
  Bash  
  grpcurl -plaintext localhost:8061 list  
  grpcurl -plaintext localhost:8061 describe AcquisitionService  
  # Example for acquire (will likely get NOT_FOUND unless Gradio input happened)  
  grpcurl -plaintext localhost:8061 AcquisitionService/acquire

* **Queue Monitoring:** During debugging, you could temporarily add print statements to check the size and content of data_acq_queue and data_display_queue to understand data flow.

**How to create the downloadable file:**

1.  **Copy the entire Markdown content** provided above.  
2.  **Open a plain text editor** (like Notepad on Windows, TextEdit on Mac, VS Code, Sublime Text, etc.).  
3.  **Paste the copied content** into the editor.  
4.  **Save the file** with a `.md` extension (e.g., `AcquisitionService_Usage_Manual.md`).

This `.md` file can then be opened by any Markdown viewer, text editor, or integrated development environment (IDE) that supports Markdown.  
