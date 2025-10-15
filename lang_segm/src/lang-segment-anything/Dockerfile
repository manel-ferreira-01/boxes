FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    #openssh-client \
    #build-essential \
    git 
    #libglib2.0-0 \
    #libgl1 \
    #libgtk-3-0

COPY . /lang-segment-anything

# Install dependencies
WORKDIR /lang-segment-anything
RUN pip install -r requirements.txt

EXPOSE 8000

# Entry point
CMD ["python3", "app.py"]
