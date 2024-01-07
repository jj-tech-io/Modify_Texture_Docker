# FROM modify-texture-app-base

# # Copying model files from host to container
# COPY ./saved_models/316/encoder.h5 /app/saved_ml_models/SmallBatchSize/encoder_19_05.h5
# COPY ./saved_models/316/decoder.h5 /app/saved_ml_models/SmallBatchSize/decoder_19_05.h5

# # Start Xvfb in the background, start x11vnc, and then run your application
# CMD Xvfb :99 -screen 0 1024x768x16 & \
#     x11vnc -display :99 -bg -nopw -listen localhost -xkb & \
#     python3 /app/main.py
# Base image with TensorFlow and GPU support
FROM tensorflow/tensorflow:devel-gpu

# Set the working directory
WORKDIR /app

# Install system dependencies for GUI, Tkinter, and virtual framebuffer
RUN apt-get update && \
    apt-get install -y --no-install-recommends xvfb x11vnc python3-tk && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set up environment variables for display
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# Copy the application code into the container
COPY . /app/

# Expose the VNC port to access the application
EXPOSE 5900

# CMD to start Xvfb, X11VNC and the application
CMD Xvfb :99 -screen 0 1024x768x16 & \
    x11vnc -display :99 -bg -nopw -listen localhost -xkb & \
    python3 /app/main.py
    # xhost +local:docker

# Build and run commands
# docker build -t modify-texture .
# docker run --gpus all -it --rm -e DISPLAY=host.docker.internal:0.0 modify-texture
# -e DISPLAY=$DISPLAY
# docker run --gpus all -it --rm -e DISPLAY=$DISPLAY modify-texture
# -e DISPLAY=172.17.0.1:0.0 192.168.1.73:0.0
# docker run --gpus all -it --rm -e DISPLAY=192.168.1.73:0.0 modify-texture
# -e DISPLAY=unix$DISPLAY
# docker run --gpus all -it --rm -e DISPLAY=unix$DISPLAY modify-texture
# -e DISPLAY=%DISPLAY% --network="host"
# docker run --gpus all -it --rm -e DISPLAY=%DISPLAY% --network="host" modify-texture

# docker run --gpus all -it --rm \
#     -e DISPLAY=192.168.1.73:0.0 \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -p 5900:5900 \
#     modify-texture