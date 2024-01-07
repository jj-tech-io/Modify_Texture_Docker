FROM modify-texture-app-base

# Copying model files from host to container
COPY ./saved_models/316/encoder.h5 /app/saved_ml_models/SmallBatchSize/encoder_19_05.h5
COPY ./saved_models/316/decoder.h5 /app/saved_ml_models/SmallBatchSize/decoder_19_05.h5

# Start Xvfb in the background, start x11vnc, and then run your application
CMD Xvfb :99 -screen 0 1024x768x16 & \
    x11vnc -display :99 -bg -nopw -listen localhost -xkb & \
    python3 /app/main.py
    
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