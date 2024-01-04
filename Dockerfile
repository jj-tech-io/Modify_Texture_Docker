# Use the base image that you've already built with Xvfb and other dependencies
FROM modify-texture-app-base

# Set the working directory in the container to /app (if not already set)
WORKDIR /app

# Copy the rest of your application into the container at /app
COPY . /app

# Expose the port your application listens on
EXPOSE 8000

# Start Xvfb in the background and then run your main application
CMD Xvfb :99 -screen 0 1024x768x16 & \
    python main.py


# # build the image
# docker build -t modify-texture-app .
# # run the container
# docker run --gpus all -it --rm -p 8000:8000 modify-texture-app