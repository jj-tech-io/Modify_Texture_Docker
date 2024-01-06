# Use the base image from the previous step
FROM modify-texture-app-base

# Set the working directory (if needed, though it should be /app from the base image)
WORKDIR /app

# Expose the port the application listens on (if the app has a web interface)
EXPOSE 8888

# Start Xvfb in the background and then run your Jupyter notebook or application
CMD Xvfb :99 -screen 0 1024x768x16 & \
    jupyter notebook --notebook-dir=/app --ip=0.0.0.0 --no-browser --allow-root

# # build the image
# docker build -t modify-texture-app .
# # run the container
# docker run --gpus all -it --rm -p 8000:8000 modify-texture-app
# docker run --gpus all -it --rm -p 8000:8000 -v "C://Users//joeli//Dropbox//Code//HM_2023//TrainedModels://app//TrainedModels" modify-texture-app
#to windows path
# docker run --gpus all -it --rm -p 8000:8000 -v "C://Users//joeli//Dropbox//Code//HM_2023//TrainedModels://app//TrainedModels" modify-texture-app

