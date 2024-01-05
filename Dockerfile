# Use the base image that you've already built with Xvfb and other dependencies
FROM modify-texture-app
# Expose the port your application listens on
EXPOSE 8080
# Start Xvfb in the background and then run your main application
CMD Xvfb :99 -screen 0 1024x768x16 & \
    python main.py
# # build the image
# docker build -t modify-texture-app .
# # run the container
# docker run --gpus all -it --rm -p 8000:8000 modify-texture-app
# docker run --gpus all -it --rm -p 8000:8000 -v "C://Users//joeli//Dropbox//Code//HM_2023//TrainedModels://app//TrainedModels" modify-texture-app
#to windows path
# docker run --gpus all -it --rm -p 8000:8000 -v "C://Users//joeli//Dropbox//Code//HM_2023//TrainedModels://app//TrainedModels" modify-texture-app

