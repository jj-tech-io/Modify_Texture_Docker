# This Dockerfile is used to build your application image using the base image

# Use the base image that contains all the dependencies
FROM python-base

# Copy the current directory contents into the container at /app
COPY . /app

# You can still expose the port if your application is a web server
EXPOSE 8000

# Define environment variable if needed
ENV NAME World

# Run main.py when the container launches
CMD ["python", "main.py"]


# # build the image
# docker build -t modify-texture-app .
# # run the container
# docker run -it --rm -p 8000:8000 modify-texture-app