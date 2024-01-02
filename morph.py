from scipy.spatial import Delaunay
import cv2
import numpy as np
import mediapipe as mp
import skimage
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh

import imgaug.augmenters as iaa
import random
import re
import os
import shutil


from scipy.spatial import Delaunay
import cv2
import numpy as np
import mediapipe as mp
import skimage
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import matplotlib.pyplot as plt

def get_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                landmark = face_landmarks.landmark[i]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    return np.array(landmarks)
# def get_extended_landmarks(landmarks, image_shape):
#     # Calculate the convex hull of the original landmarks
#     hull = cv2.convexHull(landmarks)
    
#     # Include additional points around the boundary of the convex hull
#     boundary_points = np.array([
#         [0, 0],
#         [image_shape[1] // 2, 0],
#         [image_shape[1] - 1, 0],
#         [image_shape[1] - 1, image_shape[0] // 2],
#         [image_shape[1] - 1, image_shape[0] - 1],
#         [image_shape[1] // 2, image_shape[0] - 1],
#         [0, image_shape[0] - 1],
#         [0, image_shape[0] // 2]
#     ])
    
#     # Combine the original landmarks with the boundary points
#     extended_landmarks = np.vstack((landmarks, boundary_points))
    
#     # Include the points of the convex hull
#     extended_landmarks = np.vstack((extended_landmarks, hull.squeeze()))
    
#     return extended_landmarks

def get_extended_landmarks(landmarks, image_shape):
    # Calculate the convex hull of the original landmarks
    hull = cv2.convexHull(landmarks)
    
    # Define an offset from the edges to avoid possible warping artifacts
    offset = 1

    # Include additional points around the boundary of the convex hull
    boundary_points = np.array([
        [offset, offset],
        [image_shape[1] // 2, offset],
        [image_shape[1] - 1 - offset, offset],
        [image_shape[1] - 1 - offset, image_shape[0] // 2],
        [image_shape[1] - 1 - offset, image_shape[0] - 1 - offset],
        [image_shape[1] // 2, image_shape[0] - 1 - offset],
        [offset, image_shape[0] - 1 - offset],
        [offset, image_shape[0] // 2]
    ])
    
    # Combine the original landmarks with the boundary points
    extended_landmarks = np.vstack((landmarks, boundary_points))
    
    # Include the points of the convex hull
    extended_landmarks = np.vstack((extended_landmarks, hull.squeeze()))
    
    return extended_landmarks

def warp_image(target, source, landmarks1, landmarks2):
    # Extend landmarks with boundary and convex hull points
    landmarks1_extended = get_extended_landmarks(landmarks1, target.shape)
    landmarks2_extended = get_extended_landmarks(landmarks2, source.shape)
    
    # Compute Delaunay Triangulation for the extended landmarks
    delaunay = Delaunay(landmarks1_extended)
    warped_image = target.copy()
    transformation_matrices = []
    # Iterate through each triangle in the triangulation
    for simplex in delaunay.simplices:
        # Get the vertices of the triangle in both images
        src_triangle = landmarks1_extended[simplex]
        dest_triangle = landmarks2_extended[simplex]

        # Compute the bounding box of the triangle in both images
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))

        # Crop the triangle from the source and destination images
        src_cropped_triangle = target[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        dest_cropped_triangle = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)

        # Adjust coordinates to the cropped region
        src_triangle_adjusted = src_triangle - (src_rect[0], src_rect[1])
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])

        # Compute the affine transformation
        matrix = cv2.getAffineTransform(np.float32(src_triangle_adjusted), np.float32(dest_triangle_adjusted))

        # Warp the source triangle to the shape of the destination triangle
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dest_rect[2], dest_rect[3]))

        # Mask for the destination triangle
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), (1, 1, 1), 16, 0)

        # Place the warped triangle in the destination image
        warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * (1 - mask[:, :, None]) \
            + warped_triangle * mask[:, :, None]
        transformation_matrices.append((matrix, src_triangle, dest_triangle))
    return warped_image.astype(np.uint8), delaunay, transformation_matrices

if __name__ == '__main__':
    # target_image_path = r"C:\Users\joeli\Dropbox\Data\models_4k\light\m32_4k.png"
    # source_image_path = r"C:\Users\joeli\Dropbox\Data\face_image_data\facescape\2\models_reg\1_neutral.jpg"
    #switch
    target_image_path = r"C:\Users\joeli\Dropbox\Data\face_image_data\facescape\2\models_reg\1_neutral.jpg"
    source_image_path = r"C:\Users\joeli\Dropbox\Data\models_4k\light\m32_4k.png"

    # Read the images into NumPy arrays
    target_image = cv2.imread(target_image_path)
    source_image = cv2.imread(source_image_path)

    # Check if the images are read correctly
    if source_image is None or target_image is None:
        print("Error: could not read one of the images.")
        # Handle the error, for example by exiting the script
        sys.exit()

    # Define the new width and height
    WIDTH = 2048
    HEIGHT = 2048

    # Resize the images
    target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
    source_image = cv2.resize(source_image, (WIDTH, HEIGHT))

    source_landmarks = get_landmarks(source_image)
    target_landmarks = get_landmarks(target_image)

    # Warp source_image to align with target_image
    warped_source_image, delaunay, transformation_matrices = warp_image(source_image, target_image, source_landmarks, target_landmarks)
    warped_landmarks = get_landmarks(warped_source_image)

    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.title("Source image")
    plt.show()
    plt.imshow(cv2.cvtColor(warped_source_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped image")
    plt.show()

    plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.title("Target image")
    plt.show()


