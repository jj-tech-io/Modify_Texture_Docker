import os
from pathlib import Path
from tkinter import *
from tkinter.ttk import *
import importlib
import sys
sys.path.append('morph')
sys.path.append('AE_Inference')
sys.path.append('transform_objects')
sys.path.append('segmentation')
import morph
from morph import *
import transform_objects
from transform_objects import *
import AE_Inference
from AE_Inference import encode, decode, age_mel, age_hem, get_masks
import segmentation
from segmentation import *
importlib.reload(segmentation)
importlib.reload(morph)
importlib.reload(transform_objects)

WIDTH = 4096
HEIGHT = 4096

# morph the source image to the target image
def morph_images(example_image_path, target_image_path):
    # Read the images into NumPy arrays
    target_image = cv2.imread(target_image_path.as_posix())
    example_image = cv2.imread(example_image_path.as_posix())
    # Check if the images are read correctly
    if example_image is None:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    if target_image is None:
        print(f"Error: could not read target image {target_image_path}")
        sys.exit()
    # Resize the images
    target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
    example_image = cv2.resize(example_image, (WIDTH, HEIGHT))
    landmarks1 = get_landmarks(example_image)
    landmarks2 = get_landmarks(target_image)
    warped_example_image, delaunay, transformation_matrices = warp_image(example_image, target_image,
        landmarks1, landmarks2)
    return warped_example_image, target_image, example_image

# extract masks from source
def extract_masks(image):
    Cm, Ch, Bm, Bh, T = get_masks(image)
    landmarks = get_landmarks(image)
    #inverted masks
    combined_mask, lips, eyes, nose, eye_bags, face, landmark_object, av_skin_color = create_combined_mask(image)
    skin = threshold_face_skin_area(image,av_skin_color,mask=combined_mask)
    #Cm and with combined mask
    mask1 = np.where(combined_mask == 0, 0, Bm)

    # Bh and with inverted combined mask
    mask2 = np.where(combined_mask == 0, Bh, 0)
    masks = [Cm, Ch, Bm, Bh, T, mask1, mask2]
    for i in range(len(masks)):
        masks[i] = (masks[i] - np.min(masks[i])) / (np.max(masks[i]) - np.min(masks[i]))
        masks[i] = np.clip(masks[i], 0, 1)*0.25
    return Cm, Ch, skin, face

# apply masks and transformations to target's latent space
def apply_transforms(target_image, mel_aged, oxy_aged, skin, face):
    app = SkinParameterAdjustmentApp(image=target_image, mel_aged=mel_aged, oxy_aged=oxy_aged,skin=skin,face=face)
    app.run()

if __name__ == '__main__':
    working_dir = os.getcwd()
    example_texture_path = r"textures/m32_8k.png"
    example_texture_path = Path(working_dir, example_texture_path)
    # target_texture_path = r"textures\template_base_uv.png"
    target_texture_path = r"textures/1_neutral.jpg"
    target_texture_path = r"textures/m53_4k.png"
    target_texture_path = Path(working_dir, target_texture_path)
    warped_example_image, target_image, example_image = morph_images(Path(example_texture_path), Path(target_texture_path))
    Cm, Bh, skin, face = extract_masks(warped_example_image)
    apply_transforms(target_image, Cm, Bh, skin, face)
