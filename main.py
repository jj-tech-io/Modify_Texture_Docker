import sys
sys.path.append('morph')
import morph
from morph import *
import cv2

import transform_objects
from transform_objects import *
import generate_masks
from generate_masks import *


WIDTH = 4096
HEIGHT = 4096
#morph the source image to the target image
def morph_images(example_image_path, target_image_path):
    # Read the images into NumPy arrays
    target_image = cv2.imread(target_image_path)
    example_image = cv2.imread(example_image_path)
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
    warped_example_image, delaunay, transformation_matrices = warp_image(example_image, target_image, landmarks1, landmarks2)
    return warped_example_image, target_image, example_image


#extract masks from source
def extract_masks(image_path):
    Cm, Ch, Bm, Bh, T = get_masks(image_path)
    return Cm, Bh

#apply masks and transformations to target's latent space
def apply_transforms(target_image, mel_aged, oxy_aged):
    app = SkinParameterAdjustmentApp(image=target_image, mel_aged=mel_aged, oxy_aged=oxy_aged)
    app.run()


if __name__ == '__main__':
    example_texture_path = r"textures\m32_8k.png"
    target_texture_path = r"textures\1_neutral.jpg"
    warped_example_image, target_image, example_image = morph_images(example_texture_path, target_texture_path)
    Cm, Bh = extract_masks(warped_example_image)
    fig, ax = plt.subplots(1, 5, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Example image")
    ax[2].imshow(cv2.cvtColor(warped_example_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Warped example image")
    ax[1].imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Target image")
    ax[3].imshow(Cm, cmap='gray')
    ax[3].set_title("Cm")
    ax[4].imshow(Bh, cmap='gray')
    ax[4].set_title("Bh")
    plt.tight_layout()
    plt.show()
    apply_transforms(target_image, Cm, Bh)