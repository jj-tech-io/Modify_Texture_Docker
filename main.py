import sys
sys.path.append('morph')
import morph
from morph import *
import cv2

import transform_objects
from transform_objects import *
#make model paths global constants
TARGET= r"m32_4k.png"
SOURCE = r"1_neutral.jpg"
#switch
SOURCE = r"m32_4k.png"
TARGET= r"1_neutral.jpg"
WIDTH = 4096
HEIGHT = 4096
#morph the source image to the target image
def morph_images(source_image_path = SOURCE, target_image_path = TARGET):

    # Read the images into NumPy arrays
    target_image = cv2.imread(target_image_path)
    source_image = cv2.imread(source_image_path)

    # Check if the images are read correctly
    if source_image is None or target_image is None:
        print("Error: could not read one of the images.")
        # Handle the error, for example by exiting the script
        sys.exit()

    # Resize the images
    target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
    source_image = cv2.resize(source_image, (WIDTH, HEIGHT))
    landmarks1 = get_landmarks(source_image)
    landmarks2 = get_landmarks(target_image)
    # Warp source_image to align with target_image
    warped_source_image, delaunay, transformation_matrices = warp_image(source_image, target_image, landmarks1, landmarks2)
    warped_landmarks1 = get_landmarks(warped_source_image)
    # plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    # plt.title("Source image")
    # plt.show()
    # plt.imshow(cv2.cvtColor(warped_source_image, cv2.COLOR_BGR2RGB))
    # plt.title("Warped image")
    # plt.show()
    # plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    # plt.title("Target image")
    # plt.show()

    # Load and resize the masks
    mask_dir = r"masks"
    warped_mask_dir = r"warped_masks"
    mask_paths = os.listdir(mask_dir)
    masks = []
    warped_masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(os.path.join(mask_dir, mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (WIDTH, HEIGHT))
        masks.append(mask)
    # Warp each mask to align with the target image
    for i, mask in enumerate(masks):
        warped_mask = apply_transformations_to_single_channel_image(mask, transformation_matrices)
        warped_masks.append(warped_mask)
        cv2.imwrite(os.path.join(warped_mask_dir, mask_paths[i]), warped_mask)
    return warped_source_image, target_image, warped_masks

#encode the images into biophysical latent space
def encode_images(warped_source_image, target_image):
    #encode the image
    pass

#extract masks from source
def extract_masks(source_image):
    pass

#apply masks and transformations to target's latent space
def apply_transforms(target_image):
    #save the target image
    path = r"warped_images/im1.png"
    cv2.imwrite(path, target_image)
    app = SkinParameterAdjustmentApp(image_path=path,WIDTH=128, HEIGHT=128, mel_aged_path=r"warped_masks\Melanin_Age_Mask_filled_warped.png", oxy_aged_path=r"warped_masks\oxy_deoxy_mask_filled_warped_best.png")
    app.run()

#decode the target image
def decode_target():
    pass

if __name__ == '__main__':
    warped_source_image, target_image, warped_masks = morph_images()
    encode_images(warped_source_image, target_image)

    extract_masks(warped_source_image)
    apply_transforms(warped_source_image)
    decode_target()