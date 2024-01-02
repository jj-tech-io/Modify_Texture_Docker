import sys



import sys
sys.path.append('morph')
import morph
from morph import *
import cv2

import transform
from transform import *

#morph the source image to the target image
def morph_images(source_image_path = r"C:\Users\joeli\Dropbox\Data\models_4k\light\m32_4k.png", target_image_path = r"C:\Users\joeli\Dropbox\Data\face_image_data\facescape\2\models_reg\1_neutral.jpg"):

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

    landmarks1 = get_landmarks(source_image)
    landmarks2 = get_landmarks(target_image)

    # Warp source_image to align with target_image
    warped_source_image, delaunay, transformation_matrices = warp_image(source_image, target_image, landmarks1, landmarks2)

    warped_landmarks1 = get_landmarks(warped_source_image)

    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.title("Source image")
    plt.show()
    plt.imshow(cv2.cvtColor(warped_source_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped image")
    plt.show()

    plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.title("Target image")
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
        warped_mask = cv2.warpAffine(mask, transformation_matrices[i][0], (WIDTH, HEIGHT))
        warped_masks.append(warped_mask)

    # Save the warped masks
    for i in range(len(warped_masks)):
        cv2.imwrite(os.path.join(warped_mask_dir, mask_paths[i]), warped_masks[i])

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
    # original_image_path = r"m53_4k.png"
    WIDTH = 256
    HEIGHT = 256
    # original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # original_image = cv2.resize(original_image, (WIDTH, HEIGHT))
    original_image = target_image
    modified_image = original_image.copy()
    mel_path = r"masks\Melanin_Age_Mask_filled_warped.png"
    oxy_aged_path = r"masks\oxy_deoxy_mask_filled_warped_best.png"
    mel_aged = cv2.imread(mel_path, cv2.IMREAD_COLOR)
    mel_aged = cv2.resize(mel_aged, (WIDTH, HEIGHT))
    mel_aged = cv2.bitwise_not(mel_aged)
    mel_aged = cv2.imread(mel_path, cv2.IMREAD_GRAYSCALE)
    mel_aged = cv2.resize(mel_aged, (WIDTH, HEIGHT))
    mel_aged = (mel_aged - np.min(mel_aged))/(np.max(mel_aged) - np.min(mel_aged))

    oxy_aged = cv2.imread(oxy_aged_path, cv2.IMREAD_GRAYSCALE)
    oxy_aged = cv2.resize(oxy_aged, (WIDTH, HEIGHT))
    oxy_aged = cv2.bitwise_not(oxy_aged)
    oxy_aged = cv2.GaussianBlur(oxy_aged, (15, 15), 0)
    oxy_aged = oxy_aged / np.max(np.abs(oxy_aged))
    oxy_aged *= 0.1

    mel_aged = mel_aged.reshape(-1,)
    oxy_aged = oxy_aged.reshape(-1,)
    parameter_maps_original, encode_time = encode(original_image.reshape(-1, 3) / 255.0)
    # Assuming all necessary functions and variables are defined such as decode, model, parameter_maps_original

    # Main application window
    root = tk.Tk()
    root.title("Interactive Skin Parameter Adjustment")

    # Frame for sliders
    frame_sliders = ttk.Frame(root)
    frame_sliders.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # Frame for buttons
    frame_buttons = ttk.Frame(root)
    frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    frame_images = ttk.Frame(root)  # This is where frame_images is defined
    frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # Then create each slider with the default value
    age_coef_slider = create_slider(frame_sliders, "Age Coefficient:", 0, 10, 0.01, 6)
    global_scaling_maps_slider = create_slider(frame_sliders, "All Maps:", 0, 2, 0.01, 0.1)
    global_scaling_masks_slider = create_slider(frame_sliders, "All Masks:", 0, 2, 0.01, 0.1)
    scaling_map1_slider = create_slider(frame_sliders, "Scale Cm:", 0, 2, 0.01, 1)
    scaling_map2_slider = create_slider(frame_sliders, "Scale Ch:", 0, 2, 0.01, 1)
    scaling_map3_slider = create_slider(frame_sliders, "Scale Bm:", 0, 2, 0.01, 1)
    scaling_map4_slider = create_slider(frame_sliders, "Scale Bh:", 0, 2, 0.01, 1)
    scaling_map5_slider = create_slider(frame_sliders, "Scale T:", 0, 2, 0.01, 1)
    scaling_mask1_slider = create_slider(frame_sliders, "Scale Melanin Mask:", 0, 2, 0.01, 0.1)
    scaling_mask2_slider = create_slider(frame_sliders, "Scale oxy_aged Mask:", 0, 2, 0.01, 0.1)
    # Create a save button
    save_button = ttk.Button(frame_buttons, text="Save 4K Image", command=save_4k_image)
    save_button.pack(side=tk.RIGHT, padx=5, pady=5)

    age_coef_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    global_scaling_maps_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    global_scaling_masks_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_map1_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_map2_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_map3_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_map4_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_map5_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_mask1_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    scaling_mask2_slider.bind("<ButtonRelease-1>", lambda event: update_plot())
    update_plot()
    # Start the Tkinter event loop
    root.mainloop()

#decode the target image
def decode_target():
    pass

if __name__ == '__main__':
    warped_source_image, target_image, warped_masks = morph_images()
    encode_images(warped_source_image, target_image)

    extract_masks(warped_source_image)
    apply_transforms(target_image)
    decode_target()