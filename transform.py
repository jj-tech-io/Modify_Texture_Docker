import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import PIL.Image as Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
import datetime
import colorspacious
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
import numpy as np
from scipy.ndimage import gaussian_filter
import time
import importlib
import CONFIG
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.cm import ScalarMappable
from PIL import Image, ImageTk
import CONFIG

# Initialize the labels to None
original_label = None
modified_label = None
importlib.reload(CONFIG)
decoder = load_model(CONFIG.DECODER_PATH)
encoder = load_model(CONFIG.ENCODER_PATH)

def encode(img):
    image = np.asarray(img).reshape(-1,3)
    # pred_maps = encoder.predict(image)
    start = time.time()
    pred_maps = None
    with tf.device('/device:GPU:0'):
        pred_maps = encoder.predict_on_batch(image)
    end = time.time()
    elapsed = end - start
    # pred_maps = pred_maps.reshape((WIDTH, HEIGHT, 5))
    return pred_maps, elapsed
 
def decode(encoded):
    # recovered = decoder.predict(encoded)
    start = time.time()
    recovered = None
    with tf.device('/device:GPU:0'):
        recovered = decoder.predict_on_batch(encoded)
    end = time.time()
    elapsed = end - start
    # recovered = np.clip(recovered, 0, 1)
    return recovered, elapsed

def mean_squared_dif(map1, map2):
    dif = np.sqrt(np.sum((map1 - map2)**2, axis=1))
    return dif

def age_mel(v, t, r=0.08):
    """
    v is original volume fraction of melanin
    t is number of decades
    r is rate of decline (typical is 8%)
    """
    v_prime = v-(t*r)*v
    return v_prime

def age_hem(v, t, r_Hbi=0.06, r_Hbe=0.1, zeta=0.5):
    """
    v is original volume fraction of hemoglobin
    t is number of decades
    r is rate of decline (typical is 6%)
    """
    v_prime = v-t*(r_Hbi+zeta*r_Hbe)*v
    return v_prime

# Function to create sliders with a resolution parameter using tk.Scale
def create_slider(parent, label, from_, to, resolution, default_value):
    frame = ttk.Frame(parent)
    label = ttk.Label(frame, text=label)
    label.pack(side=tk.LEFT)
    # Using tk.Scale instead of ttk.Scale to use the resolution parameter
    slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
    slider.set(default_value)  # Set the default value here
    slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    frame.pack()
    return slider

# Save button logic
def save_4k_image():
    recovered_image = update_plot(
        age_coef=age_coef_slider.value,
        global_scaling_maps=global_scaling_maps_slider.value,
        global_scaling_masks=global_scaling_masks_slider.value,
        scale_c_m=scaling_map1_slider.value,
        scale_c_h=scaling_map2_slider.value,
        scale_b_m=scaling_map3_slider.value,
        scale_b_h=scaling_map4_slider.value,
        scale_t=scaling_map5_slider.value,
        scale_mask_mel=scaling_mask1_slider.value,
        scale_mask_oxy_aged=scaling_mask2_slider.value
    )
    plt.imsave("recovered_image2_4k.png", recovered_image)


# Update plot function
def update_plot():
    global mel_aged, oxy_aged, parameter_maps_original  # Declare oxy_aged and parameter_maps_original as global variables
    age_coef = age_coef_slider.get()
    global_scaling_maps = global_scaling_maps_slider.get()
    global_scaling_masks = global_scaling_masks_slider.get()
    scale_c_m = scaling_map1_slider.get()
    scale_c_h = scaling_map2_slider.get()
    scale_b_m = scaling_map3_slider.get()
    scale_b_h = scaling_map4_slider.get()
    scale_t = scaling_map5_slider.get()
    scale_mask_mel = scaling_mask1_slider.get()
    scale_mask_oxy_aged = scaling_mask2_slider.get()
    
    # Apply scaling factors for the maps
    parameter_maps_changed = parameter_maps_original.copy()
    parameter_maps_changed[:, 0] = age_mel(parameter_maps_original[:, 0], age_coef)
    parameter_maps_changed[:, 0] += np.abs(mel_aged.reshape(-1,)) * global_scaling_maps
    parameter_maps_changed[:, 1] = age_hem(parameter_maps_original[:, 1], age_coef)
    parameter_maps_changed[:, 1] += np.abs(oxy_aged.reshape(-1,)) * global_scaling_maps
    parameter_maps_changed[:, 2] = age_mel(parameter_maps_original[:, 2], age_coef)

    # Apply scaling factors for the masks
    mel_aged_mask_scaled = mel_aged.copy() * scale_mask_mel
    oxy_aged_mask_scaled = oxy_aged.copy() * scale_mask_oxy_aged

    # Apply scaling factors for each map
    parameter_maps_changed[:, 0] *= scale_c_m
    parameter_maps_changed[:, 1] *= scale_c_h
    parameter_maps_changed[:, 2] *= scale_b_m
    parameter_maps_changed[:, 3] *= scale_b_h
    parameter_maps_changed[:, 4] *= scale_t

    # Apply scaled masks to maps
    parameter_maps_changed[:, 0] += np.abs(mel_aged_mask_scaled) * global_scaling_masks
    parameter_maps_changed[:, 1] += np.abs(oxy_aged_mask_scaled) * global_scaling_masks

    parameter_maps_changed[:, 3] *= 1
    parameter_maps_changed[:, 4] *= 1

    recovered, decode_time = decode(parameter_maps_changed)
    recovered = np.asarray(recovered).reshape((WIDTH, HEIGHT, 3))*255

    update_images(original_image, recovered)

    return recovered

# Function to update the displayed images
def update_images(original, modified):
    global original_label, modified_label
    # Convert the NumPy arrays to PIL images
    original_pil = Image.fromarray(np.uint8(original))
    modified_pil = Image.fromarray(np.uint8(modified))

    # Convert images to Tkinter compatible format
    original_photo = ImageTk.PhotoImage(original_pil)
    modified_photo = ImageTk.PhotoImage(modified_pil)
    
    # If labels do not exist, create them
    if original_label is None:
        original_label = ttk.Label(frame_images, image=original_photo)
        original_label.image = original_photo  # keep a reference
        original_label.pack(side=tk.LEFT, padx=10, pady=10)
    else:
        # If they do, just update the image
        original_label.configure(image=original_photo)
        original_label.image = original_photo  # keep a reference
    
    if modified_label is None:
        modified_label = ttk.Label(frame_images, image=modified_photo)
        modified_label.image = modified_photo  # keep a reference
        modified_label.pack(side=tk.LEFT, padx=10, pady=10)
    else:
        # If they do, just update the image
        modified_label.configure(image=modified_photo)
        modified_label.image = modified_photo  # keep a reference



if __name__ == '__main__':
    original_image_path = r"m53_4k.png"
    WIDTH = 256
    HEIGHT = 256
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (WIDTH, HEIGHT))
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