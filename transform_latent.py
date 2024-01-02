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

import CONFIG
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
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load and preprocess the model
WIDTH = HEIGHT = 512
model_path = r"C:\Users\joeli\Dropbox\Data\models_4k\all\m53_4k.png"

# model_path = r"C:\Desktop\cousin 46\FaceColor_MAIN.PNG"
model = cv2.imread(model_path, cv2.IMREAD_COLOR)
model = cv2.cvtColor(model, cv2.COLOR_BGR2RGB)
model = cv2.resize(model, (WIDTH, HEIGHT))

# Assuming you have functions encode and decode that generate parameter_maps
parameter_maps, encode_time = encode(model.reshape(-1, 3) / 255.0)

# Load and preprocess the masks
mel_path = r"C:\Users\joeli\Dropbox\Data\masks\Melanin_Age_Mask.png"
mel_path = r"C:\Users\joeli\Dropbox\HM_Oct\metahuman masks\Melanin_Age_Mask_filled_warped.png"
oxy_path = r"C:\Users\joeli\Dropbox\Data\masks\oxy_deoxy_mask_dark.png"
oxy_path = r"C:\Users\joeli\Dropbox\HM_Oct\metahuman masks\oxy_deoxy_mask_filled_warped_best.png"
aged = cv2.imread(mel_path, cv2.IMREAD_COLOR)
aged = cv2.resize(aged, (WIDTH, HEIGHT))
aged = cv2.bitwise_not(aged)
aged = cv2.imread(mel_path, cv2.IMREAD_GRAYSCALE)
aged = cv2.resize(aged, (WIDTH, HEIGHT))
aged = (aged - np.min(aged))/(np.max(aged) - np.min(aged))

oxy = cv2.imread(oxy_path, cv2.IMREAD_GRAYSCALE)
oxy = cv2.resize(oxy, (WIDTH, HEIGHT))
oxy = cv2.bitwise_not(oxy)
oxy = cv2.GaussianBlur(oxy, (15, 15), 0)
oxy = oxy / np.max(np.abs(oxy))
oxy *= 0.1

aged = aged.reshape(-1,)
oxy = oxy.reshape(-1,)

parameter_maps_original = parameter_maps.copy().reshape(-1, 5)

# Create interactive widgets for age and scaling factors
age_coef_slider = widgets.FloatSlider(value=6, min=0, max=10, step=0.01, description='Age Coefficient:')
global_scaling_maps_slider = widgets.FloatSlider(value=0.1, min=0, max=2, step=0.01, description='All Maps:')
global_scaling_masks_slider = widgets.FloatSlider(value=0.1, min=0, max=2, step=0.01, description='All Masks:')
scaling_map1_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Scale Cm:')
scaling_map2_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Scale Ch:')
scaling_map3_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Scale Bm:')
scaling_map4_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Scale Bh:')
scaling_map5_slider = widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Scale T:')
scaling_mask1_slider = widgets.FloatSlider(value=0.1, min=0, max=2, step=0.01, description='Scale Melanin Mask:')
scaling_mask2_slider = widgets.FloatSlider(value=0.1, min=0, max=2, step=0.01, description='Scale Oxy Mask:')

# Save button
save_button = widgets.Button(description="Save 4K Image")

# Function to update the plot based on user input
def update_plot(age_coef, global_scaling_maps, global_scaling_masks, scale_c_m, scale_c_h, scale_b_m, scale_b_h, scale_t, scale_mask_mel, scale_mask_oxy):
    global oxy, parameter_maps_original  # Declare oxy and parameter_maps_original as global variables

    # Apply scaling factors for the maps
    parameter_maps_changed = parameter_maps_original.copy()
    parameter_maps_changed[:, 0] = age_mel(parameter_maps_original[:, 0], age_coef)
    parameter_maps_changed[:, 0] += np.abs(aged.reshape(-1,)) * global_scaling_maps
    parameter_maps_changed[:, 1] = age_hem(parameter_maps_original[:, 1], age_coef)
    parameter_maps_changed[:, 1] += np.abs(oxy.reshape(-1,)) * global_scaling_maps
    parameter_maps_changed[:, 2] = age_mel(parameter_maps_original[:, 2], age_coef)

    # Apply scaling factors for the masks
    aged_mask_scaled = aged.copy() * scale_mask_mel
    oxy_mask_scaled = oxy.copy() * scale_mask_oxy

    # Apply scaling factors for each map
    parameter_maps_changed[:, 0] *= scale_c_m
    parameter_maps_changed[:, 1] *= scale_c_h
    parameter_maps_changed[:, 2] *= scale_b_m
    parameter_maps_changed[:, 3] *= scale_b_h
    parameter_maps_changed[:, 4] *= scale_t

    # Apply scaled masks to maps
    parameter_maps_changed[:, 0] += np.abs(aged_mask_scaled) * global_scaling_masks
    parameter_maps_changed[:, 1] += np.abs(oxy_mask_scaled) * global_scaling_masks

    parameter_maps_changed[:, 3] *= 1
    parameter_maps_changed[:, 4] *= 1

    recovered, decode_time = decode(parameter_maps_changed)
    recovered = np.asarray(recovered).reshape((WIDTH, HEIGHT, 3))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(model)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(recovered)
    plt.title("Recovered Image")
    plt.show()

    return recovered

# Function to save the 4K image
def save_4k_image(button):
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
        scale_mask_oxy=scaling_mask2_slider.value
    )
    plt.imsave("recovered_image2_4k.png", recovered_image)

# Assign the save_4k_image function to the button click event
save_button.on_click(save_4k_image)

# Create an interactive plot with the save button
interactive_plot = widgets.interactive(
    update_plot,
    age_coef=age_coef_slider,
    global_scaling_maps=global_scaling_maps_slider,
    global_scaling_masks=global_scaling_masks_slider,
    scale_c_m=scaling_map1_slider,
    scale_c_h=scaling_map2_slider,
    scale_b_m=scaling_map3_slider,
    scale_b_h=scaling_map4_slider,
    scale_t=scaling_map5_slider,
    scale_mask_mel=scaling_mask1_slider,
    scale_mask_oxy=scaling_mask2_slider
)

# Display the interactive plot and save button
display(interactive_plot, save_button)


