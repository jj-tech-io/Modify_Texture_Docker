import os
from pathlib import Path
import time
import subprocess
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# from keras.saving.save import load_model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf


import CONFIG
import importlib
importlib.reload(CONFIG)
def get_gpu_memory():
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # Decode the output to UTF-8 and parse the memory value
    memory_total_str = result.stdout.decode('utf-8').strip()
    # Assuming the output is in MiB, which is usual for nvidia-smi, convert to bytes
    memory_total = int(memory_total_str) * 1024 * 1024
    return memory_total
working_dir = os.getcwd()
encoder_path = Path(CONFIG.ENCODER_PATH)
decoder_path = Path(CONFIG.DECODER_PATH)

encoder = load_model(encoder_path)
decoder = load_model(decoder_path)
# encoder = load_model(CONFIG.ENCODER_PATH)
# decoder = load_model(CONFIG.DECODER_PATH)

# Print the number of available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def reverse_gamma_correction(img):
    """Reverse gamma correction on an image."""
    return np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

def gamma_correction(img):
    """Gamma correction on an image."""
    return np.where(img > 0.0031308, 1.055 * (img ** (1 / 2.4)) - 0.055, 12.92 * img)
def encode(img):
    image = np.asarray(img).reshape(-1,3).astype('float32')
    if np.max(image) > 1:
        print(f"max image {np.max(image)}")
        print(f"min image {np.min(image)}")
        image = image / 255.0
        print(f"max image {np.max(image)}")
        print(f"min image {np.min(image)}")
    # pred_maps = encoder.predict(image)
    image = reverse_gamma_correction(image)
    start = time.time()
    with tf.device('/device:GPU:0') as device:
        pred_maps = encoder.predict_on_batch(image)
        end = time.time()
        elapsed = end - start
        print(f"shape or encoded {pred_maps.shape}")
        return pred_maps, elapsed
    
def decode(encoded):
    # recovered = decoder.predict(encoded)
    start = time.time()
    with tf.device('/device:GPU:0') as device:
        #lower batch size to 2048
        recovered = decoder.predict_on_batch(encoded)
        # recovered = decoder.predict_on_batch(encoded)
    end = time.time()
    elapsed = end - start
    if np.max(recovered) > 2:
        #norm 0-1
        print(f"max recovered {np.max(recovered)}")
        print(f"min recovered {np.min(recovered)}")
        # recovered = (recovered - np.min(recovered)) / (np.max(recovered) - np.min(recovered))
        recovered = recovered / 255.0
        print(f"max recovered {np.max(recovered)}")
        print(f"min recovered {np.min(recovered)}")
        
    # recovered = np.clip(recovered, 0, 1)
    recovered = gamma_correction(recovered)
    return recovered, elapsed



def age_mel(v, t, r=0.08):
    """
    v is original volume fraction of melanin
    t is number of decades
    r is rate of decline (typical is 8%)
    """
    v_prime = v - (t * r) * v
    return v_prime


def age_hem(v, t, r_Hbi=0.06, r_Hbe=0.1, zeta=0.5):
    """
    v is original volume fraction of hemoglobin
    t is number of decades
    r is rate of decline (typical is 6%)
    """
    v_prime = v - t * (r_Hbi + zeta * r_Hbe) * v
    return v_prime


def get_masks(image):
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    WIDTH = 4096
    HEIGHT = 4096
    image = cv2.resize(image, (WIDTH, HEIGHT))
    parameter_maps, elapsed = encode(image)
    print("Encoding time: ", elapsed)
    print(
        f"parameter maps shape: {parameter_maps.shape} parameter maps type: {parameter_maps.dtype} parameter maps max: {np.max(parameter_maps)} parameter maps min: {np.min(parameter_maps)}")
    Cm = parameter_maps[:, 0].reshape(WIDTH, HEIGHT)
    # normalize 0-1
    print(f"min Cm: {np.min(Cm)} max Cm: {np.max(Cm)}")
    Cm = (Cm - np.min(Cm)) / (np.max(Cm) - np.min(Cm))
    print(f"min Cm: {np.min(Cm)} max Cm: {np.max(Cm)}")
    Ch = parameter_maps[:, 1].reshape(WIDTH, HEIGHT)
    # normalize 0-1
    Ch = (Ch - np.min(Ch)) / (np.max(Ch) - np.min(Ch))
    print(f"min Ch: {np.min(Ch)} max Ch: {np.max(Ch)}")
    Bm = parameter_maps[:, 2].reshape(WIDTH, HEIGHT)
    Bm = (Bm - np.min(Bm)) / (np.max(Bm) - np.min(Bm))
    print(f"min Bm: {np.min(Bm)} max Bm: {np.max(Bm)}")
    Bh = parameter_maps[:, 3].reshape(WIDTH, HEIGHT)
    Bh = (Bh - np.min(Bh)) / (np.max(Bh) - np.min(Bh))
    print(f"min Bh: {np.min(Bh)} max Bh: {np.max(Bh)}")
    T = parameter_maps[:, 4].reshape(WIDTH, HEIGHT)
    T = (T - np.min(T)) / (np.max(T) - np.min(T))
    # clip to 0-1
    Cm = np.clip(Cm, 0, 1)
    # Cm = 1 - Cm
    Ch = np.clip(Ch, 0, 1)
    # Ch = 1 - Ch
    Bm = np.clip(Bm, 0, 1)
    # Bm = 1 - Bm
    Bh = np.clip(Bh, 0, 1)
    # Bh = 1 - Bh
    T = np.clip(T, 0, 1)
    # T = 1 - T
    return Cm, Ch, Bm, Bh, T