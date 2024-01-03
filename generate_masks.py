import os
import time
import subprocess
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model

import CONFIG

def get_gpu_memory():
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    # Decode the output to UTF-8 and parse the memory value
    memory_total_str = result.stdout.decode('utf-8').strip()
    # Assuming the output is in MiB, which is usual for nvidia-smi, convert to bytes
    memory_total = int(memory_total_str) * 1024 * 1024
    return memory_total


encoder = load_model(r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\n75_l2_lr0.001_batch8192_weights0.2_0.1_0.7_epochs400\encoder.h5")
decoder = load_model(r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\n75_l2_lr0.001_batch8192_weights0.2_0.1_0.7_epochs400\decoder.h5")
#set batch size 
def reverse_gamma_correction(img):
    """Reverse gamma correction on an image."""
    return np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

def gamma_correction(img):
    """Gamma correction on an image."""
    return np.where(img > 0.0031308, 1.055 * (img ** (1 / 2.4)) - 0.055, 12.92 * img)
def encode(img):
    image = np.asarray(img).reshape(-1,3)
    # pred_maps = encoder.predict(image)
    start = time.time()
    pred_maps = None
    with tf.device('/device:GPU:0'):
        pred_maps = encoder.predict_on_batch(image)
    end = time.time()
    elapsed = end - start
    # pred_maps = pred_maps.reshape((self.WIDTH, self.HEIGHT, 5))
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
def get_masks(image):
    
    # image = cv2.imread(image_path)
    print("Image shape: ", image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (4096, 4096))
    parameter_maps, elapsed = encode(image)
    print("Encoding time: ", elapsed)
    print("Parameter maps shape: ", parameter_maps.shape)
    Cm = parameter_maps[:, 0].reshape(4096, 4096)
    #normalize 0-1
    Cm = (Cm - np.min(Cm)) / (np.max(Cm) - np.min(Cm))
    print(f"min Cm: {np.min(Cm)} max Cm: {np.max(Cm)}")
    Ch = parameter_maps[:, 1].reshape(4096, 4096)
    #normalize 0-1
    Ch = (Ch - np.min(Ch)) / (np.max(Ch) - np.min(Ch))
    print(f"min Ch: {np.min(Ch)} max Ch: {np.max(Ch)}")
    Bm = parameter_maps[:, 2].reshape(4096, 4096)
    Bm = (Bm - np.min(Bm)) / (np.max(Bm) - np.min(Bm))
    print(f"min Bm: {np.min(Bm)} max Bm: {np.max(Bm)}")
    Bh = parameter_maps[:, 3].reshape(4096, 4096)
    Bh = (Bh - np.min(Bh)) / (np.max(Bh) - np.min(Bh))
    print(f"min Bh: {np.min(Bh)} max Bh: {np.max(Bh)}")
    T = parameter_maps[:, 4].reshape(4096, 4096)
    T = (T - np.min(T)) / (np.max(T) - np.min(T))
    return Cm, Ch, Bm, Bh, T

if __name__ == "__main__":
    image_path = r"textures\m32_8k.png"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (4096, 4096))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Cm, Ch, Bm, Bh, T = get_masks(image_path)