import importlib
import sys
import time
import tkinter as tk
from tkinter import ttk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import CONFIG

importlib.reload(CONFIG)
sys.path.append(r"AE_Inference")
import AE_Inference
from AE_Inference import encode, decode, age_mel, age_hem
importlib.reload(AE_Inference)


class SkinParameterAdjustmentApp:
    def __init__(self, image, mel_aged, oxy_aged):
        self.original_label = None
        self.modified_label = None
        self.encoder = None
        self.decoder = None
        self.image = image
        self.mel_aged = mel_aged
        self.oxy_aged = oxy_aged
        self.WIDTH = 512
        self.HEIGHT = 512
        self.load_models()
        self.load_images()
        self.init_app()

    def init_app(self):
        self.root = tk.Tk()
        self.root.title("Interactive Skin Parameter Adjustment")
        self.create_gui()

    def load_models(self):
        print(f"Loading models from {CONFIG.ENCODER_PATH} and {CONFIG.DECODER_PATH}")
        self.decoder = load_model(CONFIG.DECODER_PATH)
        self.encoder = load_model(CONFIG.ENCODER_PATH)

    def load_images(self):
        original_image = None
        mel_aged = None
        oxy_aged = None
        if type(self.image) == str:
            original_image = cv2.imread(self.image, cv2.IMREAD_COLOR)
        else:
            original_image = self.image
        if type(self.mel_aged) == str:
            mel_aged = cv2.imread(self.mel_aged, cv2.IMREAD_GRAYSCALE)
        else:
            mel_aged = self.mel_aged
        if type(self.oxy_aged) == str:
            oxy_aged = cv2.imread(self.oxy_aged, cv2.IMREAD_GRAYSCALE)
        else:
            oxy_aged = self.oxy_aged
        if self.image is None or self.mel_aged is None or self.oxy_aged is None:
            print(f"Error: read [image, mel_aged, oxy_aged] = [{self.image}, {self.mel_aged}, {self.oxy_aged}]")
            sys.exit()

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        original_4k = original_image.copy().astype(np.float32)
        modified_4k = original_4k.copy().astype(np.float32)
        original_image = cv2.resize(original_image, (self.WIDTH, self.HEIGHT))
        modified_image = original_image.copy()
        mel_aged = cv2.resize(mel_aged, (self.WIDTH, self.HEIGHT))
        oxy_aged = cv2.resize(oxy_aged, (self.WIDTH, self.HEIGHT))
        print(
            f"mel_aged shape: {mel_aged.shape} dtype: {mel_aged.dtype} max: {np.max(mel_aged)} min: {np.min(mel_aged)}")
        print(
            f"oxy_aged shape: {oxy_aged.shape} dtype: {oxy_aged.dtype}  max: {np.max(oxy_aged)} min: {np.min(oxy_aged)}")
        oxy_aged = cv2.bitwise_not(oxy_aged)
        oxy_aged = cv2.GaussianBlur(oxy_aged, (15, 15), 0)
        oxy_aged = oxy_aged / np.max(np.abs(oxy_aged))
        oxy_aged *= 0.1

        mel_aged = mel_aged.reshape(-1, )
        oxy_aged = oxy_aged.reshape(-1, )
        parameter_maps_original, encode_time = encode(original_image.reshape(-1, 3) / 255.0)
        print(
            f"original image shape: {original_image.shape} dtype: {original_image.dtype} max: {np.max(original_image)} min: {np.min(original_image)}")
        print(
            f"modified image shape: {modified_image.shape} dtype: {modified_image.dtype} max: {np.max(modified_image)} min: {np.min(modified_image)}")
        print(
            f"mel_aged shape: {mel_aged.shape} dtype: {mel_aged.dtype} max: {np.max(mel_aged)} min: {np.min(mel_aged)} mean: {np.mean(mel_aged)}")
        print(
            f"oxy_aged shape: {oxy_aged.shape} dtype: {oxy_aged.dtype}  max: {np.max(oxy_aged)} min: {np.min(oxy_aged)} mean: {np.mean(oxy_aged)}")
        self.original_image = original_image
        self.modified_image = modified_image
        self.mel_aged = mel_aged
        self.oxy_aged = oxy_aged
        self.parameter_maps_original = parameter_maps_original
        self.parameter_maps_original, self.encode_time = encode(self.original_image.reshape(-1, 3) / 255.0)
        self.original_4k = cv2.resize(original_4k, (self.WIDTH, self.WIDTH))
        self.modified_4k = cv2.resize(modified_4k, (self.WIDTH, self.WIDTH))
        self.mel_aged_4k = cv2.resize(mel_aged.reshape((self.WIDTH, self.HEIGHT)), (self.WIDTH, self.WIDTH))
        self.oxy_aged_4k = cv2.resize(oxy_aged.reshape((self.WIDTH, self.HEIGHT)), (self.WIDTH, self.WIDTH))
        self.parameter_maps_original_4k, self.encode_time_4k = encode(self.original_4k.reshape(-1, 3) / 255.0)

    def create_slider(self, parent, label, from_, to, resolution, default_value):
        frame = ttk.Frame(parent)
        label = ttk.Label(frame, text=label)
        label.pack(side=tk.LEFT)
        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
        slider.set(default_value)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        frame.pack()
        return slider

    def save_4k_image(self):
        self.WIDTH = 4096
        self.HEIGHT = 4096
        self.load_images()
        age_coef = self.age_coef_slider.get()
        global_scaling_maps = self.global_scaling_maps_slider.get()
        global_scaling_masks = self.global_scaling_masks_slider.get()
        scale_c_m = self.scaling_map1_slider.get()
        scale_c_h = self.scaling_map2_slider.get()
        scale_b_m = self.scaling_map3_slider.get()
        scale_b_h = self.scaling_map4_slider.get()
        scale_t = self.scaling_map5_slider.get()
        scale_mask_mel = self.scaling_mask1_slider.get()
        scale_mask_oxy_aged = self.scaling_mask2_slider.get()

        # Apply scaling factors for the maps and masks
        parameter_maps_changed, encode_time = encode(self.original_4k.reshape(-1, 3) / 255.0)

        # Update maps based on scaling factors
        parameter_maps_changed[:, 0] = age_mel(parameter_maps_changed[:, 0], age_coef)
        parameter_maps_changed[:, 0] += np.abs(self.mel_aged_4k.reshape(-1, )) 
        parameter_maps_changed[:,0] *= global_scaling_masks
        parameter_maps_changed[:, 1] = age_hem(parameter_maps_changed[:, 1], age_coef)
        parameter_maps_changed[:, 3] += np.abs(self.oxy_aged_4k.reshape(-1, ))
        parameter_maps_changed[:, 3] *= global_scaling_masks
        
        parameter_maps_changed[:, 2] = age_mel(parameter_maps_changed[:, 2], age_coef)

        parameter_maps_changed[:, 0] *= scale_c_m
        parameter_maps_changed[:, 0] *= global_scaling_maps
        parameter_maps_changed[:, 1] *= scale_c_h
        parameter_maps_changed[:, 1] *= global_scaling_maps
        parameter_maps_changed[:, 2] *= scale_b_m
        parameter_maps_changed[:, 2] *= global_scaling_maps
        parameter_maps_changed[:, 3] *= scale_b_h
        parameter_maps_changed[:, 3] *= global_scaling_maps
        parameter_maps_changed[:, 4] *= scale_t
        parameter_maps_changed[:, 4] *= global_scaling_maps
        # Apply scaling factors for the masks
        mel_aged_mask_scaled = self.mel_aged_4k.copy() * scale_mask_mel
        mel_aged_mask_scaled = cv2.resize(mel_aged_mask_scaled, (4096, 4096))
        oxy_aged_mask_scaled = self.oxy_aged_4k.copy() * scale_mask_oxy_aged
        oxy_aged_mask_scaled = cv2.resize(oxy_aged_mask_scaled, (4096, 4096))

        # Apply scaling factors for the masks
        mel_aged_mask_scaled = self.mel_aged_4k.copy() * scale_mask_mel
        oxy_aged_mask_scaled = self.oxy_aged_4k.copy() * scale_mask_oxy_aged

        parameter_maps_changed[:, 0] += np.abs(mel_aged_mask_scaled.reshape(-1)) * global_scaling_masks
        parameter_maps_changed[:, 1] += np.abs(oxy_aged_mask_scaled.reshape(-1)) * global_scaling_masks

        # Decode the updated parameter maps to get the recovered image
        recovered, decode_time = decode(parameter_maps_changed)
        recovered = np.asarray(recovered).reshape((4096, 4096, 3)) * 255

        cv2.imwrite(r"modified/modified.png", cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB))
        self.WIDTH = 512
        self.HEIGHT = 512
        self.load_images()

    def update_plot(self):
        age_coef = self.age_coef_slider.get()
        # global_scaling_maps = self.global_scaling_maps_slider.get()
        # global_scaling_masks = self.global_scaling_masks_slider.get()
        scale_c_m = self.scaling_map1_slider.get()
        scale_c_h = self.scaling_map2_slider.get()
        scale_b_m = self.scaling_map3_slider.get()
        scale_b_h = self.scaling_map4_slider.get()
        scale_t = self.scaling_map5_slider.get()
        scale_mask_mel = self.scaling_mask1_slider.get()
        scale_mask_oxy_aged = self.scaling_mask2_slider.get()
        # Apply scaling factors for the maps and masks
        parameter_maps_changed, encode_time = encode(self.original_image.reshape(-1, 3) / 255.0)
        mel_aged_changed = self.mel_aged.copy()
        oxy_aged_changed = self.oxy_aged.copy()
        # Update maps based on scaling factors
        parameter_maps_changed[:, 0] = age_mel(parameter_maps_changed[:, 0], age_coef)
        parameter_maps_changed[:, 1] = age_hem(parameter_maps_changed[:, 1], age_coef)


        parameter_maps_changed[:, 0] += scale_c_m
        parameter_maps_changed[:, 1] += scale_c_h
        parameter_maps_changed[:, 2] += scale_b_m
        parameter_maps_changed[:, 3] += scale_b_h
        parameter_maps_changed[:, 4] += scale_t

        # Apply scaling factors for the masks
        mel_aged_mask_scaled = mel_aged_changed * scale_mask_mel
        oxy_aged_mask_scaled = oxy_aged_changed * scale_mask_oxy_aged

        parameter_maps_changed[:, 0] += np.abs(mel_aged_mask_scaled) 
        parameter_maps_changed[:, 1] += np.abs(oxy_aged_mask_scaled) 

        # Decode the updated parameter maps to get the recovered image
        recovered, decode_time = decode(parameter_maps_changed)
        recovered = np.asarray(recovered).reshape((self.WIDTH, self.HEIGHT, 3)) * 255
        self.modified_image = recovered
        # Update the displayed images
        self.update_images(self.original_image, recovered)

        return recovered

    def update_images(self, original, modified):
        if np.max(original) < 1:
            plt.imshow(original)
            plt.show()
            original *= 255

        if np.max(modified) < 1:
            plt.imshow(modified)
            plt.show()
            modified *= 255
        try:
            original_pil = Image.fromarray(np.uint8(original))
            modified_pil = Image.fromarray(np.uint8(modified))
        except:
            print(f"Error: could not convert original or modified to PIL images")
            sys.exit()
        original_photo = ImageTk.PhotoImage(original_pil.resize((512, 512)))
        modified_photo = ImageTk.PhotoImage(modified_pil.resize((512, 512)))

        if self.original_label is None:
            self.original_label = ttk.Label(self.frame_images, image=original_photo)
            self.original_label.image = original_photo
            self.original_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.original_label.configure(image=original_photo)
            self.original_label.image = original_photo

        if self.modified_label is None:
            self.modified_label = ttk.Label(self.frame_images, image=modified_photo)
            self.modified_label.image = modified_photo
            self.modified_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.modified_label.configure(image=modified_photo)
            self.modified_label.image = modified_photo

    def create_gui(self):
        self.frame_sliders = ttk.Frame(self.root)
        self.frame_sliders.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #make sliders wider
        self.frame_sliders.configure(width=500)

        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.01, 1.0)
        # self.global_scaling_maps_slider = self.create_slider(self.frame_sliders, "Map(s):", 0, 2, 0.01, 1.0)
        # self.global_scaling_masks_slider = self.create_slider(self.frame_sliders, "Mask(s):", 0, 2, 0.01, 0.1)
        self.scaling_map1_slider = self.create_slider(self.frame_sliders, "Cm:", -0.6, 0.6, 0.01, 0)
        self.scaling_map2_slider = self.create_slider(self.frame_sliders, "Ch:", -0.32, 0.32, 0.01, 0)
        self.scaling_map3_slider = self.create_slider(self.frame_sliders, "Bm:", -1, 1, 0.01, 0)
        self.scaling_map4_slider = self.create_slider(self.frame_sliders, "Bh:", -1, 1, 0.01, 0)
        self.scaling_map5_slider = self.create_slider(self.frame_sliders, "T:", -0.2, 0.2, 0.01, 0)
        self.scaling_mask1_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.01, 0)
        self.scaling_mask2_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -1, 1, 0.01, 0)


        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        # self.global_scaling_maps_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        # self.global_scaling_masks_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map1_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map2_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map3_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map4_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map5_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_mask1_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_mask2_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        # make window resizable
        self.root.resizable(True, True)
        #black background
        self.root.configure(background='black')
        self.update_plot()

    def run(self):
        self.root.mainloop()
