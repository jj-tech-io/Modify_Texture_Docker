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
from pathlib import Path

importlib.reload(CONFIG)
sys.path.append(r"AE_Inference")
import AE_Inference
from AE_Inference import encode, decode, age_mel, age_hem
importlib.reload(AE_Inference)


class SkinParameterAdjustmentApp:
    def __init__(self, image, mel_aged, oxy_aged, skin, face, oxy_mask, save_name="recovered"):
        self.save_name = save_name
        self.skin = cv2.resize(skin, (512, 512))
        self.face = cv2.resize(face, (512, 512))
        self.oxy_mask = cv2.resize(oxy_mask, (512, 512))
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
        self.root.configure(background='black')
        self.root["bg"] = "black"
        self.root.title("Interactive Skin Parameter Adjustment")
        self.create_gui()

    def load_models(self):
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
        oxy_aged = cv2.bitwise_not(oxy_aged)
        oxy_aged = cv2.GaussianBlur(oxy_aged, (15, 15), 0)
        oxy_aged = oxy_aged / np.max(np.abs(oxy_aged))
        oxy_aged *= 0.5

        mel_aged = mel_aged.reshape(-1, )
        oxy_aged = oxy_aged.reshape(-1, )
        parameter_maps_original, encode_time = encode(original_image.reshape(-1, 3) / 255.0)
       
        self.original_image = original_image
        self.modified_image = modified_image
        self.mel_aged = mel_aged
        self.oxy_aged = oxy_aged
        self.parameter_maps = parameter_maps_original.copy()
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
        parameter_maps = self.parameter_maps_original_4k.copy()
        age_coef = self.age_coef_slider.get()
        scale_c_m = self.cm_slider.get()
        scale_c_h = self.ch_slider.get()
        scale_b_m = self.bm_slider.get()
        scale_b_h = self.bh_slider.get()
        scale_t = self.t_slider.get()
        cm_mask_slider = self.cm_mask_slider.get()
        bh_mask_slider = self.bh_mask_slider.get()
        global_scaling_maps = self.global_scaling_maps_slider.get()
        global_scaling_masks = self.global_scaling_masks_slider.get()
        parameter_maps[:, 0] = age_mel(parameter_maps[:, 0], age_coef)
        parameter_maps[:, 1] = age_hem(parameter_maps[:, 1], age_coef)
        parameter_maps[:, 0] = scale_c_m* parameter_maps[:, 0]
        parameter_maps[:, 1] = scale_c_h* parameter_maps[:, 1]
        parameter_maps[:, 2] = scale_b_m*parameter_maps[:, 2]
        parameter_maps[:, 3] = scale_b_h*parameter_maps[:, 3]
        parameter_maps[:, 4] = scale_t*parameter_maps[:, 4]
        cm_new =  (cm_mask_slider * self.mel_aged.reshape(-1)) + (1 - cm_mask_slider) * parameter_maps[:, 0]
        parameter_maps[:, 0] = cm_new
        self.oxy_mask = cv2.resize(self.oxy_mask, (self.WIDTH, self.HEIGHT))
        parameter_maps[:, 3] = np.where(self.oxy_mask.reshape(-1) == 0, parameter_maps[:, 3], (bh_mask_slider * self.oxy_aged.reshape(-1)) + parameter_maps[:, 3])
        recovered, decode_time = decode(parameter_maps)
        recovered = np.asarray(recovered).reshape((self.WIDTH, self.HEIGHT, 3))
        recovered = (recovered * 255).clip(0, 255).astype(np.uint8)  # Ensure proper range and type
        recovered = cv2.cvtColor(recovered, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        save_path = f"{self.save_name}.png"
        try:
            cv2.imwrite(save_path, recovered)
            print(f"Saved {save_path}")
        except Exception as e:
            print(f"Error: could not save image {save_path}")
            print(e)

        #print all shapes
        print(f"original_image.shape: {self.original_image.shape}")
        print(f"modified_image.shape: {self.modified_image.shape}")
        print(f"original_4k.shape: {self.original_4k.shape}")
        print(f"modified_4k.shape: {self.modified_4k.shape}")
        print(f"mel_aged.shape: {self.mel_aged.shape}")
        print(f"oxy_aged.shape: {self.oxy_aged.shape}")
        print(f"mel_aged_4k.shape: {self.mel_aged_4k.shape}")
        print(f"oxy_aged_4k.shape: {self.oxy_aged_4k.shape}")
        print(f"parameter_maps_original.shape: {self.parameter_maps_original.shape}")
        self.WIDTH = 512
        self.HEIGHT = 512
        self.oxy_mask = cv2.resize(self.oxy_mask, (self.WIDTH, self.HEIGHT))

        self.load_images()

    def update_plot(self, changed_slider=None):
        parameter_maps_original = self.parameter_maps_original.copy()
        parameter_maps = self.parameter_maps_original.copy()
        age_coef = self.age_coef_slider.get()
        scale_c_m = self.cm_slider.get()
        scale_c_h = self.ch_slider.get()
        scale_b_m = self.bm_slider.get()
        scale_b_h = self.bh_slider.get()
        scale_t = self.t_slider.get()
        cm_mask_slider = self.cm_mask_slider.get()
        bh_mask_slider = self.bh_mask_slider.get()
        global_scaling_maps = self.global_scaling_maps_slider.get()
        global_scaling_masks = self.global_scaling_masks_slider.get()
        parameter_maps[:, 0] = age_mel(parameter_maps[:, 0], age_coef)
        parameter_maps[:, 1] = age_hem(parameter_maps[:, 1], age_coef)
        parameter_maps[:, 0] = scale_c_m* parameter_maps[:, 0]
        parameter_maps[:, 1] = scale_c_h* parameter_maps[:, 1]
        parameter_maps[:, 2] = scale_b_m*parameter_maps[:, 2]
        parameter_maps[:, 3] = scale_b_h*parameter_maps[:, 3]
        parameter_maps[:, 4] = scale_t*parameter_maps[:, 4]
        cm_new =  (cm_mask_slider * self.mel_aged.reshape(-1)) + (1 - cm_mask_slider) * parameter_maps[:, 0]
        parameter_maps[:, 0] = cm_new
        parameter_maps[:, 3] = np.where(self.oxy_mask.reshape(-1) == 0, parameter_maps[:, 3], (bh_mask_slider * self.oxy_aged.reshape(-1)) + parameter_maps[:, 3])
        recovered, decode_time = decode(parameter_maps)
        recovered = np.asarray(recovered).reshape((self.WIDTH, self.HEIGHT, 3)) * 255
        self.parameter_maps = parameter_maps
        self.modified_image = recovered
        self.update_images(self.original_image, self.modified_image)
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
        self.frame_sliders.configure(width=900)
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.1,2.0)
        self.global_scaling_maps_slider = self.create_slider(self.frame_sliders, "Map(s):", 0, 2, 0.1, 1.0)
        self.global_scaling_masks_slider = self.create_slider(self.frame_sliders, "Mask(s):", 0, 2, 0.1, 1.0)
        self.cm_slider = self.create_slider(self.frame_sliders, "Cm:", 0, 2, 0.1, 1)
        self.ch_slider = self.create_slider(self.frame_sliders, "Ch:", 0, 2, 0.1, 1)
        self.bm_slider = self.create_slider(self.frame_sliders, "Bm:", 0, 2, 0.1, 1)
        self.bh_slider = self.create_slider(self.frame_sliders, "Bh:", 0, 2, 0.1, 1)
        self.t_slider = self.create_slider(self.frame_sliders, "T:", 0, 2, 0.1, 1)
        self.cm_mask_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.1, 0.6)
        self.bh_mask_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -1, 1, 0.1, 0.1)
        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot(changed_slider='age_coef'))
        self.global_scaling_maps_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot(changed_slider='global_scaling_maps'))
        self.global_scaling_masks_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot(changed_slider='global_scaling_masks'))

        # Correct the bindings to use the correct button release event and the slider name
        self.cm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('cm'))
        self.ch_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('ch'))
        self.bm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bm'))
        self.bh_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bh'))
        self.t_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('t'))
        self.cm_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('cm_mask'))
        self.bh_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bh_mask'))
        self.save_button.bind("<ButtonRelease-1>", lambda event: self.save_4k_image())

        # make window resizable
        self.root.resizable(True, True)
        self.update_plot()

    def run(self):
        self.root.mainloop()