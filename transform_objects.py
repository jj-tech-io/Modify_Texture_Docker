import importlib
import sys
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import CONFIG
from pathlib import Path
import main
from main import *
importlib.reload(CONFIG)
sys.path.append(r"AE_Inference")
import AE_Inference
from AE_Inference import encode, decode, age_mel, age_hem
importlib.reload(AE_Inference)


class SkinParameterAdjustmentApp:
    def __init__(self, image, mel_aged, oxy_aged, skin, face, oxy_mask, save_name="recovered"):
        self.example_texture_path = "textures/m32_8k.png"
        self.target_texture_path = "textures/m53_4k.png"
        self.save_name = save_name
        self.WIDTH = 4096
        self.HEIGHT = 4096
        try:
            self.skin = cv2.resize(skin, (self.WIDTH, self.HEIGHT))
            self.face = cv2.resize(face, (self.WIDTH, self.HEIGHT))
            self.oxy_mask = cv2.resize(oxy_mask, (self.WIDTH, self.HEIGHT))
            self.mel_aged = cv2.resize(mel_aged, (self.WIDTH, self.HEIGHT))
            self.oxy_aged = cv2.resize(oxy_aged, (self.WIDTH, self.HEIGHT))
            assert self.oxy_mask.shape == self.mel_aged.shape == self.oxy_aged.shape, "Error: Image shapes do not match"
            assert np.isnan(self.oxy_aged).any() == False and np.isnan(self.mel_aged).any() == False and np.isnan(self.skin).any() == False and np.isnan(self.face).any() == False and np.isnan(self.oxy_mask).any() == False, "Error: NaN values in image"
        except Exception as e:
            print(f"Error: could not resize skin, face, or oxy_mask: {e}")
            sys.exit()
        self.original_label = None
        self.modified_label = None
        self.encoder = None
        self.decoder = None
        self.image = image
        self.mel_aged = mel_aged
        self.oxy_aged = oxy_aged
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
        # oxy_aged = np.abs(oxy_aged) / np.max(np.abs(oxy_aged))
        oxy_aged *= 0.025

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

    def create_slider1(self, parent, label, from_, to, resolution, default_value):
        frame = ttk.Frame(parent)
        label = ttk.Label(frame, text=label)
        label.pack(side=tk.LEFT)
        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
        slider.set(default_value)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        frame.pack()
        return slider
    def create_slider(self, parent, label_text, from_, to, resolution, default_value):
        frame = ttk.Frame(parent)
        
        # Place the frame itself in the parent's grid
        frame.grid(sticky='ew')
        parent.grid_columnconfigure(0, weight=1)  # This makes the frame expand to fill the grid cell
        
        # Create label and slider within the frame using grid layout
        label = ttk.Label(frame, text=label_text)
        label.grid(row=0, column=0, sticky='w')  # Align label to the left (west)

        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
        slider.set(default_value)
        slider.grid(row=0, column=1, sticky='ew')  # Align slider to the right, expand horizontally

        frame.grid_columnconfigure(1, weight=1)  # This allows the slider to expand

        return slider
    def save_4k_image(self):
        self.update_images(self.original_image, self.modified_image, save=True)

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
        parameter_maps[:, 0] = age_mel(parameter_maps[:, 0], age_coef)
        parameter_maps[:, 1] = age_hem(parameter_maps[:, 1], age_coef)
        parameter_maps[:, 0] = scale_c_m * parameter_maps[:, 0]
        parameter_maps[:, 1] = scale_c_h * parameter_maps[:, 1]
        parameter_maps[:, 2] = scale_b_m * parameter_maps[:, 2]
        parameter_maps[:, 3] = scale_b_h * parameter_maps[:, 3]
        parameter_maps[:, 4] = scale_t * parameter_maps[:, 4]
        cm_new =  (cm_mask_slider * self.mel_aged.reshape(-1)) + (1 - cm_mask_slider) * parameter_maps[:, 0]
        parameter_maps[:, 0] = cm_new
        #only apply mask to eyes and lips
        print(f"shape skin {np.asarray(self.skin).shape}, shape face {np.asarray(self.face).shape}, shape oxy_aged {np.asarray(self.oxy_aged).shape}, shape parameter_maps {np.asarray(parameter_maps).shape}")
        # parameter_maps[:, 1] = np.where(self.skin[:,:,0].reshape(-1) != 0, 0, (bh_mask_slider * self.oxy_aged.reshape(-1)) + parameter_maps[:, 1])
        print(f"abs(bh_mask_slider * self.oxy_aged.reshape(-1)) = {bh_mask_slider * self.oxy_aged}")
        try:
            # bh_new = bh_mask_slider * self.oxy_aged.reshape(-1) + (1-bh_mask_slider)*parameter_maps[:, 3]
            bh_new = np.where(self.oxy_mask.reshape(-1) != 0, parameter_maps[:, 3], (bh_mask_slider * self.oxy_aged.reshape(-1)) + parameter_maps[:, 3])
            parameter_maps[:, 3] = bh_new
        except Exception as e:
            print(f"Error: could not update bh_mask {e}")
            sys.exit()
        recovered, decode_time = decode(parameter_maps)
        recovered = np.asarray(recovered).reshape((self.WIDTH, self.HEIGHT, 3)) * 255
        self.parameter_maps = parameter_maps
        self.modified_image = recovered
        self.update_images(self.original_image, self.modified_image)
        return recovered

    def update_images(self, original, modified, save=False):
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
            if save:
                save_path = f"{self.save_name}.png"
                modified_pil.save(save_path)
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
    def load_new(self):
        # Here you can call your morph_images and extract_masks functions
        warped_example_image, target_image, example_image = morph_images(Path(self.example_texture_path), Path(self.target_texture_path))
        Cm, Bh, skin, face, oxy_mask = extract_masks(warped_example_image)
        # Update the app's image attributes
        self.image = target_image
        self.mel_aged = Cm
        self.oxy_aged = Bh
        self.skin = skin
        self.face = face
        self.oxy_mask = oxy_mask
        self.original_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.modified_image = target_image
        self.load_images()
        self.update_plot()  # This is just a placeholder for whatever update you need to do

    def load_new_image(self):
        # Open the file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an image",
                                               filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.target_texture_path = file_path
        if file_path:  # If a file was selected
            print(f"Selected image: {file_path}")
            self.load_new()
    def load_example_image(self):
        # Open the file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an image",
                                               filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.example_texture_path = file_path
        if file_path:  # If a file was selected
            print(f"Selected image: {file_path}")
            self.load_new()

    def create_gui(self):
        self.frame_sliders = ttk.Frame(self.root)
        self.frame_sliders.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.frame_sliders.configure(width=900)
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.1,2.0)

        self.cm_slider = self.create_slider(self.frame_sliders, "Cm:", 0, 2, 0.1, 1)
        self.ch_slider = self.create_slider(self.frame_sliders, "Ch:", 0, 2, 0.1, 1)
        self.bm_slider = self.create_slider(self.frame_sliders, "Bm:", 0, 2, 0.1, 1)
        self.bh_slider = self.create_slider(self.frame_sliders, "Bh:", 0, 2, 0.1, 0.9)
        self.t_slider = self.create_slider(self.frame_sliders, "T:", 0, 2, 0.1, 1)
        self.cm_mask_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.1, 0.6)
        self.bh_mask_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -1, 1, 0.1, 0.1)
        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot(changed_slider='age_coef'))

        # Correct the bindings to use the correct button release event and the slider name
        self.cm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('cm'))
        self.ch_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('ch'))
        self.bm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bm'))
        self.bh_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bh'))
        self.t_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('t'))
        self.cm_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('cm_mask'))
        self.bh_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot('bh_mask'))
        self.save_button.bind("<ButtonRelease-1>", lambda event: self.save_4k_image())
        #add load image button
        load_image_button = ttk.Button(self.frame_buttons, text="Load New Image", command=self.load_new_image)
        load_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        #add load example image button
        load_example_image_button = ttk.Button(self.frame_buttons, text="Load Example Image", command=self.load_example_image)
        load_example_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        # make window resizable
        self.root.resizable(True, True)
        #size window
        self.root.geometry("1100x900")
        #window top left corner
        self.root.geometry("+0+0")
        self.update_plot()
    def create_gui1(self):
        # Make the window fullscreen
        self.root.state('zoomed')

        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Initialize frames with grid layout
        self.frame_sliders = ttk.Frame(self.root)
        self.frame_sliders.grid(row=0, column=0, sticky="ew", padx=10)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.grid(row=1, column=0, sticky="nsew")
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.grid(row=2, column=0, sticky="ew")

        # Configure the frame's column so that it expands equally
        self.frame_sliders.grid_columnconfigure(0, weight=1)

        # Create sliders and place them in the grid
        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.1, 2.0)
        self.cm_slider = self.create_slider(self.frame_sliders, "Cm:", 0, 2, 0.1, 1)
        self.ch_slider = self.create_slider(self.frame_sliders, "Ch:", 0, 2, 0.1, 1)
        self.bm_slider = self.create_slider(self.frame_sliders, "Bm:", 0, 2, 0.1, 1)
        self.bh_slider = self.create_slider(self.frame_sliders, "Bh:", 0, 2, 0.1, 0.9)
        self.t_slider = self.create_slider(self.frame_sliders, "T:", 0, 2, 0.1, 1)
        self.cm_mask_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.1, 0.6)
        self.bh_mask_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -1, 1, 0.1, 0.1)


        # Place the sliders evenly in the grid
        slider_names = ['age_coef', 'cm', 'ch', 'bm', 'bh', 't', 'cm_mask', 'bh_mask']
        for i, name in enumerate(slider_names):
            getattr(self, f"{name}_slider").grid(row=0, column=i, padx=5, pady=5, sticky="ew")

        # Add load image button
        load_image_button = ttk.Button(self.frame_buttons, text="Load New Image", command=self.load_new_image)
        load_image_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Add save button
        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")

        # Bind the sliders to the update function
        for name in slider_names:
            getattr(self, f"{name}_slider").bind("<ButtonRelease-1>", lambda event, name=name: self.update_plot(changed_slider=name))
        self.root.geometry("1100x900")
        #window top left corner
        self.root.geometry("+0+0")
        #full screen + center components
        # Update plot
        self.update_plot()
    def run(self):
        self.root.mainloop()