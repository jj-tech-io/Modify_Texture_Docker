import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import CONFIG
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

class SkinParameterAdjustmentApp:
    def __init__(self, image_path, WIDTH=256, HEIGHT=256, mel_path=r"masks\Melanin_Age_Mask_filled_warped.png", oxy_aged_path=r"masks\oxy_deoxy_mask_filled_warped_best.png"):
        self.original_label = None
        self.modified_label = None
        self.encoder = None
        self.decoder = None
        self.init_app()

    def init_app(self):
        self.load_models()
        self.load_images()
        self.root = tk.Tk()
        self.root.title("Interactive Skin Parameter Adjustment")
        self.create_gui()

    def load_models(self):
        self.decoder = load_model(CONFIG.DECODER_PATH)
        self.encoder = load_model(CONFIG.ENCODER_PATH)

    def load_images(self):
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
        self.original_image = original_image
        self.modified_image = modified_image
        self.mel_aged = mel_aged
        self.oxy_aged = oxy_aged
        self.parameter_maps_original = parameter_maps_original


        self.parameter_maps_original, self.encode_time = self.encode(self.original_image.reshape(-1, 3) / 255.0)

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
        recovered_image = self.update_plot(
            age_coef=age_coef_slider.get(),
            global_scaling_maps=global_scaling_maps_slider.get(),
            global_scaling_masks=global_scaling_masks_slider.get(),
            scale_c_m=scaling_map1_slider.get(),
            scale_c_h=scaling_map2_slider.get(),
            scale_b_m=scaling_map3_slider.get(),
            scale_b_h=scaling_map4_slider.get(),
            scale_t=scaling_map5_slider.get(),
            scale_mask_mel=scaling_mask1_slider.get(),
            scale_mask_oxy_aged=scaling_mask2_slider.get()
        )
        plt.imsave("recovered_image2_4k.png", recovered_image)

    def update_plot(self, age_coef, global_scaling_maps, global_scaling_masks,
                    scale_c_m, scale_c_h, scale_b_m, scale_b_h, scale_t, scale_mask_mel, scale_mask_oxy_aged):
                    
        # Apply scaling factors for the maps and masks
        parameter_maps_changed = self.parameter_maps_original.copy()
        
        # Update maps based on scaling factors
        parameter_maps_changed[:, 0] = age_mel(parameter_maps_original[:, 0], age_coef)
        parameter_maps_changed[:, 0] += np.abs(mel_aged.reshape(-1,)) * global_scaling_maps
        parameter_maps_changed[:, 1] = age_hem(parameter_maps_original[:, 1], age_coef)
        parameter_maps_changed[:, 1] += np.abs(oxy_aged.reshape(-1,)) * global_scaling_maps
        parameter_maps_changed[:, 2] = age_mel(parameter_maps_original[:, 2], age_coef)
        
        parameter_maps_changed[:, 0] *= scale_c_m
        parameter_maps_changed[:, 1] *= scale_c_h
        parameter_maps_changed[:, 2] *= scale_b_m
        parameter_maps_changed[:, 3] *= scale_b_h
        parameter_maps_changed[:, 4] *= scale_t

        # Apply scaling factors for the masks
        mel_aged_mask_scaled = mel_aged.copy() * scale_mask_mel
        oxy_aged_mask_scaled = oxy_aged.copy() * scale_mask_oxy_aged

        parameter_maps_changed[:, 0] += np.abs(mel_aged_mask_scaled) * global_scaling_masks
        parameter_maps_changed[:, 1] += np.abs(oxy_aged_mask_scaled) * global_scaling_masks

        # Decode the updated parameter maps to get the recovered image
        recovered, decode_time = self.decode(parameter_maps_changed)
        recovered = np.asarray(recovered).reshape((WIDTH, HEIGHT, 3)) * 255

        # Update the displayed images
        self.update_images(original_image, recovered)

        return recovered


    def update_images(self, original, modified):
        original_pil = Image.fromarray(np.uint8(original))
        modified_pil = Image.fromarray(np.uint8(modified))
        original_photo = ImageTk.PhotoImage(original_pil)
        modified_photo = ImageTk.PhotoImage(modified_pil)
    
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
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age Coefficient:", 0, 10, 0.01, 6)
        self.global_scaling_maps_slider = self.create_slider(self.frame_sliders, "All Maps:", 0, 2, 0.01, 0.1)
        self.global_scaling_masks_slider = self.create_slider(self.frame_sliders, "All Masks:", 0, 2, 0.01, 0.1)
        self.scaling_map1_slider = self.create_slider(self.frame_sliders, "Scale Cm:", 0, 2, 0.01, 1)
        self.scaling_map2_slider = self.create_slider(self.frame_sliders, "Scale Ch:", 0, 2, 0.01, 1)
        self.scaling_map3_slider = self.create_slider(self.frame_sliders, "Scale Bm:", 0, 2, 0.01, 1)
        self.scaling_map4_slider = self.create_slider(self.frame_sliders, "Scale Bh:", 0, 2, 0.01, 1)
        self.scaling_map5_slider = self.create_slider(self.frame_sliders, "Scale T:", 0, 2, 0.01, 1)
        self.scaling_mask1_slider = self.create_slider(self.frame_sliders, "Scale Melanin Mask:", 0, 2, 0.01, 0.1)
        self.scaling_mask2_slider = self.create_slider(self.frame_sliders, "Scale oxy_aged Mask:", 0, 2, 0.01, 0.1)

        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.global_scaling_maps_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.global_scaling_masks_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map1_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map2_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map3_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map4_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_map5_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_mask1_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.scaling_mask2_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.update_plot(
            age_coef=self.age_coef_slider.get(),
            global_scaling_maps=self.global_scaling_maps_slider.get(),
            global_scaling_masks=self.global_scaling_masks_slider.get(),
            scale_c_m=self.scaling_map1_slider.get(),
            scale_c_h=self.scaling_map2_slider.get(),
            scale_b_m=self.scaling_map3_slider.get(),
            scale_b_h=self.scaling_map4_slider.get(),
            scale_t=self.scaling_map5_slider.get(),
            scale_mask_mel=self.scaling_mask1_slider.get(),
            scale_mask_oxy_aged=self.scaling_mask2_slider.get()
        )

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = SkinParameterAdjustmentApp()
    app.run()
