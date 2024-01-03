# ENCODER_PATH = r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\SmallBatchSize\encoder_19_05.h5"
# DECODER_PATH = r"C:\Users\joeli\Dropbox\AE_MC\saved_ml_models\SmallBatchSize\decoder_19_05.h5"

DECODER_PATH = r"C:\Users\joeli\Dropbox\Code\HM_2023\TrainedModels\no_duplicates_75_2_mask\decoder.h5"
ENCODER_PATH = r"C:\Users\joeli\Dropbox\Code\HM_2023\TrainedModels\no_duplicates_75_2_mask\encoder.h5"

# ENCODER_PATH = r"saved_models/encoder.h5"
# DECODER_PATH = r"saved_models/decoder.h5"

ANGER_DIRECTORY = r"C:\Users\joeli\Dropbox\Data\Emotion Data\warped_anger"
NEUTRAL_DIRECTORY =  r"C:\Users\joeli\Dropbox\Data\Emotion Data\warped_neutral"
HAPPY_DIRECTORY = r"C:\Users\joeli\Dropbox\Data\Emotion Data\warped_happy"
SADNESS_DIRECTORY = r"C:\Users\joeli\Dropbox\Data\Emotion Data\warped_sad"

ANGER_DIRECTORY_2 = r"C:\Users\joeli\Dropbox\HM_Oct\faces\angry"
NEUTRAL_DIRECTORY_2 =  r"C:\Users\joeli\Dropbox\HM_Oct\faces\neutral"
HAPPY_DIRECTORY_2 = r"C:\Users\joeli\Dropbox\HM_Oct\faces\happy"
SADNESS_DIRECTORY_2 = r"C:\Users\joeli\Dropbox\HM_Oct\faces\sad"

MASK_DIRECTORY = r"C:\Users\joeli\Dropbox\HM_Oct\mask_combined"
MASK_COMBINED = r"C:\Users\joeli\Dropbox\HM_Oct\mask_combined\combined.png"

MASK_PS = r"C:\Users\joeli\Dropbox\Data\masks\ps_mask_1.png"
MASK_PS_2 = r"C:\Users\joeli\Dropbox\Data\masks\combined_p3.png"

TEST_IMAGE_PATH = r"C:\Users\joeli\Dropbox\Data\models_4k\m53_4k.png"
TEST_IMAGE_PATH_2 = r"C:\Users\joeli\Dropbox\HM_Oct\facescape\1\models_reg\1_neutral.jpg"
UTILITY_PATH = r"C:\Users\joeli\OneDrive\Desktop\m46_XYZ_utility_lin_srgb.1001.png"
UTILITY_PATH_2 = r"C:\Users\joeli\OneDrive\Desktop\m53_XYZ_utility_lin_srgb.1001.png"

MODEL_46 = r"C:\Users\joeli\Dropbox\Data\models_4k\m46_4k.png"
MODEL_53 = r"C:\Users\joeli\Dropbox\Data\models_4k\m53_4k.png"
MODEL_32 = r"C:\Users\joeli\Dropbox\Data\models_4k\m32_4k.png"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_parameter_maps(target, parameter_maps, WIDTH, HEIGHT, name,  vertical_colorbar=True, original=True):
    labels = ["Original", "Cm", "Ch", "Bm", "Bh", "T"]
    if not original:
        labels = ["Recovered", "Cm", "Ch", "Bm", "Bh", "T"]

    # Clear all plots
    plt.close('all')

    # Calculate figure size based on the number of plots
    num_plots = 6
    figsize = (6 * num_plots, 6)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])

    # Ensure that the parameter maps are reshaped
    parameter_maps = parameter_maps.reshape(-1, 5)

    # Find the low and high values
    lows = [np.min(parameter_maps[:, i]) for i in range(5)]
    highs = [np.max(parameter_maps[:, i]) for i in range(5)]

    # Display the target
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(target, aspect='auto')
    ax0.axis('off')

    # Display each parameter map
    for i in range(5):
        ax = fig.add_subplot(gs[0, i + 1])
        im = ax.imshow(parameter_maps[:, i].reshape(WIDTH, HEIGHT), cmap='gray', vmin=lows[i], vmax=highs[i], aspect='auto')
        # low = 0
        # high = 1
        # im = ax.imshow(parameter_maps[:, i].reshape(WIDTH, HEIGHT), cmap='viridis', vmin=low, vmax=high, aspect='auto')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout(pad=0)  # Tighten layout
    plt.savefig(name, dpi=400, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()