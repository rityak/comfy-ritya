import torch
import numpy as np
import io
from PIL import Image
from collections import OrderedDict
import re

from comfy.model_patcher import ModelPatcher

from .model_block_patterns import SD15_UNET_BLOCKS, SDXL_UNET_BLOCKS

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib/scipy not found. Please install it with 'pip install matplotlib scipy'")
    plt = None

# Presets for grouping UNet blocks based on model type.
GROUPING_PRESETS = {"SD1.5 UNet": SD15_UNET_BLOCKS, "SDXL UNet": SDXL_UNET_BLOCKS}
# A key used to identify the UNet's prefix in the state dictionary.
UNET_ANCHOR_KEY = "input_blocks.0.0.weight"


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def find_unet_prefix(state_dict):
    """
    Finds the common prefix for UNet keys within a state dictionary.
    This is necessary because some models might have a wrapper prefix.
    """
    for key in state_dict.keys():
        if UNET_ANCHOR_KEY in key:
            return key.split(UNET_ANCHOR_KEY)[0]
    return ""


def get_materialized_state_dict(model_input, desc):
    """
    Retrieves the state dictionary from a model input.
    Handles both ModelPatcher objects and raw state dictionaries.
    """
    unwrapped_input = model_input[0] if isinstance(model_input, tuple) else model_input

    if isinstance(unwrapped_input, ModelPatcher):
        model_patcher = unwrapped_input
        print(f"Plotter ({desc}): Input is a ModelPatcher. Materializing with .patch_model()...")
        # Patch the model to get its full state dictionary.
        patched_model_object = model_patcher.patch_model()
        state_dict = patched_model_object.state_dict()
        return state_dict

    elif isinstance(unwrapped_input, dict):
        print(f"Plotter ({desc}): Input is a raw state_dict. Using directly.")
        return unwrapped_input

    else:
        raise TypeError(f"Unsupported model input type: {type(unwrapped_input)}")


# --- L2 NORM PLOTTER NODE ---

class L2NormPlotter:
    """
    A ComfyUI node for plotting L2 norms of UNet blocks.
    It can compare L2 norms between two models or visualize a single model's norms.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Defines the input types for the ComfyUI node.
        return {
            "required": {
                "model": ("MODEL",),
                "grouping_preset": (list(GROUPING_PRESETS.keys()),),
                "y_axis_scale": (["Linear", "Logarithmic", "Normalized Difference (%)"],),
                "y_axis_fit": (["Auto-Fit", "Full Range"], {"default": "Auto-Fit"}),
                "plot_title": ("STRING", {"default": "UNet Block L2 Norms"}),
                "font_size": ("INT", {"default": 12, "min": 5, "max": 30}),
                "figure_width": ("INT", {"default": 20, "min": 5, "max": 50}),
                "figure_height": ("INT", {"default": 8, "min": 3, "max": 30}),
            },
            "optional": {"comparison_model": ("MODEL",)}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_norms"
    CATEGORY = "comfy-ritya/model_analysis"

    def get_l2_norms(self, model_input, block_patterns, desc):
        """
        Calculates the L2 norms for each specified UNet block within a model's state dictionary.
        """
        state_dict = get_materialized_state_dict(model_input, desc)
        unet_prefix = find_unet_prefix(state_dict)
        # Initialize sums of squares for each block, with a small epsilon for stability.
        block_sum_sq = {name: 1e-12 for name in block_patterns}

        print(f"L2NormPlotter ({desc}): Calculating L2 norms for {len(state_dict)} tensors...")
        # Iterate through the state dictionary to accumulate squared values for each UNet block.
        for key, tensor in state_dict.items():
            if not key.startswith(unet_prefix): continue  # Skip non-UNet parameters

            sub_key = key[len(unet_prefix):]  # Get the key without the UNet prefix
            for block_name, pattern in block_patterns.items():
                if re.match(pattern, sub_key):
                    # Sum the squared values of the tensor and add to the block's total.
                    block_sum_sq[block_name] += torch.sum(tensor.cpu().to(torch.float32).pow(2)).item()
                    break  # Move to the next key once a block match is found

        # Calculate the L2 norm (square root of sum of squares) for each block.
        return OrderedDict((name, np.sqrt(s)) for name, s in block_sum_sq.items())

    def plot_norms(self, model, grouping_preset, y_axis_scale, y_axis_fit, plot_title, font_size,
                   figure_width, figure_height, comparison_model=None):
        """
        Generates and returns a plot of L2 norms for UNet blocks.
        Supports linear, logarithmic, or normalized difference scales.
        """
        if plt is None: raise ImportError("Matplotlib required: 'pip install matplotlib scipy'")

        block_patterns = GROUPING_PRESETS[grouping_preset]
        # Get L2 norms for the primary model.
        norms_a = self.get_l2_norms(model, block_patterns, desc="Model A")

        norms_b = None
        has_comparison = comparison_model is not None
        if has_comparison:
            # Get L2 norms for the comparison model, if provided.
            norms_b = self.get_l2_norms(comparison_model, block_patterns, desc="Model B")
        elif y_axis_scale == "Normalized Difference (%)":
            # Error if normalized difference is requested without a comparison model.
            raise ValueError("Normalized Difference scale requires a comparison model.")

        # Configure plot aesthetics (colors, background).
        plt.style.use('dark_background')
        bg_color, text_color, grid_color = '#2e3440', '#d8dee9', '#4c566a'
        color_a, color_b = '#88c0d0', '#bf616a'
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=100)
        fig.patch.set_facecolor(bg_color);
        ax.set_facecolor(bg_color)

        labels = list(norms_a.keys());
        x = np.arange(len(labels))

        all_values = []
        y_label = "L2 Norm"
        # Plotting logic based on the selected Y-axis scale.
        if y_axis_scale == "Normalized Difference (%)":
            y_label = "Difference from Model B (%)"
            values_b = np.array(list(norms_b.values()))
            # Calculate normalized difference in percentage, handle division by zero.
            values_a = (np.array(list(norms_a.values())) / np.where(values_b == 0, 1e-12, values_b) - 1) * 100
            all_values = list(values_a)
            ax.plot(x, values_a, marker='o', linestyle='-', color=color_a, zorder=10)
            ax.axhline(0, color=grid_color, linestyle='--',
                       linewidth=1)  # Add a horizontal line at 0 for difference plots.
            # Add text labels for values on the plot.
            for i, val in enumerate(values_a): ax.text(i, val, f'{val:+.2f}%', ha='center',
                                                       va='bottom' if val >= 0 else 'top', fontsize=font_size * 0.8,
                                                       color=text_color, zorder=11)
        else:
            values_a = list(norms_a.values())
            all_values.extend(values_a)
            ax.plot(x, values_a, marker='o', linestyle='-', color=color_a, label='Model A (Primary)', zorder=10)
            # Add text labels for values.
            for i, val in enumerate(values_a): ax.text(i, val, f'{val:.4f}', ha='center', va='bottom',
                                                       fontsize=font_size * 0.8, color=text_color, zorder=11)
            if norms_b:
                values_b = list(norms_b.values())
                all_values.extend(values_b)
                ax.plot(x, values_b, marker='x', linestyle='--', color=color_b, label='Model B (Comparison)', zorder=10)
                # Add text labels for values.
                for i, val in enumerate(values_b): ax.text(i, val, f'{val:.4f}', ha='center', va='top',
                                                           fontsize=font_size * 0.8, color=text_color, zorder=11)
                ax.legend(facecolor=bg_color, edgecolor=grid_color, fontsize=font_size)

        # Apply logarithmic scale if chosen.
        if y_axis_scale == "Logarithmic":
            ax.set_yscale('log');
            y_label += " (Log Scale)"

        # Adjust Y-axis limits based on the auto-fit option.
        if y_axis_fit == 'Auto-Fit' and all_values:
            min_val = min(all_values);
            max_val = max(all_values)
            padding = (max_val - min_val) * 0.10
            # For difference plots, ensure the zero line is always visible.
            final_min = min(min_val - padding,
                            -padding) if y_axis_scale == "Normalized Difference (%)" else min_val - padding
            final_max = max(max_val + padding,
                            padding) if y_axis_scale == "Normalized Difference (%)" else max_val + padding
            ax.set_ylim(final_min, final_max)

        # Set plot title, axis labels, and tick marks.
        ax.set_title(plot_title, fontsize=font_size * 1.5, color=text_color, loc='left', pad=20)
        ax.set_ylabel(y_label, fontsize=font_size, color=text_color)
        ax.set_xticks(x);
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=font_size, color=text_color)
        ax.tick_params(colors=text_color);
        ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=1)
        # Customize the appearance of the plot's frame (spines).
        for spine in ax.spines.values(): spine.set_color(grid_color)

        fig.tight_layout()  # Adjust plot parameters for a tight layout.
        buf = io.BytesIO();
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0);
        image = Image.open(buf);
        plt.close(fig)  # Close the plot to free memory.
        return (pil_to_tensor(image),)


# --- COSINE SIMILARITY PLOTTER NODE ---

class CosineSimilarityPlotter:
    """
    A ComfyUI node for plotting cosine similarity between corresponding UNet blocks of two models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Defines the input types for the ComfyUI node.
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "grouping_preset": (list(GROUPING_PRESETS.keys()),),
                "y_axis_fit": (["Auto-Fit", "Full Range (-1 to 1)"], {"default": "Auto-Fit"}),
                "plot_title": ("STRING", {"default": "UNet Block Cosine Similarity"}),
                "font_size": ("INT", {"default": 12, "min": 5, "max": 30}),
                "figure_width": ("INT", {"default": 20, "min": 5, "max": 50}),
                "figure_height": ("INT", {"default": 8, "min": 3, "max": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_similarity"
    CATEGORY = "comfy-ritya/model_analysis"

    def get_block_cosine_similarities(self, model_a_input, model_b_input, block_patterns):
        """
        Calculates the cosine similarity for each corresponding UNet block between two models.
        """
        sd_a = get_materialized_state_dict(model_a_input, "Model A")
        sd_b = get_materialized_state_dict(model_b_input, "Model B")
        unet_prefix = find_unet_prefix(sd_a)

        # Initialize metrics (dot product, squared norms) for each block.
        block_metrics = {name: {'dot_prod': 0.0, 'norm_a_sq': 1e-12, 'norm_b_sq': 1e-12} for name in block_patterns}

        print(f"CosineSimilarityPlotter: Calculating similarities for {len(sd_a)} tensors...")
        # Iterate through keys in state dict A to find corresponding keys in B and calculate metrics.
        for key, tensor_a in sd_a.items():
            if not key.startswith(unet_prefix) or key not in sd_b: continue  # Skip non-UNet or missing keys.

            tensor_b = sd_b[key]
            sub_key = key[len(unet_prefix):]

            for block_name, pattern in block_patterns.items():
                if re.match(pattern, sub_key):
                    tensor_a_f32 = tensor_a.cpu().to(torch.float32)
                    tensor_b_f32 = tensor_b.cpu().to(torch.float32)
                    # Accumulate dot product and squared norms for cosine similarity calculation.
                    block_metrics[block_name]['dot_prod'] += torch.sum(tensor_a_f32 * tensor_b_f32).item()
                    block_metrics[block_name]['norm_a_sq'] += torch.sum(tensor_a_f32.pow(2)).item()
                    block_metrics[block_name]['norm_b_sq'] += torch.sum(tensor_b_f32.pow(2)).item()
                    break  # Move to the next key once a block match is found.

        # Calculate the cosine similarity for each block.
        similarities = OrderedDict()
        for name, metrics in block_metrics.items():
            cos_sim = metrics['dot_prod'] / (np.sqrt(metrics['norm_a_sq']) * np.sqrt(metrics['norm_b_sq']))
            similarities[name] = cos_sim
        return similarities

    def plot_similarity(self, model_a, model_b, grouping_preset, y_axis_fit, plot_title, font_size,
                        figure_width, figure_height):
        """
        Generates and returns a plot of cosine similarities between UNet blocks of two models.
        """
        if plt is None: raise ImportError("Matplotlib required: 'pip install matplotlib scipy'")

        block_patterns = GROUPING_PRESETS[grouping_preset]
        # Get cosine similarities between the two models.
        similarities = self.get_block_cosine_similarities(model_a, model_b, block_patterns)

        # Configure plot aesthetics (colors, background).
        plt.style.use('dark_background')
        bg_color, text_color, grid_color = '#2e3440', '#d8dee9', '#4c566a'
        color_a = '#88c0d0'

        fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=100)
        fig.patch.set_facecolor(bg_color);
        ax.set_facecolor(bg_color)

        labels = list(similarities.keys())
        values = list(similarities.values())
        x = np.arange(len(labels))

        ax.plot(x, values, marker='o', linestyle='-', color=color_a, zorder=10)
        # Add horizontal lines for reference cosine similarity values (1, 0, -1).
        ax.axhline(1.0, color=grid_color, linestyle='--', linewidth=1);
        ax.axhline(0.0, color=grid_color, linestyle='-', linewidth=1);
        ax.axhline(-1.0, color=grid_color, linestyle='--', linewidth=1)

        # Add text labels for values on the plot.
        for i, val in enumerate(values): ax.text(i, val, f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                                                 fontsize=font_size * 0.8, color=text_color, zorder=11)

        # Adjust Y-axis limits based on the auto-fit option.
        if y_axis_fit == 'Auto-Fit' and values:
            min_val = min(values);
            max_val = max(values)
            padding = (max_val - min_val) * 0.10
            ax.set_ylim(max(min_val - padding, -1.1), min(max_val + padding, 1.1))
        else:
            ax.set_ylim(-1.1, 1.1)  # Set full range for Y-axis.

        # Set plot title, axis labels, and tick marks.
        ax.set_title(plot_title, fontsize=font_size * 1.5, color=text_color, loc='left', pad=20)
        ax.set_ylabel("Cosine Similarity", fontsize=font_size, color=text_color)
        ax.set_xticks(x);
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=font_size, color=text_color)
        ax.tick_params(colors=text_color);
        ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=1)
        # Customize the appearance of the plot's frame (spines).
        for spine in ax.spines.values(): spine.set_color(grid_color)

        fig.tight_layout()  # Adjust plot parameters for a tight layout.
        buf = io.BytesIO();
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0);
        image = Image.open(buf);
        plt.close(fig)  # Close the plot to free memory.
        return (pil_to_tensor(image),)