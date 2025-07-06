# clip_comparison.py - CLIP Comparison Node for ComfyUI

import torch
import numpy as np
import io
from PIL import Image
from collections import OrderedDict
import re

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not found. Please install it with 'pip install matplotlib'")
    plt = None

# --- CLIP BLOCK PATTERNS ---

CLIP_L_BLOCKS = OrderedDict({
    'EMB': r'clip_l\.transformer\.text_model\.embeddings\..*',
    'L00': r'clip_l\.transformer\.text_model\.encoder\.layers\.0\..*',
    'L01': r'clip_l\.transformer\.text_model\.encoder\.layers\.1\..*',
    'L02': r'clip_l\.transformer\.text_model\.encoder\.layers\.2\..*',
    'L03': r'clip_l\.transformer\.text_model\.encoder\.layers\.3\..*',
    'L04': r'clip_l\.transformer\.text_model\.encoder\.layers\.4\..*',
    'L05': r'clip_l\.transformer\.text_model\.encoder\.layers\.5\..*',
    'L06': r'clip_l\.transformer\.text_model\.encoder\.layers\.6\..*',
    'L07': r'clip_l\.transformer\.text_model\.encoder\.layers\.7\..*',
    'L08': r'clip_l\.transformer\.text_model\.encoder\.layers\.8\..*',
    'L09': r'clip_l\.transformer\.text_model\.encoder\.layers\.9\..*',
    'L10': r'clip_l\.transformer\.text_model\.encoder\.layers\.10\..*',
    'L11': r'clip_l\.transformer\.text_model\.encoder\.layers\.11\..*',
    'FINAL': r'clip_l\.transformer\.text_model\.final_layer_norm\..*',
})

CLIP_G_BLOCKS = OrderedDict({
    'EMB_G': r'clip_g\.transformer\.text_model\.embeddings\..*',
    'G00': r'clip_g\.transformer\.text_model\.encoder\.layers\.0\..*',
    'G01': r'clip_g\.transformer\.text_model\.encoder\.layers\.1\..*',
    'G02': r'clip_g\.transformer\.text_model\.encoder\.layers\.2\..*',
    'G03': r'clip_g\.transformer\.text_model\.encoder\.layers\.3\..*',
    'G04': r'clip_g\.transformer\.text_model\.encoder\.layers\.4\..*',
    'G05': r'clip_g\.transformer\.text_model\.encoder\.layers\.5\..*',
    'G06': r'clip_g\.transformer\.text_model\.encoder\.layers\.6\..*',
    'G07': r'clip_g\.transformer\.text_model\.encoder\.layers\.7\..*',
    'G08': r'clip_g\.transformer\.text_model\.encoder\.layers\.8\..*',
    'G09': r'clip_g\.transformer\.text_model\.encoder\.layers\.9\..*',
    'G10': r'clip_g\.transformer\.text_model\.encoder\.layers\.10\..*',
    'G11': r'clip_g\.transformer\.text_model\.encoder\.layers\.11\..*',
    'G12': r'clip_g\.transformer\.text_model\.encoder\.layers\.12\..*',
    'G13': r'clip_g\.transformer\.text_model\.encoder\.layers\.13\..*',
    'G14': r'clip_g\.transformer\.text_model\.encoder\.layers\.14\..*',
    'G15': r'clip_g\.transformer\.text_model\.encoder\.layers\.15\..*',
    'G16': r'clip_g\.transformer\.text_model\.encoder\.layers\.16\..*',
    'G17': r'clip_g\.transformer\.text_model\.encoder\.layers\.17\..*',
    'G18': r'clip_g\.transformer\.text_model\.encoder\.layers\.18\..*',
    'G19': r'clip_g\.transformer\.text_model\.encoder\.layers\.19\..*',
    'G20': r'clip_g\.transformer\.text_model\.encoder\.layers\.20\..*',
    'G21': r'clip_g\.transformer\.text_model\.encoder\.layers\.21\..*',
    'G22': r'clip_g\.transformer\.text_model\.encoder\.layers\.22\..*',
    'G23': r'clip_g\.transformer\.text_model\.encoder\.layers\.23\..*',
    'FINAL_G': r'clip_g\.transformer\.text_model\.final_layer_norm\..*',
})

GROUPING_PRESETS = {
    "CLIP-L": CLIP_L_BLOCKS,
    "CLIP-G": CLIP_G_BLOCKS,
}

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_clip_state_dict(clip_input, desc):
    if hasattr(clip_input, 'get_sd'):
        print(f"CLIP Comparison ({desc}): Using get_sd() to get state_dict.")
        return clip_input.get_sd()
    else:
        raise AttributeError(f"CLIP input does not have get_sd method: {type(clip_input)}")

# --- PLOT CLIP L2 NORMS NODE ---

class PlotClipL2Norms:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "grouping_preset": (list(GROUPING_PRESETS.keys()),),
                "y_axis_scale": (["Linear", "Logarithmic", "Normalized Difference (%)"],),
                "y_axis_fit": (["Auto-Fit", "Full Range"], {"default": "Auto-Fit"}),
                "plot_title": ("STRING", {"default": "CLIP Block L2 Norms"}),
                "font_size": ("INT", {"default": 12, "min": 5, "max": 30}),
                "figure_width": ("INT", {"default": 20, "min": 5, "max": 50}),
                "figure_height": ("INT", {"default": 8, "min": 3, "max": 30}),
            },
            "optional": {"comparison_clip": ("CLIP",)}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_norms"
    CATEGORY = "comfy-ritya/model_analysis"

    def get_l2_norms(self, clip_input, block_patterns, desc):
        state_dict = get_clip_state_dict(clip_input, desc)
        block_sum_sq = {name: 1e-12 for name in block_patterns}
        
        for key, tensor in state_dict.items():
            for block_name, pattern in block_patterns.items():
                if re.match(pattern, key):
                    block_sum_sq[block_name] += torch.sum(tensor.cpu().to(torch.float32).pow(2)).item()
                    break
        
        return OrderedDict((name, np.sqrt(s)) for name, s in block_sum_sq.items())

    def plot_norms(self, clip, grouping_preset, y_axis_scale, y_axis_fit, plot_title, font_size,
                   figure_width, figure_height, comparison_clip=None):
        if plt is None:
            raise ImportError("Matplotlib required: 'pip install matplotlib'")
        
        block_patterns = GROUPING_PRESETS[grouping_preset]
        norms_a = self.get_l2_norms(clip, block_patterns, desc="Clip A")
        
        norms_b = None
        has_comparison = comparison_clip is not None
        if has_comparison:
            norms_b = self.get_l2_norms(comparison_clip, block_patterns, desc="Clip B")
        elif y_axis_scale == "Normalized Difference (%)":
            raise ValueError("Normalized Difference scale requires a comparison clip.")

        plt.style.use('dark_background')
        bg_color, text_color, grid_color = '#2e3440', '#d8dee9', '#4c566a'
        color_a, color_b = '#88c0d0', '#bf616a'
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        labels = list(norms_a.keys())
        x = np.arange(len(labels))
        
        all_values = []
        y_label = "L2 Norm"
        if y_axis_scale == "Normalized Difference (%)":
            y_label = "Difference from Clip B (%)"
            values_b = np.array(list(norms_b.values()))
            values_a = (np.array(list(norms_a.values())) / np.where(values_b == 0, 1e-12, values_b) - 1) * 100
            all_values = list(values_a)
            ax.plot(x, values_a, marker='o', linestyle='-', color=color_a, zorder=10)
            ax.axhline(0, color=grid_color, linestyle='--', linewidth=1)
            for i, val in enumerate(values_a):
                ax.text(i, val, f'{val:+.2f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=font_size*0.8, color=text_color, zorder=11)
        else:
            values_a = list(norms_a.values())
            all_values.extend(values_a)
            ax.plot(x, values_a, marker='o', linestyle='-', color=color_a, label='Clip A (Primary)', zorder=10)
            for i, val in enumerate(values_a):
                ax.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=font_size*0.8, color=text_color, zorder=11)
            if norms_b:
                values_b = list(norms_b.values())
                all_values.extend(values_b)
                ax.plot(x, values_b, marker='x', linestyle='--', color=color_b, label='Clip B (Comparison)', zorder=10)
                for i, val in enumerate(values_b):
                    ax.text(i, val, f'{val:.4f}', ha='center', va='top', fontsize=font_size*0.8, color=text_color, zorder=11)
                ax.legend(facecolor=bg_color, edgecolor=grid_color, fontsize=font_size)

        if y_axis_scale == "Logarithmic":
            ax.set_yscale('log')
            y_label += " (Log Scale)"

        if y_axis_fit == 'Auto-Fit' and all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            padding = (max_val - min_val) * 0.10
            final_min = min(min_val - padding, -padding) if y_axis_scale == "Normalized Difference (%)" else min_val - padding
            final_max = max(max_val + padding, padding) if y_axis_scale == "Normalized Difference (%)" else max_val + padding
            ax.set_ylim(final_min, final_max)

        ax.set_title(plot_title, fontsize=font_size*1.5, color=text_color, loc='left', pad=20)
        ax.set_ylabel(y_label, fontsize=font_size, color=text_color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=font_size, color=text_color)
        ax.tick_params(colors=text_color)
        ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=1)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return (pil_to_tensor(image),)

# --- PLOT CLIP COSINE SIMILARITY NODE ---

class PlotClipCosineSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("CLIP",),
                "clip_b": ("CLIP",),
                "grouping_preset": (list(GROUPING_PRESETS.keys()),),
                "y_axis_fit": (["Auto-Fit", "Full Range (-1 to 1)"], {"default": "Auto-Fit"}),
                "plot_title": ("STRING", {"default": "CLIP Block Cosine Similarity"}),
                "font_size": ("INT", {"default": 12, "min": 5, "max": 30}),  # Fixed syntax error here
                "figure_width": ("INT", {"default": 20, "min": 5, "max": 50}),
                "figure_height": ("INT", {"default": 8, "min": 3, "max": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_similarity"
    CATEGORY = "comfy-ritya/model_analysis"

    def get_block_cosine_similarities(self, clip_a_input, clip_b_input, block_patterns):
        sd_a = get_clip_state_dict(clip_a_input, "Clip A")
        sd_b = get_clip_state_dict(clip_b_input, "Clip B")
        
        block_metrics = {name: {'dot_prod': 0.0, 'norm_a_sq': 1e-12, 'norm_b_sq': 1e-12} for name in block_patterns}
        
        for key in sd_a:
            if key not in sd_b:
                continue
            tensor_a = sd_a[key]
            tensor_b = sd_b[key]
            for block_name, pattern in block_patterns.items():
                if re.match(pattern, key):
                    tensor_a_f32 = tensor_a.cpu().to(torch.float32)
                    tensor_b_f32 = tensor_b.cpu().to(torch.float32)
                    block_metrics[block_name]['dot_prod'] += torch.sum(tensor_a_f32 * tensor_b_f32).item()
                    block_metrics[block_name]['norm_a_sq'] += torch.sum(tensor_a_f32.pow(2)).item()
                    block_metrics[block_name]['norm_b_sq'] += torch.sum(tensor_b_f32.pow(2)).item()
                    break
        
        similarities = OrderedDict()
        for name, metrics in block_metrics.items():
            cos_sim = metrics['dot_prod'] / (np.sqrt(metrics['norm_a_sq']) * np.sqrt(metrics['norm_b_sq']))
            similarities[name] = cos_sim
        return similarities

    def plot_similarity(self, clip_a, clip_b, grouping_preset, y_axis_fit, plot_title, font_size,
                        figure_width, figure_height):
        if plt is None:
            raise ImportError("Matplotlib required: 'pip install matplotlib'")

        block_patterns = GROUPING_PRESETS[grouping_preset]
        similarities = self.get_block_cosine_similarities(clip_a, clip_b, block_patterns)

        plt.style.use('dark_background')
        bg_color, text_color, grid_color = '#2e3440', '#d8dee9', '#4c566a'
        color_a = '#88c0d0'
        
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        labels = list(similarities.keys())
        values = list(similarities.values())
        x = np.arange(len(labels))
        
        ax.plot(x, values, marker='o', linestyle='-', color=color_a, zorder=10)
        ax.axhline(1.0, color=grid_color, linestyle='--', linewidth=1)
        ax.axhline(0.0, color=grid_color, linestyle='-', linewidth=1)
        ax.axhline(-1.0, color=grid_color, linestyle='--', linewidth=1)

        for i, val in enumerate(values):
            ax.text(i, val, f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=font_size*0.8, color=text_color, zorder=11)

        if y_axis_fit == 'Auto-Fit' and values:
            min_val = min(values)
            max_val = max(values)
            padding = (max_val - min_val) * 0.10
            ax.set_ylim(max(min_val - padding, -1.1), min(max_val + padding, 1.1))
        else:
            ax.set_ylim(-1.1, 1.1)

        ax.set_title(plot_title, fontsize=font_size*1.5, color=text_color, loc='left', pad=20)
        ax.set_ylabel("Cosine Similarity", fontsize=font_size, color=text_color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=font_size, color=text_color)
        ax.tick_params(colors=text_color)
        ax.grid(axis='y', color=grid_color, linestyle=':', linewidth=1)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)
        return (pil_to_tensor(image),)