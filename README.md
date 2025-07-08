# Comfy-ritya Nodes ✨

A set of utility nodes for **model analysis** and **model merging** in ComfyUI, along with other useful functionalities.

---

## 🚀 Features

### Model Analysis
Gain deep insights into your Diffusion and CLIP models:

* **⚡️ Plot Model L2 Norms**: Visualize the L2 norms of weights across different blocks of your UNet models (SD1.5 & SDXL). Compare how primary models differ from comparison models or baselines, with options for linear, logarithmic, or **normalized difference (%)** scales.
* **⚡️ Plot Model Cosine Similarity**: Quantify the structural similarity between two UNet models by plotting the cosine similarity of their block-wise weights.
* **⚡️ Plot CLIP L2 Norms**: Analyze the L2 norms of weights within the CLIP text encoders (CLIP-L, CLIP-G).
* **⚡️ Plot CLIP Cosine Similarity**: Measure the cosine similarity between two CLIP models at a block level.

---

## 💻 Installation

1.  **Navigate to your `custom_nodes` directory**:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  **Clone this repository**:
    ```bash
    git clone https://github.com/rityak/comfy-ritya.git comfy-ritya-nodes # Replace with your actual repo URL
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r comfy-ritya-nodes/requirements.txt
    ```
4.  **Restart ComfyUI**.

---