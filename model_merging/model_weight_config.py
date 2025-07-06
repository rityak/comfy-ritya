class SDXLModelWeightConfig:
    @classmethod
    def INPUT_TYPES(cls):
        # Defines the input types and their properties for the node.
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "alpha_global": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_input_blocks": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_middle_block": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_output_blocks": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_l": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_clip_g": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("mocw",)
    RETURN_NAMES = ("model_clip_weight",)
    FUNCTION = "configure"
    CATEGORY = "comfy-ritya/model_merging"

    def configure(self, model, clip, alpha_global, alpha_input_blocks, alpha_middle_block,
                  alpha_output_blocks, alpha_clip_l, alpha_clip_g):
        # Configures and bundles the model, clip, and their respective block weights.
        weights = {
            "global": alpha_global,
            "input_blocks": alpha_input_blocks,
            "middle_block": alpha_middle_block,
            "output_blocks": alpha_output_blocks,
            "clip_l": alpha_clip_l,
            "clip_g": alpha_clip_g,
        }
        return ({"model": model, "clip": clip, "weights": weights},)