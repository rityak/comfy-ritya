import math
import logging

class SDXLMergeWeightedBlocks:
    @classmethod
    def INPUT_TYPES(cls):
        # Defines the input types and their properties for the node.
        return {
            "required": {
                "num_models": ("INT", {"default": 2, "min": 2, "max": 5, "step": 1}),
                "model_clip_weight1": ("mocw",),
                "model_clip_weight2": ("mocw",),
            },
            "optional": {
                "model_clip_weight3": ("mocw",),
                "model_clip_weight4": ("mocw",),
                "model_clip_weight5": ("mocw",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "merge"
    CATEGORY = "comfy-ritya/model_merging"

    def merge(self, num_models, model_clip_weight1, model_clip_weight2,
              model_clip_weight3=None, model_clip_weight4=None, model_clip_weight5=None):
        # Collects input configurations.
        configs = [model_clip_weight1, model_clip_weight2, model_clip_weight3,
                   model_clip_weight4, model_clip_weight5][:num_models]
        if len(configs) != num_models:
            raise ValueError("Number of configurations must match num_models")

        # Extracts models, CLIPs, and weights from configurations.
        models = [config["model"] for config in configs]
        clips = [config["clip"] for config in configs]
        weights_list = [config["weights"] for config in configs]

        # Normalizes weights for each block.
        normalized_weights_list = self.normalize_alphas(weights_list, num_models)

        # Groups model and clip keys by blocks.
        model_blocks = self.get_model_blocks(models[0])
        clip_blocks = self.get_clip_blocks(clips[0])

        # Merges the models.
        merged_model = models[0].clone()
        for block, keys in model_blocks.items():
            if keys:
                for i, model in enumerate(models):
                    alphas = normalized_weights_list[i][block if block in normalized_weights_list[i] else "global"]
                    kp = {k: model.get_key_patches("diffusion_model.")[k] for k in keys}
                    merged_model.add_patches(kp, strength_patch=alphas, strength_model=1 if i > 0 else 0)

        # Merges the CLIPs.
        merged_clip = clips[0].clone()
        for block, keys in clip_blocks.items():
            if keys:
                for i, clip in enumerate(clips):
                    alphas = normalized_weights_list[i][block if block in normalized_weights_list[i] else "global"]
                    kp = {k: clip.get_key_patches()[k] for k in keys}
                    merged_clip.add_patches(kp, strength_patch=alphas, strength_model=1 if i > 0 else 0)

        return (merged_model, merged_clip)

    def normalize_alphas(self, weights_list, num_models):
        # Normalizes alpha values for each block across multiple models.
        normalized_weights = []
        for block in ["global", "input_blocks", "middle_block", "output_blocks", "clip_l", "clip_g"]:
            # Collects alphas for the current block.
            block_alphas = [weights[block] for weights in weights_list]
            total = sum(block_alphas)

            # Handles zero or negative sum.
            if total <= 0:
                logging.warning(f"Sum of alphas for block {block} <= 0 ({total}). Using uniform distribution.")
                normalized_alphas = [1.0 / num_models] * num_models
            else:
                # Normalizes alphas.
                normalized_alphas = [round(alpha / total, 4) for alpha in block_alphas]
                # Adjusts the first alpha to ensure sum is 1.0.
                if not math.isclose(sum(normalized_alphas), 1.0, rel_tol=1e-5):
                    diff = 1.0 - sum(normalized_alphas)
                    normalized_alphas[0] = round(normalized_alphas[0] + diff, 4)

            # Clamps alphas to [0.0, 1.0] range.
            normalized_alphas = [max(0.0, min(1.0, alpha)) for alpha in normalized_alphas]

            # Re-checks sum after adjustment.
            if not math.isclose(sum(normalized_alphas), 1.0, rel_tol=1e-5):
                logging.warning(f"Sum of alphas for block {block} after normalization: {sum(normalized_alphas)}. Applying re-normalization.")
                total = sum(normalized_alphas)
                if total > 0:
                    normalized_alphas = [round(alpha / total, 4) for alpha in block_alphas]
                    normalized_alphas[0] = round(normalized_alphas[0] + (1.0 - sum(normalized_alphas)), 4)

            # Updates weights for each model.
            for i, weights in enumerate(weights_list):
                if block not in normalized_weights:
                    normalized_weights.append(weights.copy())
                normalized_weights[i][block] = normalized_alphas[i]

        return normalized_weights

    def get_model_blocks(self, model):
        # Organizes model keys into block categories.
        keys = model.get_key_patches("diffusion_model.").keys()
        blocks = {}
        for key in keys:
            if key.startswith("model.diffusion_model.input_blocks"):
                blocks.setdefault("input_blocks", []).append(key)
            elif key.startswith("model.diffusion_model.middle_block"):
                blocks.setdefault("middle_block", []).append(key)
            elif key.startswith("model.diffusion_model.output_blocks"):
                blocks.setdefault("output_blocks", []).append(key)
            else:
                blocks.setdefault("global", []).append(key)
        return blocks

    def get_clip_blocks(self, clip):
        # Organizes CLIP keys into block categories.
        keys = clip.get_key_patches().keys()
        blocks = {}
        for key in keys:
            if key.startswith("conditioner.embedders.0"):
                blocks.setdefault("clip_l", []).append(key)
            elif key.startswith("conditioner.embedders.1"):
                blocks.setdefault("clip_g", []).append(key)
            else:
                blocks.setdefault("global", []).append(key)
        return blocks