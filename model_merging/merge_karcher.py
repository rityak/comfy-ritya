import torch
import math
import logging
import torch.nn.functional as F


class SDXLMergeKarcher:
    @classmethod
    def INPUT_TYPES(cls):
        # Defines the input types and their properties for the node.
        return {
            "required": {
                "num_models": ("INT", {"default": 2, "min": 2, "max": 5, "step": 1}),
                "model_clip_weight1": ("mocw",),
                "model_clip_weight2": ("mocw",),
                "karcher_iter": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "karcher_tol": ("FLOAT", {"default": 3e-8, "min": 1e-10, "max": 1e-3, "step": 1e-9}),
            },
            "optional": {
                "model_clip_weight3": ("mocw",),
                "model_clip_weight4": ("mocw",),
                "model_clip_weight5": ("mocw",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "merge"
    CATEGORY = "advanced/model_merging"

    def resize_tensors(self, tensors):
        # Resizes a list of tensors to have the same maximum height and width.
        if not tensors or len(tensors[0].shape) < 2:
            return tensors

        max_h, max_w = 0, 0
        for t in tensors:
            h, w = t.shape[-2], t.shape[-1]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        resized_tensors = []
        for t in tensors:
            h, w = t.shape[-2], t.shape[-1]
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h))
            resized_tensors.append(t)
        return resized_tensors

    def extract_tensor_from_patch(self, patch):
        # Extracts a torch.Tensor from a given patch, which can be a tensor or a tuple containing a tensor.
        if isinstance(patch, torch.Tensor):
            return patch
        elif isinstance(patch, tuple) and len(patch) > 0 and isinstance(patch[0], torch.Tensor):
            return patch[0]
        else:
            raise ValueError(f"Invalid patch format: expected tensor or tuple with tensor, got {type(patch)}")

    def preserve_patch_format(self, original_patch, merged_tensor):
        # Preserves the original patch format after merging, typically for tuples.
        if isinstance(original_patch, torch.Tensor):
            return merged_tensor
        elif isinstance(original_patch, tuple) and len(original_patch) > 1:
            return (merged_tensor,) + original_patch[1:]
        else:
            return merged_tensor

    def merge(self, num_models, model_clip_weight1, model_clip_weight2,
              karcher_iter, karcher_tol,
              model_clip_weight3=None, model_clip_weight4=None, model_clip_weight5=None):
        # Merges multiple models and clips using Karcher mean.
        configs = [model_clip_weight1, model_clip_weight2, model_clip_weight3,
                   model_clip_weight4, model_clip_weight5][:num_models]
        if len(configs) != num_models:
            raise ValueError("Number of configurations must match num_models")

        models = [config["model"] for config in configs]
        clips = [config["clip"] for config in configs]
        weights_list = [config["weights"] for config in configs]

        # Normalizes the weights for each model.
        normalized_weights_list = self.normalize_alphas(weights_list, num_models)

        model_keys = models[0].get_key_patches("diffusion_model.").keys()
        clip_keys = clips[0].get_key_patches().keys()

        merged_model = models[0].clone()
        for key in model_keys:
            block = self.get_block_name(key)
            alphas = [weights[block if block in weights else "global"] for weights in normalized_weights_list]

            patches_list = [model.get_key_patches("diffusion_model.")[key] for model in models]
            if not patches_list or not all(len(p) == len(patches_list[0]) for p in patches_list):
                logging.warning(f"Skipped key {key}: inconsistent number of patches")
                continue

            num_patches = len(patches_list[0])
            merged_patches = []

            try:
                first_tensor = self.extract_tensor_from_patch(patches_list[0][0])
                initial_dtype = first_tensor.dtype
                initial_device = first_tensor.device
            except ValueError as e:
                logging.warning(f"Skipped key {key}: {str(e)}")
                continue

            for i in range(num_patches):
                try:
                    tensors = [self.extract_tensor_from_patch(patches[i]).to(device=initial_device, dtype=torch.float)
                               for patches in patches_list]

                    # Checks and aligns tensor shapes.
                    if not all(t.shape == tensors[0].shape for t in tensors):
                        tensors = self.resize_tensors(tensors)
                        if not all(t.shape == tensors[0].shape for t in tensors):
                            logging.warning(f"Skipped patch {i} for key {key}: shapes mismatch after alignment")
                            continue

                    # Performs Karcher mean merging on tensors.
                    merged_tensor = self.karcher_merge_tensors(tensors, alphas, karcher_iter, karcher_tol)
                    if merged_tensor.numel() > 0 and not torch.all(merged_tensor == 0):
                        merged_patch = self.preserve_patch_format(patches_list[0][i], merged_tensor.to(initial_dtype))
                        merged_patches.append(merged_patch)
                    else:
                        logging.warning(f"Patch {i} for key {key}: merged tensor is empty or zero, using first patch")
                        merged_patches.append(patches_list[0][i])
                except ValueError as e:
                    logging.warning(f"Skipped patch {i} for key {key}: {str(e)}")
                    continue

            if merged_patches:
                merged_model.add_patches({key: merged_patches}, strength_patch=1.0, strength_model=0)
            else:
                logging.warning(f"Key {key}: no merged patches, using original")
                merged_model.add_patches({key: patches_list[0]}, strength_patch=1.0, strength_model=0)

        merged_clip = clips[0].clone()
        for key in clip_keys:
            block = self.get_block_name(key)
            alphas = [weights[block if block in weights else "global"] for weights in normalized_weights_list]

            patches_list = [clip.get_key_patches()[key] for clip in clips]
            if not patches_list or not all(len(p) == len(patches_list[0]) for p in patches_list):
                logging.warning(f"Skipped key {key}: inconsistent number of patches")
                continue

            num_patches = len(patches[0])
            merged_patches = []

            try:
                first_tensor = self.extract_tensor_from_patch(patches_list[0][0])
                initial_dtype = first_tensor.dtype
                initial_device = first_tensor.device
            except ValueError as e:
                logging.warning(f"Skipped key {key}: {str(e)}")
                continue

            for i in range(num_patches):
                try:
                    tensors = [self.extract_tensor_from_patch(patches[i]).to(device=initial_device, dtype=torch.float)
                               for patches in patches_list]

                    if not all(t.shape == tensors[0].shape for t in tensors):
                        tensors = self.resize_tensors(tensors)
                        if not all(t.shape == tensors[0].shape for t in tensors):
                            logging.warning(f"Skipped patch {i} for key {key}: shapes mismatch after alignment")
                            continue

                    merged_tensor = self.karcher_merge_tensors(tensors, alphas, karcher_iter, karcher_tol)
                    if merged_tensor.numel() > 0 and not torch.all(merged_tensor == 0):
                        merged_patch = self.preserve_patch_format(patches_list[0][i], merged_tensor.to(initial_dtype))
                        merged_patches.append(merged_patch)
                    else:
                        logging.warning(f"Patch {i} for key {key}: merged tensor is empty or zero, using first patch")
                        merged_patches.append(patches_list[0][i])
                except ValueError as e:
                    logging.warning(f"Skipped patch {i} for key {key}: {str(e)}")
                    continue

            if merged_patches:
                merged_clip.add_patches({key: merged_patches}, strength_patch=1.0, strength_model=0)
            else:
                logging.warning(f"Key {key}: no merged patches, using original")
                merged_clip.add_patches({key: patches_list[0]}, strength_patch=1.0, strength_model=0)

        return (merged_model, merged_clip)

    def normalize_alphas(self, weights_list, num_models):
        # Normalizes a list of weight dictionaries.
        normalized_weights_list = []

        for i, weights in enumerate(weights_list):
            normalized_weights = {}
            for block_name in ["global", "input_blocks", "middle_block", "output_blocks", "clip_l", "clip_g"]:
                block_weights = [w.get(block_name, w["global"]) for w in weights_list]

                total = sum(block_weights)
                if total <= 0:
                    normalized_weights[block_name] = 1.0 / num_models
                else:
                    normalized_weights[block_name] = block_weights[i] / total

            normalized_weights_list.append(normalized_weights)

        return normalized_weights_list

    def karcher_merge_tensors(self, tensors, alphas, max_iter, tol):
        # Computes the Karcher mean of a list of tensors.
        if len(tensors) == 1:
            return tensors[0]

        norms = []
        units = []
        for i, t in enumerate(tensors):
            t_float = t.float()
            n = torch.linalg.norm(t_float).item()
            logging.info(f"Tensor {i} norm: {n:.6f}")
            if n < tol:
                norms.append(0.0)
                units.append(torch.zeros_like(t_float))
            else:
                norms.append(n)
                units.append(t_float / n)

        if all(n < tol for n in norms):
            logging.warning("All tensors have norm near zero. Using first tensor.")
            return tensors[0]

        valid_indices = [i for i, n in enumerate(norms) if n > tol]
        if not valid_indices:
            logging.warning("No tensors with norm > tol. Using first tensor.")
            return tensors[0]

        valid_alphas = [alphas[i] for a, i in zip(alphas, valid_indices)]
        alpha_sum = sum(valid_alphas)
        normalized_alphas = [a / alpha_sum for a in valid_alphas]
        valid_units = [units[i] for i in valid_indices]

        # Initializes the Karcher mean.
        u = torch.zeros_like(valid_units[0], dtype=torch.float)
        for a, ui in zip(normalized_alphas, valid_units):
            u += a * ui
        norm_u = torch.linalg.norm(u).item()
        if norm_u < tol:
            u = valid_units[0].clone()
        else:
            u = u / norm_u

        # Iteratively refines the Karcher mean.
        for _ in range(max_iter):
            T = torch.zeros_like(u, dtype=torch.float)
            for a, ui in zip(normalized_alphas, valid_units):
                dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
                theta = torch.arccos(dot)
                theta_val = theta.item()
                if theta_val < tol:
                    continue
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)

            norm_T = torch.linalg.norm(T)
            if norm_T.item() < tol:
                break

            cos_norm_T = torch.cos(norm_T)
            sin_norm_T = torch.sin(norm_T)
            u = cos_norm_T * u + sin_norm_T * (T / norm_T)

            u_norm = torch.linalg.norm(u)
            if u_norm.item() > tol:
                u = u / u_norm

        s = sum(a * n for a, n in zip(alphas, norms))
        merged = s * u.to(tensors[0].dtype)
        logging.info(f"Merged tensor: norm={s:.6f}, shape={merged.shape}")
        return merged

    def get_block_name(self, key):
        # Determines the block name from a given key.
        if key.startswith("model.diffusion_model.input_blocks"):
            return "input_blocks"
        elif key.startswith("model.diffusion_model.middle_block"):
            return "middle_block"
        elif key.startswith("model.diffusion_model.output_blocks"):
            return "output_blocks"
        elif key.startswith("conditioner.embedders.0"):
            return "clip_l"
        elif key.startswith("conditioner.embedders.1"):
            return "clip_g"
        return "global"