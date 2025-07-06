from .model_analysis import L2NormPlotter, CosineSimilarityPlotter, PlotClipL2Norms, PlotClipCosineSimilarity
from .model_merging import SDXLMergeWeightedBlocks, SDXLModelWeightConfig, SDXLMergeKarcher

NODE_CLASS_MAPPINGS = {
  "SDXLKarcherMerge": SDXLMergeKarcher,
  "SDXLMergeWeightedBlocks": SDXLMergeWeightedBlocks,
  "SDXLModelWeightConfig": SDXLModelWeightConfig,
  "L2NormPlotter": L2NormPlotter,
  "CosineSimilarityPlotter": CosineSimilarityPlotter,
  "PlotClipL2Norms": PlotClipL2Norms,
  "PlotClipCosineSimilarity": PlotClipCosineSimilarity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "SDXLKarcherMerge": "⚡️ SDXL Karcher Merge",
  "SDXLModelWeightConfig": "⚡️ SDXL Model Weight Config",
  "SDXLMergeWeightedBlocks": "⚡️ SDXL Merge Weighted Blocks",
  "L2NormPlotter": "⚡️ Plot Model L2 Norms",
  "CosineSimilarityPlotter": "⚡️ Plot Model Cosine Similarity",
  "PlotClipL2Norms": "⚡️ Plot CLIP L2 Norms",
  "PlotClipCosineSimilarity": "⚡️ Plot CLIP Cosine Similarity",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]