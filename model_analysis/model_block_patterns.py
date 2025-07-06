from collections import OrderedDict

# Regular expression patterns for UNet blocks in SD1.5 models.
SD15_UNET_BLOCKS = OrderedDict({
    'IN00': r'input_blocks\.0\..*', 'IN01': r'input_blocks\.1\..*', 'IN02': r'input_blocks\.2\..*',
    'IN03': r'input_blocks\.3\..*', 'IN04': r'input_blocks\.4\..*', 'IN05': r'input_blocks\.5\..*',
    'IN06': r'input_blocks\.6\..*', 'IN07': r'input_blocks\.7\..*', 'IN08': r'input_blocks\.8\..*',
    'IN09': r'input_blocks\.9\..*', 'IN10': r'input_blocks\.10\..*', 'IN11': r'input_blocks\.11\..*',
    'M00': r'middle_block\..*',
    'OUT00': r'output_blocks\.0\..*', 'OUT01': r'output_blocks\.1\..*', 'OUT02': r'output_blocks\.2\..*',
    'OUT03': r'output_blocks\.3\..*', 'OUT04': r'output_blocks\.4\..*', 'OUT05': r'output_blocks\.5\..*',
    'OUT06': r'output_blocks\.6\..*', 'OUT07': r'output_blocks\.7\..*', 'OUT08': r'output_blocks\.8\..*',
    'OUT09': r'output_blocks\.9\..*', 'OUT10': r'output_blocks\.10\..*', 'OUT11': r'output_blocks\.11\..*',
})

# Regular expression patterns for UNet blocks in SDXL models.
SDXL_UNET_BLOCKS = OrderedDict({
    "IN00": r"input_blocks\.0\..*", "IN01": r"input_blocks\.1\..*", "IN02": r"input_blocks\.2\..*",
    "IN03": r"input_blocks\.3\..*", "IN04": r"input_blocks\.4\..*", "IN05": r"input_blocks\.5\..*",
    "IN06": r"input_blocks\.6\..*", "IN07": r"input_blocks\.7\..*", "IN08": r"input_blocks\.8\..*",
    "M00": r"middle_block\..*",
    "OUT00": r"output_blocks\.0\..*", "OUT01": r"output_blocks\.1\..*", "OUT02": r"output_blocks\.2\..*",
    "OUT03": r"output_blocks\.3\..*", "OUT04": r"output_blocks\.4\..*", "OUT05": r"output_blocks\.5\..*",
    "OUT06": r"output_blocks\.6\..*", "OUT07": r"output_blocks\.7\..*", "OUT08": r"output_blocks\.8\..*",
})

# Defines regular expression patterns for identifying different blocks within CLIP-L model's state dictionary.
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

# Defines regular expression patterns for identifying different blocks within CLIP-G model's state dictionary.
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