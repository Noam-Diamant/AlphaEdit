from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    KEYS = [
        "layers",
        "fact_token",
        "v_num_grad_steps",
        "v_lr",
        "v_loss_layer",
        "v_weight_decay",
        "clamp_norm_factor",
        "kl_factor",
        "mom2_adjustment",
        "context_template_length_params",
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
        "stats_dir",
    ]
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    stats_dir: str
