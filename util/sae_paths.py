PATH_MAPPING = {
    # Mapping between model identifiers and the on-disk location of the corresponding
    # dictionary-learning SAE. Update this mapping whenever a new SAE is trained.
    "gpt2-xl": "/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/dictionary_learning_demo/._gpt2xl_gpt2-xl_batch_top_k_tokens500M",
    "Qwen/Qwen2-0.5B": "/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/dictionary_learning_demo/._qwen2.5_0.5B_Qwen_Qwen2.5-0.5B_batch_top_k_tokens500M",
    "Qwen/Qwen2.5-0.5B": "/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/dictionary_learning_demo/._qwen2.5_0.5B_Qwen_Qwen2.5-0.5B_batch_top_k_tokens500M",
    "EleutherAI/gpt-j-6B": "/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/dictionary_learning_demo/._gptj6b_EleutherAI_gpt-j-6b_batch_top_k_tokens500M"
}


def get_sae_path(model_name: str, layer: int = 0) -> str:
    """Return the path to the dictionary-learning SAE that matches *model_name*.

    Parameters
    ----------
    model_name: str
        The *_name_or_path* field that appears in the loaded transformer
        ``model.config``.
    layer: int, optional
        The layer number to get the SAE path for. Defaults to 0.

    Returns
    -------
    str
        Absolute path to the SAE checkpoint/trainer folder.

    Raises
    ------
    ValueError
        If *model_name* does not have a corresponding entry in ``PATH_MAPPING``,
        or if the SAE path is registered but not yet available (None).
    """
    if model_name not in PATH_MAPPING:
        raise ValueError(
            f"No SAE path registered for model '{model_name}'. "
            "Update `rome/sae_paths.py::PATH_MAPPING` to add a new entry."
        )

    path = PATH_MAPPING[model_name]
    if path is None:
        raise ValueError(
            f"SAE path for model '{model_name}' is registered but not yet available. "
            "Train the SAE first and update the path in `rome/sae_paths.py::PATH_MAPPING`."
        )

    return path + f"/mlp_out_layer_{layer}/trainer_0"

if __name__ == "__main__":
    from typing import Dict, List, Tuple
    import wandb
    import numpy as np
    import torch
    from matplotlib.style import context
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F
    from rome import repr_tools
    from rome.sae_paths import get_sae_path
    from util import nethook
    from util.globals import *
    from dictionary_learning.utils import load_dictionary
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch



    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # for layer in [13, 14, 15, 16, 17]:  
    #     sae_path = get_sae_path("gpt2-xl", layer)
    #     print(f"Loading GPT2-XL SAE layer {layer}")
    #     try:
    #         sae, _, _ = load_dictionary(sae_path, device)
    #         # Load the model with safetensors to avoid the PyTorch vulnerability
    #         model = AutoModelForCausalLM.from_pretrained(
    #             "gpt2-xl",
    #             device_map="auto",  # This will handle device placement automatically
    #             torch_dtype="auto",  # This will use the most efficient dtype for your GPU
    #             use_safetensors=True  # This avoids the PyTorch vulnerability
    #         )
    #         hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))
    #         delta = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
    #         # Load the tokenizer
    #         delta_features = sae.encode(delta)
    #         print("GPT2-XL delta features shape:", delta_features.shape)
    #         print(delta_features)
    #     except Exception as e:
    #         print(f"Error loading GPT2-XL SAE layer {layer}: {e}")

    # Now test GPT-J
    for layer in [8]:
        try:
            sae_path = get_sae_path("EleutherAI/gpt-j-6B", layer)
            print(f"Loading GPT-J SAE layer {layer}")
            sae, _, _ = load_dictionary(sae_path, device)

            # Load GPT-J model
            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                device_map="auto",  # This will handle device placement automatically
                torch_dtype="auto",  # This will use the most efficient dtype for your GPU
                use_safetensors=True  # This avoids the PyTorch vulnerability
            )
            hidden_size = getattr(model.config, "n_embd", getattr(model.config, "hidden_size", None))
            delta = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
            delta_features = sae.encode(delta)
            print("GPT-J delta features shape:", delta_features.shape)
            print(delta_features)
        except Exception as e:
            print(f"Error loading GPT-J SAE layer {layer}: {e}")