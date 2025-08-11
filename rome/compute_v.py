from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .rome_hparams import ROMEHyperParams

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    use_modified: bool = False,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    # Hidden size handling compatible with Qwen, LLaMA, GPT families
    if hasattr(model.config, 'n_embd') and isinstance(getattr(model.config, 'n_embd'), int):
        hidden_dim = int(getattr(model.config, 'n_embd'))
    elif hasattr(model.config, 'hidden_size') and isinstance(getattr(model.config, 'hidden_size'), int):
        hidden_dim = int(getattr(model.config, 'hidden_size'))
    else:
        layer_module = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        sample_param = next(layer_module.parameters())
        hidden_dim = sample_param.shape[-1]
    delta = torch.zeros((hidden_dim,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Optional SAE-based modified behavior
    if use_modified:
        import torch.nn.functional as F
        from rome.sae_paths import get_sae_path
        try:
            from dictionary_learning.utils import load_dictionary
        except Exception:
            load_dictionary = None
        device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        if load_dictionary is None:
            raise ImportError("dictionary_learning.utils.load_dictionary not found but use_modified=True")
        sae_path = get_sae_path(model.config._name_or_path, layer)
        sae_original, _, _ = load_dictionary(sae_path, device)
        sae_modified, _, _ = load_dictionary(
            sae_path, device, division_factor=getattr(hparams, "v_division_factor", 1.0)
        )
        sae = sae_original
        do_clamp_relu = True

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            if not use_modified:
                for i, idx in enumerate(lookup_idxs):
                    if len(lookup_idxs)!=len(cur_out):
                        cur_out[idx, i, :] += delta
                    else:
                        cur_out[i, idx, :] += delta
            else:
                batch_size = len(lookup_idxs)
                # Encode delta into feature space and broadcast to batch
                delta_features = sae_modified.encode(delta.unsqueeze(0)).expand(batch_size, -1)

                # Gather residual vectors at lookup positions (handle layout)
                if len(lookup_idxs) != len(cur_out):
                    residual_vectors = torch.stack(
                        [cur_out[idx, i, :].clone() for i, idx in enumerate(lookup_idxs)], dim=0
                    )
                else:
                    residual_vectors = torch.stack(
                        [cur_out[i, idx, :].clone() for i, idx in enumerate(lookup_idxs)], dim=0
                    )

                # Encode current vectors, apply delta in feature space, optional clamp, decode + residual correction
                feature_acts = sae.encode(residual_vectors).detach()
                new_feature_acts = feature_acts + delta_features
                if do_clamp_relu:
                    new_feature_acts = F.relu(new_feature_acts)

                feature_acts_init = sae.encode(target_init.unsqueeze(0)).detach()
                sae_out_init = sae.decode(feature_acts_init).detach()
                diff = (target_init - sae_out_init).detach()

                sae_out = sae.decode(new_feature_acts) + diff

                # Write back
                if len(lookup_idxs) != len(cur_out):
                    for i, idx in enumerate(lookup_idxs):
                        cur_out[idx, i, :] = sae_out[i]
                else:
                    for i, idx in enumerate(lookup_idxs):
                        cur_out[i, idx, :] = sae_out[i]

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            if not use_modified:
                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        if not use_modified:
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
        else:
            kl_loss = torch.zeros((), device=nll_loss.device)
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # Add L1 sparsity term on delta features for modified mode
        if use_modified:
            # Reuse delta_features from last forward by re-encoding here safely
            delta_features_for_loss = sae_modified.encode(delta.unsqueeze(0))
            l1_loss = delta_features_for_loss.norm(p=1)
            alpha = float(getattr(hparams, "v_alpha", 0.0))
            loss = nll_loss + weight_decay + alpha * l1_loss
        else:
            loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret