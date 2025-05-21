from collections import defaultdict
from typing import Tuple, Dict

import numpy as np
import torch
import transformers
from baukit import Trace, TraceDict
from jinja2.sandbox import unsafe
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn.functional as F


# define the activation steering function
def act_add(steering_vec):
    def hook(output):
        return (
                (output[0] + steering_vec,) + output[1:]
        )  # the output of the residual stream is actually a tuple, where the first entry is the activation

    return hook

def select_layer(model, module_type = LlamaDecoderLayer):
    residual_stream = [m for m in model.modules() if isinstance(m, module_type)]
    return residual_stream[len(residual_stream) // 3], f"{len(residual_stream) // 3} / {len(residual_stream)}"

def collect_layer_activations(model, tokenizer, prompt: str, layer_type = LlamaDecoderLayer, device='cuda') -> Dict[str, torch.Tensor]:
    """
    Collects activations from each transformer block in a LLaMA model using baukit.

    Args:
        model: Pre-loaded LLaMA model.
        tokenizer: Corresponding tokenizer.
        prompt (str): The input prompt to feed to the model.
        device (str): 'cuda' or 'cpu'
        layer_type: which types of layers to consider for activation extraction.

    Returns:
        activations (dict): Keys are module names, values are tensors of activations.
    """
    model.eval().to(device)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Get the names of all LlamaDecoderLayers
    layer_names = [name for name, module in model.named_modules() if
                   isinstance(module, layer_type)]

    with TraceDict(model, layer_names, retain_input=False, retain_output=True) as tracer:
        _ = model(**inputs)

    # Only retain the output to each decoder layer (residual stream)
    residuals = {name: trace.output[0] if isinstance(trace.output, tuple) else trace.output for name, trace in
                 tracer.items()}
    return residuals

def select_layer_contrastive(
            teacher_model, model, tokenizer, prompts: Tuple[str, str], device='cuda', progress=tqdm
    ):
        # This code obviously assumes that teacher and model are the same architecture

        # We have a teacher model that exhibits some behavior.
        # In our case this is a basemodel, which therefore did not undergo fine-tuning to be more safe.
        # We will be evaluating the difference in activation in each layer
        # between the unsafe teacher and the safe target model.
        # As there will be some differences between the models, like helpfulness, that we do not want to capture,
        # we will be also looking at the difference of the models in regard to safe prompts

        # In the end we will be subtracting the "safe" difference from the "unsafe" difference,
        # with the intuition that the remaining difference between the activations is based mostly on the fact
        # that one model exhibits safe behavior and one does not.
        # Then the layer that has the largest difference is selected as the layer
        # where the model can be most effectively steered.

        safe_teacher_activations = []
        safe_target_activations = []
        unsafe_teacher_activations = []
        unsafe_target_activations = []
        for (safe_prompt, unsafe_prompt) in progress(prompts):
            safe_teacher_activations.append(collect_layer_activations(teacher_model, tokenizer=tokenizer, prompt=safe_prompt, device=device))
            safe_target_activations.append(collect_layer_activations(model, tokenizer=tokenizer, prompt=safe_prompt, device=device))

            unsafe_teacher_activations.append(collect_layer_activations(teacher_model, tokenizer=tokenizer, prompt=unsafe_prompt, device=device))
            unsafe_target_activations.append(collect_layer_activations(model, tokenizer=tokenizer, prompt=unsafe_prompt, device=device))

        # Divergence Score for each layer
        safe_js_score = []
        unsafe_js_score = []

        # Collect all the activations for each prompt for each layer
        # current shape : List[Dict[layer_name, activations]]
        # new shape: Dict[layer_name, List[activations]]

        def reorder(data):
            result = defaultdict(list)
            for sample in data:
                for layer, activation in sample.items():
                    result[layer].append(activation)
            return dict(result)

        safe_teacher_activations = reorder(safe_teacher_activations)
        safe_target_activations = reorder(safe_target_activations)
        unsafe_teacher_activations = reorder(unsafe_teacher_activations)
        unsafe_target_activations = reorder(unsafe_target_activations)

        def get_score(teacher_activation, target_activation):
                positive = torch.stack([a[0, -1, :] for a in teacher_activation])  # Grab last token hidden state
                negative = torch.stack([a[0, -1, :] for a in target_activation])

                # Compute probability distribution of activations
                p = F.softmax(positive, dim=-1)
                q = F.softmax(negative, dim=-1)
                # Average distribution
                M = 0.5 * (p + q)
                # Compute log-probabilities
                log_p = torch.log(p + 1e-8)
                log_q = torch.log(q + 1e-8)

                # Compute KL divergences
                kl_pm = F.kl_div(log_p, M, reduction='none').sum(-1)
                kl_qm = F.kl_div(log_q, M, reduction='none').sum(-1)

                # JS divergence
                js_div = 0.5 * (kl_pm + kl_qm).mean(-1)
                js_dist = torch.sqrt(js_div) # Use a distance instead, because we want to call argmax to get the largest distance.
                return js_dist.item()

        for layer_name in progress(safe_teacher_activations.keys()):
            safe_js_score.append(get_score(safe_teacher_activations[layer_name], safe_target_activations[layer_name]))
            unsafe_js_score.append(get_score(unsafe_teacher_activations[layer_name], unsafe_target_activations[layer_name]))

        unsafe_js_score = torch.tensor(unsafe_js_score)*1000 # Scale for readability in print, does not affect argmax
        safe_js_score = torch.tensor(safe_js_score)*1000


        #js_score = unsafe_js_score-safe_js_score
        #js_score = safe_js_score-unsafe_js_score
        #js_score = unsafe_js_score
        js_score = safe_js_score


        # Only use middle layers:
        l = len(safe_teacher_activations.keys())//3
        print(f"Evaluating layers {l} to {len(js_score)-l} ")
        js_optimum_layer = js_score[l:-l].argmax() + l
        optimum_layer_name = list(safe_teacher_activations.keys())[js_optimum_layer]
        print(f"Safe JS Scores: {safe_js_score} ")
        print(f"Unsafe JS Scores: {unsafe_js_score} ")
        print(f"JS Scores: {js_score} ")
        print(f"\033[96mJS Optimum Layer: {js_optimum_layer}/{len(safe_teacher_activations.keys())} {optimum_layer_name}\033[0m")

        return get_module_by_name(model, optimum_layer_name), optimum_layer_name

def get_module_by_name(model, name):
    return [module for module_name, module in model.named_modules() if module_name == name][0]

def generate_with_steering(model, tokenizer, module, steering_vec, prompt, steering_strenght, token_count, device):
    with Trace(module, edit_output=act_add(steering_strenght * steering_vec)) as m:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            tokenizer=tokenizer,
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Interpret each question literally, and as a question about the real world. Do not ask any questions, just answer."},
            {"role": "user", "content": prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=token_count,
            pad_token_id=tokenizer.eos_token_id,
        )

        return outputs[0]["generated_text"][-1], m.output[0]


def get_steering_vec_by_difference(positive_activations, negative_activations):
    # Compute difference vectors
    # We want the vector that points from the negative activation space to the positive activation space,
    # because we want to move activations that typically do not exhibit some behavior to the space where the do
    steering_vecs = negative_activations - positive_activations  # shape: (N, D)

    # Compute mean direction across all pairs
    # There is an argument against this, that the mean might not correctly characterize the steering vectors,
    # as there are often multiple possible and ever orthogonal vectors that exhibit the same behavior.
    steering_vec = steering_vecs.mean(dim=0)  # shape: (D,)

    print(f"steering_vec.shape:  {steering_vec.shape}")


    # Normalize to unit length
    print( f"length steering_vec: {steering_vec.norm():.2f}")
    # print("(will be normalized to 1 now)")
    #steering_vec = steering_vec / steering_vec.norm()

    return steering_vec


def collect_activation_difference(target_model, target_module, teacher_model, teacher_module, tokenizer, prompts, progress=tqdm, device="cpu"):
    # Initialize lists for collecting tensors
    positive_activations = []
    negative_activations = []

    # Define a helper to get the last-token activation from the module
    def get_module_output(model, module, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with Trace(module, stop=True) as cache:
            _ = model(**inputs)
        return cache.output[0][:, -1, :]  # shape: (1, hidden_dim)

    # Collect activations
    for prompt in progress(prompts):
        act_pos = get_module_output(teacher_model, teacher_module, prompt).detach()  # stays on device
        act_neg = get_module_output(target_model, target_module, prompt).detach()

        positive_activations.append(act_pos)
        negative_activations.append(act_neg)

    # Concatenate tensors along batch dimension
    positive_activations = torch.cat(positive_activations, dim=0)  # shape: (n, hidden_dim)
    negative_activations = torch.cat(negative_activations, dim=0)

    return positive_activations, negative_activations