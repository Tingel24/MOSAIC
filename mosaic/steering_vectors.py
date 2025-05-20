import numpy as np
import torch
import transformers
from baukit import Trace, TraceDict
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
    # TODO use contrastive layer selection
    linear_layers = [m for m in model.modules() if isinstance(m, module_type)]
    return linear_layers[len(linear_layers) // 2], f"{len(linear_layers) // 2} / {len(linear_layers)}"

def collect_layer_activations(model, tokenizer, prompt, layer_type = LlamaDecoderLayer, device='cuda'):
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

    # Only retain the input to each decoder layer (residual stream)
    residuals = {name: trace.output for name, trace in tracer.items()}
    return residuals

def select_layer_contrastive(
            teacher_model, model, tokenizer, prompts, device='cuda', progress=tqdm
    ):
        positive_activations = {}
        negative_activations = {}
        for prompt in progress(prompts):
            positive_activations[prompt] = collect_layer_activations(teacher_model, tokenizer=tokenizer, prompt=prompt, device=device)
            negative_activations[prompt] = collect_layer_activations(model, tokenizer=tokenizer, prompt=prompt, device=device)

        js_score = []

        for index in range(len(positive_activations.items())):
            module_name = list(positive_activations.items())[index][0]
            positive_activation = list(positive_activations.items())[index][1]
            negative_activation = list(negative_activations.items())[index][1]

            positive = torch.stack(positive_activation)
            negative = torch.stack(negative_activation)

            p = F.softmax(positive, dim=-1)
            n = F.softmax(negative, dim=-1)
            M = 0.5 * (F.softmax(positive, dim=-1) + F.softmax(negative, dim=-1))
            kl1 = F.kl_div(p, M, reduction="none").mean(-1)
            kl2 = F.kl_div(n, M, reduction="none").mean(-1)
            js_divs = 0.5 * (kl1 + kl2).mean(-1)
            js_score.append(js_divs.item())

        js_optimum_layer = js_score.index(max(js_score))
        optimum_layer_name = list(positive_activations.keys())[js_optimum_layer]
        print(f"\033[96mJS Optimum Layer: {js_optimum_layer}\033[0m")

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
            {"role": "system", "content": "You are a chatbot"},
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
    print(
        f"length steering_vec: {steering_vec.norm():.2f} (will be normalized to 1 now)"
    )

    # Normalize to unit length
    steering_vec = steering_vec / steering_vec.norm()

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
