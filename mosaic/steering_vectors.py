import torch
import transformers
from baukit import Trace
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention


# define the activation steering function
def act_add(steering_vec):
    def hook(output):
        return (
                (output[0] + steering_vec,) + output[1:]
        )  # the output of the residual stream is actually a tuple, where the first entry is the activation

    return hook

def select_layer(model, module_type = LlamaAttention):
    # TODO use contrastive layer selection
    linear_layers = [m for m in model.modules() if isinstance(m, module_type)]
    return linear_layers[len(linear_layers) // 4], f"{len(linear_layers) // 4} / {len(linear_layers)}"


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
    steering_vecs = positive_activations - negative_activations  # shape: (N, D)

    # Compute mean direction across all pairs
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
