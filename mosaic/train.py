from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from baukit import Trace
import torch
from transformers import LlamaForCausalLM
import numpy as np
import transformers
import marimo as mo
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mosaic.utils import cosine_similarity


def prompt_token_proximity_loss(soft_prompt, embeddings):
    sim = F.cosine_similarity(soft_prompt.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    max_sim = sim.max(dim=1).values
    return 1 - max_sim.mean()

def alignment_loss(activation, unmodified_activation):
    return  1 - cosine_similarity(activation, unmodified_activation).mean()


def magnitude_loss(activation, steering_strength):
    a_norm = torch.norm(activation, dim=-1, keepdim=True) + 1e-6
    return (a_norm - steering_strength).pow(2).mean()

import torch.nn.functional as F


# define the activation steering function
def act_add(steering_vec):
    def hook(output):
        return (
            (output[0] + steering_vec,) + output[1:]
        )  # the output of the residual stream is actually a tuple, where the first entry is the activation

    return hook


def generate_with_steering(model, tokenizer, module,steering_vec, prompt, steering_strenght, token_count, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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

def train_soft_prompt(model: LlamaForCausalLM,
                      tokenizer: LlamaTokenizer,
                      module: nn.Module,
                      soft_prompt_length=5,
                      learning_rate=0.01,
                      num_steps=100,
                      steering_strength=1.0,
                      loss_weight_alignment=1.0,
                      loss_weight_magnitude=1.0,
                      loss_weight_proximity=0.0,
                      device="cpu",
                      ):
    training_prompts = [
        "The following is a helpful and honest answer:",
        "Here's how to solve this problem clearly:",
        "This is what I would do in that situation:",
        "Here’s a safe and respectful explanation:",
        "An appropriate and kind response might be:",
        "Let me help you understand this better:",
        "As a responsible assistant, here’s my advice:",
        "Here's a factually accurate answer to that:",
        "Let’s approach this question constructively:",
        "To clarify this thoughtfully, consider:"
    ]

    embedding_dim = model.config.hidden_size
    soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, embedding_dim, device=device))

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embedding_layer = model.get_input_embeddings()
    optimizer = optim.Adam([soft_prompt], lr=learning_rate)

    token_embeddings = embedding_layer.weight.detach()[:100] # TODO: Filter by perplexity

    losses = []
    for _ in mo.status.progress_bar(range(num_steps)):
        total_loss = 0.0

        for prompt in training_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            embedded_input = embedding_layer(input_ids).squeeze(0)
            modified_input = torch.cat([embedded_input, soft_prompt], dim=0).unsqueeze(0)
            new_attention_mask = torch.cat([
                torch.ones(1, soft_prompt_length).to(device),
                attention_mask
            ], dim=1)

            with Trace(module, stop=True) as unmodified_cache:
                _ = model(input_ids=input_ids)

            unmodified_activation = unmodified_cache.output[0][:, -1, :]

            with Trace(module, stop=True) as cache:
                _ = model(inputs_embeds=modified_input, attention_mask=new_attention_mask)

            activation = cache.output[0][:, -1, :]

            alignment = alignment_loss(activation, unmodified_activation)
            proximity = prompt_token_proximity_loss(soft_prompt, token_embeddings)
            magnitude = magnitude_loss(activation, steering_strength)
            total_loss += (loss_weight_proximity * proximity +
                           loss_weight_alignment * alignment +
                           loss_weight_magnitude * magnitude)

        total_loss = total_loss / len(training_prompts)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

    return soft_prompt.detach()