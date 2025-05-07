import marimo as mo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baukit import Trace
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

from mosaic.utils import cosine_similarity


def soft_prompt_metrics(model, tokenizer, module, soft_prompt, soft_prompt_length, steering_vec, steering_strength,
                        prompt: str, device="cpu"):
    # Tokenize and embed
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get embedding layer
    embedding_layer = model.get_input_embeddings()
    embedded_input = embedding_layer(input_ids).squeeze(0)  # (seq_len, D)

    # Prepare combined input
    soft_prompt_device = soft_prompt.to(device)
    modified_input = torch.cat(
        [soft_prompt_device, embedded_input], dim=0
    ).unsqueeze(0)

    # Adjust attention mask
    new_attention_mask = torch.cat(
        [torch.ones(1, soft_prompt_length).to(device), attention_mask], dim=1
    )

    # Run and trace
    with Trace(module, stop=True) as cache:
        _ = model(
            inputs_embeds=modified_input, attention_mask=new_attention_mask
        )

    activation = cache.output[0][:, -1, :].squeeze(0)  # (D,)

    # Cosine similarity with steering vector
    cosine_sim = F.cosine_similarity(activation, steering_vec, dim=0).item()

    # Norms
    act_norm = activation.norm().item()
    target_norm = steering_strength
    norm_diff = abs(act_norm - target_norm)

    return {
        "cosine_similarity": round(cosine_sim, 4),
        "activation_norm": round(act_norm, 4),
        "target_norm": round(target_norm, 4),
        "norm_difference": round(norm_diff, 4),
    }


def prompt_token_proximity_loss(soft_prompt, embeddings):
    sim = F.cosine_similarity(soft_prompt.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    max_sim = sim.max(dim=1).values
    return 1 - max_sim.mean()


def alignment_loss(activation, unmodified_activation):
    return 1 - cosine_similarity(activation, unmodified_activation).mean()


def magnitude_loss(activation, steering_strength):
    a_norm = torch.norm(activation, dim=-1, keepdim=True) + 1e-6
    return (a_norm - steering_strength).pow(2).mean()


def generate_with_soft_prompt(model, tokenizer, soft_prompt, soft_prompt_length, test_sentence: str,
                              token_count: int = 100, device="cpu"):
    # Tokenize input
    inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get token embeddings
    embedding_layer = model.get_input_embeddings()
    embedded_input = embedding_layer(input_ids).squeeze(0)  # (seq_len, D)

    # Combine soft prompt with token embeddings
    soft_prompt_device = soft_prompt.to(device)
    combined_input = torch.cat(
        [embedded_input, soft_prompt_device], dim=0
    ).unsqueeze(0)

    # Adjust attention mask to include soft prompt
    new_attention_mask = torch.cat(
        [torch.ones(1, soft_prompt_length).to(device), attention_mask], dim=1
    )

    # Generate from embeddings
    output_ids = model.generate(
        inputs_embeds=combined_input,
        attention_mask=new_attention_mask,
        max_new_tokens=token_count,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def get_vocab_token_embeddings(model):
    embedding_layer = model.get_input_embeddings()
    return embedding_layer.weight.detach()[:100]  # TODO: Filter by perplexity


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
                      progress=tqdm
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

    token_embeddings = get_vocab_token_embeddings(model)

    losses = []
    for _ in progress(range(num_steps)):
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

    return soft_prompt.detach(), losses
