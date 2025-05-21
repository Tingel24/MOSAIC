from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baukit import Trace
from tqdm import tqdm
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

from mosaic.steering_vectors import act_add
from mosaic.utils import cosine_similarity, get_vocab_token_embeddings


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
    return 1. - max_sim


def alignment_loss(activation, steered_activation):
    return 1 - cosine_similarity(activation, steered_activation)


def magnitude_loss(activation, steering_strength):
    a_norm = torch.norm(activation, dim=-1, keepdim=True) + 1e-6
    return (a_norm - steering_strength).pow(2)


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


def train_soft_prompt(model: LlamaForCausalLM,
                      tokenizer: LlamaTokenizer,
                      module: nn.Module,
                      steering_vec,
                      prompts: List[str] = None,
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
    if prompts is None:
        training_prompts = [
            "Write a cover letter for a software engineering job",
            "Explain quantum computing in simple terms",
            "Summarize this article",
            "Translate this paragraph into Spanish",
            "Generate a meal plan for weight loss",
            "Fix bugs in this Python code",
            "Help me brainstorm business names",
            "Create a social media caption for a product launch",
            "Write a story about a time-traveling cat",
            "Compare the pros and cons of electric vs. gas cars",
            "Help me study for the SAT",
            "Design a workout routine for beginners",
            "Convert this text into a professional email",
            "Explain the plot of Hamlet",
            "Generate SQL queries based on this dataset",
            "Make a packing list for a two-week trip to Japan",
            "Create a lesson plan for teaching photosynthesis",
            "Draft a privacy policy for a mobile app",
            "Write a wedding speech for the best man",
            "Give me ideas for a D&D campaign setting",
        ]
    else:
        training_prompts = prompts
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
            modified_input = torch.cat([soft_prompt, embedded_input], dim=0).unsqueeze(0)
            new_attention_mask = torch.cat([
                torch.ones(1, soft_prompt_length).to(device),
                attention_mask
            ], dim=1)

            with Trace(module, stop=True, edit_output=act_add(steering_strength * steering_vec)) as steered_cache:
                _ = model(input_ids=input_ids)

            steered_activation = steered_cache.output[0][:, -1, :]

            with Trace(module, stop=True) as cache:
                _ = model(inputs_embeds=modified_input, attention_mask=new_attention_mask)

            activation = cache.output[0][:, -1, :]

            l = []
            if loss_weight_magnitude > 0:
                l.append(loss_weight_magnitude * magnitude_loss(activation, steering_strength))
            if loss_weight_proximity > 0:
                l.append(loss_weight_proximity * prompt_token_proximity_loss(soft_prompt, token_embeddings))
            if loss_weight_alignment > 0:
                l.append(loss_weight_alignment * alignment_loss(activation, steered_activation))
            total_loss += sum(l)

        total_loss = total_loss / len(training_prompts)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

    return soft_prompt.detach(), losses
