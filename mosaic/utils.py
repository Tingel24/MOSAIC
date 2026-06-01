from natsort import natsorted
import torch
import torch.nn.functional as F


def cosine_similarity(
        tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.
    Both tensors should be 1D or have the same shape.
    """
    tensor1 = F.normalize(tensor1, p=2, dim=-1)
    tensor2 = F.normalize(tensor2, p=2, dim=-1)
    return (tensor1 * tensor2).sum(dim=-1)


def get_vocab_token_embeddings(model):
    embedding_layer = model.get_input_embeddings()
    return embedding_layer.weight.detach()[:1000]  # TODO: Filter by perplexity


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_tsne_from_dict_activations(class_activations: dict, class_names=None, perplexity=30, n_iter=1000,
                                    normalize=False):
    """
    Plots t-SNE projections per layer from activations stored as List[Dict[layer_name, tensor]].

    Parameters:
        class_activations (dict): Mapping class_label -> List[Dict[str, Tensor]]
        class_names (dict): Optional mapping class_label -> readable name
        perplexity (int): t-SNE perplexity
        n_iter (int): t-SNE iteration count
        normalize (bool): Whether to normalize activations before t-SNE
    """
    # Get list of layer names from first prompt of first class
    first_prompt = next(iter(class_activations.values()))[0]
    layer_names = list(first_prompt.keys())

    for layer in layer_names:
        layer_data = []
        layer_labels = []

        for label, prompts in class_activations.items():
            for prompt_dict in prompts:
                vec = prompt_dict[layer]
                vec = vec.detach().cpu().numpy()
                layer_data.append(vec)
                layer_labels.append(label)

        X = np.stack(layer_data)
        y = np.array(layer_labels)

        if normalize:
            X = StandardScaler().fit_transform(X)

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        X_reduced = tsne.fit_transform(X)

        plt.figure(figsize=(8, 6))
        colors = ['orange', 'lightblue', 'red', 'blue']
        for i, label in enumerate(np.unique(y)):
            idx = y == label
            name = class_names[label] if class_names else f"Class {label}"
            plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], color=colors[i % len(colors)],
                        label=name, alpha=0.7, s=50)

        plt.title(f"t-SNE: {layer}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/tsne_{layer}.png")


if __name__ == '__main__':
    # Parameters
    num_prompts_per_class = 30
    layers = ["layer_1", "layer_2", "layer_3"]
    seq_lens = [5, 10, 15]
    hidden_dim = 2048


    # Generate random activations mimicking your shape: (1, seq_len, hidden_dim)
    def random_activation(seq_len):
        return torch.randn(1, seq_len, hidden_dim)


    # Build random data for two classes (0 and 1)
    class_activations = {
        0: [{layer: random_activation(seq_lens[i % len(seq_lens)]) for i, layer in enumerate(layers)}
            for _ in range(num_prompts_per_class)],
        1: [{layer: random_activation(seq_lens[i % len(seq_lens)]) for i, layer in enumerate(layers)}
            for _ in range(num_prompts_per_class)]
    }

    class_names = {
        0: "Class 0",
        1: "Class 1"
    }

    # Call your plotting function
    plot_tsne_from_dict_activations(
        class_activations=class_activations,
        class_names=class_names,
        perplexity=20,
        n_iter=500,
        normalize=True
    )

from PIL import Image
import os


def stack_pngs_vertically(folder_path, output_path="stacked_image.png"):
    """
    Stacks all PNG images in a folder vertically and saves the result.

    Parameters:
        folder_path (str): Path to the folder containing .png images.
        output_path (str): Path to save the stacked output image.
    """
    # Get list of PNG files
    png_files = [f for f in natsorted(os.listdir(folder_path)) if f.lower().endswith('.png')]
    if not png_files:
        raise ValueError("No PNG files found in the folder.")

    # Load images
    images = [Image.open(os.path.join(folder_path, f)) for f in png_files]

    # Optionally: Resize all images to same width (based on first image)
    base_width = images[0].width
    resized_images = [
        img if img.width == base_width else img.resize((base_width, int(img.height * base_width / img.width)))
        for img in images
    ]

    # Compute total height
    total_height = sum(img.height for img in resized_images)
    max_width = max(img.width for img in resized_images)

    # Create new blank image and paste all
    stacked_image = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in resized_images:
        stacked_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save output
    stacked_image.save(output_path)
    print(f"Saved stacked image to {output_path}")
