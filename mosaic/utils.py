import torch.nn.functional as F
import torch

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