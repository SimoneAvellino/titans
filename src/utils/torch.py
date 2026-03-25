import torch


def get_device():
    """Rileva l'hardware migliore disponibile (Apple Silicon, NVIDIA o CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")  # CPU
