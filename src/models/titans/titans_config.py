from dataclasses import dataclass


@dataclass
class TitansConfig:
    chunk_size: int = 64            # C — segment length
    n_persistent: int = 16          # N_p — persistent memory tokens
    memory_depth: int = 2           # L_M — MLP depth (>=2 = "deep memory")
    use_linear_memory: bool = True  # True=Delta Rule (fast), False=MLP (expressive)
    inner_lr: float = 0.01          # θ — inner-loop learning rate scale
    freeze_base: bool = True        # freeze Qwen2 weights (set False to fine-tune all)
