from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemoryLinear(nn.Module):
    """
    Matrix-valued memory M ∈ ℝ^{D×D}.
    Update rule (generalized Gated DeltaNet with momentum):
        S_t = η_t · S_{t-1} − θ_t · (Mk − v)k^T    # momentum surprise
        M_t = (1−α_t) · M_{t-1} + S_t               # forget + accumulate
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d = d_model
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # Data-dependent gates (all ∈ (0,1) via sigmoid)
        self.alpha_proj = nn.Linear(d_model, 1)  # forget
        self.theta_proj = nn.Linear(d_model, 1)  # lr scale
        self.eta_proj = nn.Linear(d_model, 1)    # momentum decay
        # Buffers reset per sequence
        self.M: Optional[torch.Tensor] = None
        self.S: Optional[torch.Tensor] = None

    def reset(self, B: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.M = torch.zeros(B, self.d, self.d, device=device, dtype=dtype)
        self.S = torch.zeros_like(self.M)

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        """q → M*q  (read without updating weights)."""
        q = F.normalize(self.W_Q(x), dim=-1)           # (B,T,D)
        return torch.bmm(q, self.M.transpose(-1, -2))  # (B,T,D)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Surprise-based write (in-place, no autograd)."""
        B, T, D = x.shape
        k = F.normalize(self.W_K(x), dim=-1)  # (B,T,D)
        v = self.W_V(x)

        avg = x.mean(1, keepdim=True)          # (B,1,D) for gates
        alpha = torch.sigmoid(self.alpha_proj(avg)).squeeze(-1)  # (B,1)
        theta = torch.sigmoid(self.theta_proj(avg)).squeeze(-1) * 0.1
        eta = torch.sigmoid(self.eta_proj(avg)).squeeze(-1)

        # Momentary surprise: ∇ℓ = (Mk − v)k^T
        pred = torch.bmm(k, self.M.transpose(-1, -2))       # (B,T,D)
        error = pred - v                                      # (B,T,D)
        grad_M = torch.bmm(error.transpose(1, 2), k) / T    # (B,D,D)

        a = alpha.unsqueeze(-1)
        t = theta.unsqueeze(-1)
        e = eta.unsqueeze(-1)

        self.S = e * self.S - t * grad_M
        self.M = (1 - a) * self.M + self.S


class NeuralMemoryMLP(nn.Module):
    """
    MLP-valued memory updated via inner-loop gradient descent.
    Loss: ℓ(M; x) = ‖M(k) − v‖²
    Requires PyTorch >= 2.0 (torch.func).
    """

    def __init__(self, d_model: int, n_layers: int = 2, inner_lr: float = 0.01):
        super().__init__()
        self.inner_lr = inner_lr
        layers: list[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_model, d_model))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.alpha_proj = nn.Linear(d_model, 1)
        self._fast: dict[str, torch.Tensor] = {}

    def reset(self, B: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self._fast = {n: p.detach().clone().to(dtype) for n, p in self.mlp.named_parameters()}

    def _fwd(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        from torch.func import functional_call
        return functional_call(self.mlp, params, x)

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.W_Q(x), dim=-1)
        if self._fast:
            try:
                return self._fwd(q, self._fast)
            except Exception:
                pass
        return self.mlp(q)

    def update(self, x: torch.Tensor):
        k = F.normalize(self.W_K(x), dim=-1)
        v = self.W_V(x)
        alpha = torch.sigmoid(self.alpha_proj(x.mean(1))).item()
        theta = self.inner_lr

        try:
            from torch.func import grad, functional_call

            def loss_fn(params):
                return F.mse_loss(functional_call(self.mlp, params, k), v)

            grads = grad(loss_fn)(self._fast)
            with torch.no_grad():
                self._fast = {
                    n: (1 - alpha) * p - theta * grads[n]
                    for n, p in self._fast.items()
                }
        except ImportError:
            # Fallback: manual SGD (no gradient flow through memory update)
            pred = self.mlp(k)
            loss = F.mse_loss(pred, v)
            loss.backward(retain_graph=True)
            with torch.no_grad():
                for n, p in self.mlp.named_parameters():
                    if p.grad is not None:
                        self._fast[n] = (1 - alpha) * self._fast[n] - theta * p.grad
                        p.grad = None
