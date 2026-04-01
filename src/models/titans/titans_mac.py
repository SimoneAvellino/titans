from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.models.titans.titans_config import TitansConfig
from src.models.titans.neural_memory import NeuralMemoryLinear, NeuralMemoryMLP
from src.models.titans.persistent_memory import PersistentMemory


class Qwen2TitansMAC(nn.Module):
    """
    Drop-in wrapper: same input/output signature as Qwen2ForCausalLM.
    New trainable params: neural_memory, persistent_memory, out_gate, conv.
    """

    def __init__(self, base: Qwen2ForCausalLM, cfg: TitansConfig):
        super().__init__()
        self.cfg = cfg
        self.base = base
        D = base.config.hidden_size  # 1536 for Qwen2-1.5B

        if cfg.freeze_base:
            for p in base.parameters():
                p.requires_grad_(False)

        # Memory modules
        self.neural_memory: NeuralMemoryLinear | NeuralMemoryMLP
        if cfg.use_linear_memory:
            self.neural_memory = NeuralMemoryLinear(D)
        else:
            self.neural_memory = NeuralMemoryMLP(D, cfg.memory_depth, cfg.inner_lr)

        self.persistent = PersistentMemory(D, cfg.n_persistent)

        # Output gate: o_t = y_t ⊗ σ(gate(norm(M*(y_t))))
        self.out_norm = nn.RMSNorm(D, eps=1e-6)
        self.out_gate = nn.Linear(D, D, bias=False)

        # Depthwise conv on embeddings (§4.4)
        self.conv = nn.Conv1d(D, D, kernel_size=3, padding=1, groups=D, bias=False)

    # ── internal helpers ──────────────────────

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.base.model.embed_tokens(input_ids)
        # depthwise conv (operates on sequence dim)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x

    def _qwen2_layers(
        self,
        h: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run decoder stack + norm on already-embedded hidden states."""
        B, S, _ = h.shape
        if pos is None:
            pos = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1)
        out = self.base.model(
            inputs_embeds=h,
            attention_mask=None,
            position_ids=pos,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state

    # ── forward ──────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        B, N = input_ids.shape
        C, Np = self.cfg.chunk_size, self.cfg.n_persistent
        device = input_ids.device

        embeds = self._embed(input_ids)  # (B, N, D)
        self.neural_memory.reset(B, device, dtype=embeds.dtype)

        chunks_out: list[torch.Tensor] = []

        for i in range((N + C - 1) // C):
            s, e = i * C, min((i + 1) * C, N)
            chunk = embeds[:, s:e]  # (B, C', D)
            C_ = e - s

            # 1. Retrieve: h_t = M*(q_t)
            h_t = self.neural_memory.retrieve(chunk)  # (B, C', D)

            # 2. Persistent tokens
            p = self.persistent(B)  # (B, Np, D)

            # 3. [P || h_t || S(t)]
            x_tilde = torch.cat([p, h_t, chunk], 1)  # (B, Np+2C', D)

            # 4. Qwen2 attention over extended context
            S_total = x_tilde.shape[1]
            pos = torch.arange(S_total, device=device).unsqueeze(0).expand(B, -1)
            y_all = self._qwen2_layers(x_tilde, pos)  # (B, Np+2C', D)

            # Extract S(t) portion only
            y_t = y_all[:, Np + C_ : Np + 2 * C_]  # (B, C', D)

            # 5. Update neural memory with attention output
            self.neural_memory.update(y_t)

            # 6. Gate: o_t = y_t ⊗ σ(gate(norm(M*(y_t))))
            m_out = self.neural_memory.retrieve(y_t)
            gate = torch.sigmoid(self.out_gate(self.out_norm(m_out)))
            o_t = y_t * gate  # (B, C', D)

            chunks_out.append(o_t)

        hidden = torch.cat(chunks_out, dim=1)  # (B, N, D)
        logits = self.base.lm_head(hidden)  # (B, N, vocab)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, **kw):
        """Minimal greedy generation (wraps base model for compatibility)."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(input_ids)
                next_t = out.logits[:, -1, :].argmax(-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                if next_t.item() == self.base.config.eos_token_id:
                    break
        return input_ids


def build_titans_mac(
    model_name: str = "Qwen/Qwen2-1.5B",
    cfg: Optional[TitansConfig] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple["Qwen2TitansMAC", object]:
    """Return (model, tokenizer)."""
    from transformers import AutoTokenizer

    if cfg is None:
        cfg = TitansConfig()
    base = Qwen2ForCausalLM.from_pretrained(model_name, dtype=dtype)
    model = Qwen2TitansMAC(base, cfg).to(dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok
