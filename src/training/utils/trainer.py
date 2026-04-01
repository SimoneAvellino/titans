from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.datasets.simple_text_dataset import make_labels  # noqa: F401 (re-exported)


def set_base_trainable(model: nn.Module, trainable: bool):
    for p in model.base.parameters():
        p.requires_grad_(trainable)


def get_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.1,
    )


def train_phase(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    warmup_steps: int = 50,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    pad_token_id: Optional[int] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = get_optimizer(model, lr)
    total_steps = max(1, epochs * len(loader))
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=min(warmup_steps, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )
    model.train()
    for epoch in range(epochs):
        running = 0.0
        pbar = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = make_labels(input_ids, pad_token_id)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}"
            )
        avg_loss = running / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} complete — avg loss: {avg_loss:.4f}")
