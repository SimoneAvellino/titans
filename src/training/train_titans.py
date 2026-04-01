#!/usr/bin/env python
"""
Fine-tune Qwen2 with the Titans MAC architecture.

Usage:
    python train.py [--model MODEL] [--output-dir DIR] [--epochs N]
                    [--batch-size N] [--max-length N] [--lr LR]
                    [--chunk-size N] [--n-persistent N]
                    [--no-linear-memory] [--freeze-base]
                    [--phase1-epochs N] [--phase2-epochs N]

Example:
    python train.py --model Qwen/Qwen2.5-1.5B-Instruct --output-dir experiments/checkpoints/run1
"""

import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.datasets import SimpleTextDataset
from src.models import TitansConfig, build_titans_mac
from src.training.utils import set_base_trainable, train_phase
from src.utils.torch import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Titans MAC fine-tuning")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="experiments/checkpoints/titans")
    p.add_argument("--hf-dataset", default="wikitext")
    p.add_argument("--hf-dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--hf-dataset-split", default="train[:1%]")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--phase1-epochs", type=int, default=1, help="Epochs with frozen base"
    )
    p.add_argument(
        "--phase2-epochs", type=int, default=1, help="Epochs with unfrozen base"
    )
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--n-persistent", type=int, default=16)
    p.add_argument("--memory-depth", type=int, default=2)
    p.add_argument(
        "--no-linear-memory",
        action="store_true",
        help="Use MLP memory instead of linear (Delta Rule)",
    )
    p.add_argument(
        "--freeze-base",
        action="store_true",
        help="Skip phase 2 (keep base frozen throughout)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = TitansConfig(
        chunk_size=args.chunk_size,
        n_persistent=args.n_persistent,
        memory_depth=args.memory_depth,
        use_linear_memory=not args.no_linear_memory,
        freeze_base=True,  # always start frozen; phase 2 unfreezes if requested
        inner_lr=0.01,
    )

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"Loading model '{args.model}' ...")
    model, tokenizer = build_titans_mac(args.model, cfg, dtype=dtype)
    model = model.to(device)

    n_all = sum(p.numel() for p in model.parameters())
    n_new = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_all:,}")
    print(f"Trainable params: {n_new:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset '{args.hf_dataset}' / '{args.hf_dataset_config}' ...")
    hf_ds = load_dataset(
        args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split
    )
    texts = [t for t in hf_ds["text"] if t and t.strip()]
    print(f"Dataset size: {len(texts)} samples")

    dataset = SimpleTextDataset(texts, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ── Phase 1: frozen base ───────────────────────────────────────────────────
    if args.phase1_epochs > 0:
        print("\nPhase 1: training new modules (base frozen) ...")
        set_base_trainable(model, False)
        train_phase(
            model,
            loader,
            epochs=args.phase1_epochs,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
        )
        print("Phase 1 complete.")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    if args.phase2_epochs > 0 and not args.freeze_base:
        print("\nPhase 2: full fine-tuning (base unfrozen) ...")
        set_base_trainable(model, True)
        train_phase(
            model,
            loader,
            epochs=args.phase2_epochs,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
        )
        print("Phase 2 complete.")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "titans_mac.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "model_name": args.model,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to {checkpoint_path}")

    # Also save tokenizer alongside the checkpoint for easy reloading
    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
