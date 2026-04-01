from typing import Optional

import torch
from torch.utils.data import Dataset


class SimpleTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc.get("attention_mask", None)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {"input_ids": self.input_ids[idx]}
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        return item


def make_labels(input_ids: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
    labels = input_ids.clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels
