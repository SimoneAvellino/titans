import json
from torch.utils.data import Dataset
from src.datasets.divina_commedia_prompt_builder import DivinaCommediaPromptBuilder
from src.datasets.data_classes import CantoModRecord


class DivinaCommediaHaystackDataset(Dataset):

    def __init__(
        self,
        divina_commedia_mod_path: str,
        prompt_builder: DivinaCommediaPromptBuilder,
        tokenizer,
        num_cantos_to_include: int,
        needle_position: int,
        max_tokens: int = 16000,
    ):
        """
        Dataset to evaluate long context retrieval (Needle In A Haystack).

        Args:
            divina_commedia_mod_path (str): Path to the modified cantos dataset.
            prompt_builder (DivinaCommediaPromptBuilder): An instance of the prompt builder to construct contexts.
            tokenizer: The tokenizer to use for encoding the contexts.
            num_cantos_to_include (int): Number of cantos to include in the context.
            needle_position (int): The position of the modified canto within the context (0-based index).
            max_tokens (int): Maximum number of tokens for the context. Defaults to 16000.
        """
        if needle_position < 0 or needle_position >= num_cantos_to_include:
            raise ValueError(
                "Needle position must be between 0 and num_cantos_to_include - 1"
            )

        self.prompt_builder = prompt_builder
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.num_cantos_to_include = num_cantos_to_include
        self.needle_position = needle_position
        self.canto_mod_records: list[CantoModRecord] = []

        with open(divina_commedia_mod_path, "r", encoding="utf-8") as f:
            for line in f:
                record_dict = json.loads(line)
                record = CantoModRecord(**record_dict)
                self.canto_mod_records.append(record)

        if len(self.canto_mod_records) == 0:
            raise ValueError("No records found in the modified cantos dataset")

        if self.needle_position >= len(self.canto_mod_records):
            raise ValueError(
                "Needle position exceeds the number of modified cantos available"
            )

    def __len__(self):
        return len(self.canto_mod_records)

    def __getitem__(self, idx: int):
        canto_mod = self.canto_mod_records[idx]
        num_preceding = self.num_cantos_to_include - self.needle_position - 1
        num_following = self.num_cantos_to_include - num_preceding - 1
        long_context = self.prompt_builder.build_long_context(
            canto_mod=canto_mod,
            num_preceding=num_preceding,
            num_following=num_following,
        )

        prompt = (
            f"Sei un assistente AI. Leggi il seguente testo tratto dalla Divina Commedia "
            f"e rispondi alla domanda finale basandoti ESCLUSIVAMENTE sulle informazioni contenute nel testo.\n\n"
            f"--- INIZIO TESTO ---\n"
            f"{long_context}\n"
            f"--- FINE TESTO ---\n\n"
            f"Domanda: {canto_mod.question}\n"
            f"Risposta:"
        )

        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors=None,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt": prompt,
            "question": canto_mod.question,
            "expected_answer": canto_mod.expected_answer,
            "canto_header": canto_mod.canto_header,
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("\n\nTesting DivinaCommediaHaystackDataset\n\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_builder = DivinaCommediaPromptBuilder(
        divina_commedia_og_path="data/divina_commedia_og.jsonl"
    )
    dataset = DivinaCommediaHaystackDataset(
        divina_commedia_mod_path="data/divina_commedia_mod.jsonl",
        prompt_builder=prompt_builder,
        tokenizer=tokenizer,
        num_cantos_to_include=5,
        needle_position=0,
        max_tokens=16000,
    )

    sample = dataset[0]

    for key, value in sample.items():
        val_str = str(value)
        print(f"{key}: {val_str[:100]}{'...' if len(val_str) > 100 else ''}")
