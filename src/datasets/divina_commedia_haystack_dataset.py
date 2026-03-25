import json
from typing import TypedDict
from torch.utils.data import Dataset
from src.datasets.divina_commedia_prompt_builder import DivinaCommediaPromptBuilder
from src.datasets.data_classes import CantoModRecord


class DivinaCommediaHaystackSample(TypedDict):
    long_context: str
    canto_mod: CantoModRecord


class DivinaCommediaHaystackDataset(Dataset):

    def __init__(
        self,
        divina_commedia_mod_path: str,
        prompt_builder: DivinaCommediaPromptBuilder,
        num_cantos_to_include: int,
        needle_position: int,
    ):
        """
        Dataset to evaluate long context retrieval (Needle In A Haystack).

        Args:
            divina_commedia_mod_path (str): Path to the modified cantos dataset.
            prompt_builder (DivinaCommediaPromptBuilder): An instance of the prompt builder to construct contexts.
            num_cantos_to_include (int): Number of cantos to include in the context.
            needle_position (int): The position of the modified canto within the context (0-based index).
        """
        if needle_position < 0 or needle_position >= num_cantos_to_include:
            raise ValueError(
                "Needle position must be between 0 and num_cantos_to_include - 1"
            )

        self.prompt_builder = prompt_builder
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

    def __getitem__(self, idx: int) -> DivinaCommediaHaystackSample:
        canto_mod = self.canto_mod_records[idx]
        num_preceding = self.num_cantos_to_include - self.needle_position - 1
        num_following = self.num_cantos_to_include - num_preceding - 1
        long_context = self.prompt_builder.build_long_context(
            canto_mod=canto_mod,
            num_preceding=num_preceding,
            num_following=num_following,
        )

        return {
            "long_context": long_context,
            "canto_mod": canto_mod,
        }


if __name__ == "__main__":
    print("\n\nTesting DivinaCommediaHaystackDataset\n\n")

    prompt_builder_test = DivinaCommediaPromptBuilder(
        divina_commedia_og_path="data/divina_commedia_og.jsonl"
    )
    dataset = DivinaCommediaHaystackDataset(
        divina_commedia_mod_path="data/divina_commedia_mod.jsonl",
        prompt_builder=prompt_builder_test,
        num_cantos_to_include=5,
        needle_position=0,
    )

    sample = dataset[0]

    for key, value in sample.items():
        val_str = str(value)
        print(f"{key}: {val_str[:100]}{'...' if len(val_str) > 100 else ''}")
