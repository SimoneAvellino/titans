import json
from src.datasets.data_classes import CantoModRecord


class DivinaCommediaPromptBuilder:

    def __init__(self, divina_commedia_og_path: str):
        self.canti_og = []
        with open(divina_commedia_og_path, "r", encoding="utf-8") as f:
            for line in f:
                self.canti_og.append(json.loads(line))

        self.canto_to_idx = {
            canto["canto_header"]: idx for idx, canto in enumerate(self.canti_og)
        }

    def build_long_context(
        self, canto_mod: CantoModRecord, num_preceding: int, num_following: int
    ) -> str:
        """
        Constructs a long context string by embedding a modified canto between
        original cantos.

        Args:
            canto_mod (CantoModRecord): The modified canto record to act as the target context.
            num_preceding (int): Number of original cantos to include before the modified one.
            num_following (int): Number of original cantos to include after the modified one.

        Returns:
            str: The concatenated long context text.
        """
        idx_mod = self.canto_to_idx[canto_mod.canto_header]
        idx_start = max(0, idx_mod - num_preceding)
        idx_end = min(len(self.canti_og), idx_mod + num_following + 1)
        context_preceding = self.canti_og[idx_start:idx_mod]
        context_following = self.canti_og[idx_mod + 1 : idx_end]
        context = ""

        for canto in context_preceding:
            context += f"{canto['canto_header']}:\n{canto['text']}\n\n"
        context += f"{canto_mod.canto_header}:\n{canto_mod.modified_text}\n\n"
        for canto in context_following:
            context += f"{canto['canto_header']}:\n{canto['text']}\n\n"
        return context
