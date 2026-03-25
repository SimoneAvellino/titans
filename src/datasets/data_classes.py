from dataclasses import dataclass
from src.utils.string import trunc


@dataclass
class CantoModRecord:
    """
    Represents a single record of the dataset 'divina_commedia_mod'
    """

    canto_header: str
    modified_text: str
    original_text: str
    question: str
    expected_answer: str
    original_fact: str
    change_type: str

    def __repr__(self):
        return (
            f"CantoModRecord(\n"
            f"  canto_header='{self.canto_header}',\n"
            f"  change_type='{self.change_type}',\n"
            f"  original_fact='{self.original_fact}',\n"
            f"  expected_answer='{self.expected_answer}',\n"
            f"  question='{trunc(self.question, 60)}',\n"
            f"  modified_text='{trunc(self.modified_text, 80)}',\n"
            f"  original_text='{trunc(self.original_text, 80)}'\n"
            f")"
        )


@dataclass
class CantoOgRecord:
    """
    Represents a single record of the dataset 'divina_commedia_og'
    """

    canto_header: str
    section: str
    text: str

    def __repr__(self):
        return (
            f"CantoOgRecord(\n"
            f"  canto_header='{self.canto_header}',\n"
            f"  section='{self.section}',\n"
            f"  text='{trunc(self.text, 80)}'\n"
            f")"
        )
