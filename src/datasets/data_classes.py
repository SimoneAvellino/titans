from dataclasses import dataclass


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


@dataclass
class CantoOgRecord:
    """
    Represents a single record of the dataset 'divina_commedia_og'
    """

    canto_header: str
    section: str
    text: str
