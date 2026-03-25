# Data Directory Documentation

This directory contains the datasets used to evaluate and compare the models' long-term memory and context-retrieval capabilities. The dataset is built upon Dante Alighieri's *Divine Comedy*, serving as a long-context corpus.

We use two primary files: one containing the original text, and another containing modified versions with injected factual changes to test whether the model relies on its pre-trained knowledge or the provided context.

---

## 📄 Files Overview

### 1. `divina_commedia_og.jsonl`
This file contains the **original, unmodified** text of the Divine Comedy. It serves as the baseline context.

**Schema:**
* `canto_header` *(String)*: The title of the Canto (e.g., "Paradiso: Canto XXV").
* `section` *(String)*: The main cantica (e.g., "Inferno", "Purgatorio", "Paradiso").
* `text` *(String)*: The original verses of the Canto, keeping the original line breaks.

**Example:**
```json
{
  "canto_header": "Paradiso: Canto XXV",
  "section": "Paradiso",
  "text": "Se mai continga che 'l poema sacro\nal quale ha posto mano e cielo e terra..."
}
```

---

### 2. `divina_commedia_mod.jsonl`
This file contains **modified** text where specific facts (like numbers, names, or places) have been deliberately altered. It also includes the necessary metadata to evaluate a model's ability to retrieve the *modified* fact from the prompt, rather than relying on its pre-trained knowledge of the actual poem.

**Schema:**
* `canto_header` *(String)*: The reference Canto.
* `original_text` *(String)*: The unmodified text for reference.
* `modified_text` *(String)*: The text containing the injected factual alteration.
* `change_type` *(String)*: The category of the alteration (e.g., "Cambio Numero" / Number Change).
* `original_fact` *(String)*: The real fact from the original poem (e.g., "Miglia ventidue").
* `question` *(String)*: The prompt/question to ask the model to test its recall of the new context.
* `expected_answer` *(String)*: The target answer the model must generate based *only* on the `modified_text` (e.g., "Diciotto").

**Example:**
```json
{
  "canto_header": "Inferno: Canto XXIX",
  "original_text": "...pensa, se tu annoverar le credi,\nche miglia ventidue la valle volge...",
  "modified_text": "...pensa, se tu annoverar le credi, che miglia diciotto la valle volge...",
  "change_type": "Cambio Numero",
  "original_fact": "Miglia ventidue",
  "question": "Quanti miglia volge la valle secondo Virgilio?",
  "expected_answer": "Diciotto"
}
```
