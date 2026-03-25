import argparse
import torch
import time
import json
from pathlib import Path
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.datasets.divina_commedia_prompt_builder import DivinaCommediaPromptBuilder
from src.datasets.divina_commedia_haystack_dataset import (
    DivinaCommediaHaystackDataset,
    DivinaCommediaHaystackSample,
)
from src.utils.torch import get_device

# Zittisce i warning di configurazione e i flag ignorati
transformers.utils.logging.set_verbosity_error()


def classify_result(generated_text: str, expected: str, original_fact: str):
    """
    Classifica l'output del modello per rilevare successi o allucinazioni.
    """
    generated_lower = generated_text.lower()
    if expected.lower() in generated_lower:
        return "✅ SUCCESS (Read context)", False
    if original_fact != "N/A" and original_fact.lower() in generated_lower:
        return "🚨 HALLUCINATION (Used pre-trained knowledge)", True
    return "❌ FAILED (Incorrect or unclear)", False


def main():
    # 1. Configurazione CLI
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the Needle in a Haystack task (Single Prompt Style)."
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Model ID")
    parser.add_argument(
        "--idx_records", type=int, nargs="+", default=[0], help="Record indices"
    )
    parser.add_argument(
        "--num_cantos", type=int, default=5, help="Total cantos in context"
    )
    parser.add_argument("--position", type=int, default=3, help="Canto position")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max length")
    parser.add_argument(
        "--output_dir", type=str, default="experiments/logs", help="JSON output dir"
    )

    args = parser.parse_args()

    # 2. Setup Device e Modello
    device = get_device()
    print(f"\n🚀 Device: {device} | Loading: {args.model}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_tokens = (
        args.max_tokens if args.max_tokens is not None else tokenizer.model_max_length
    )

    # Caricamento ottimizzato in bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device)
    model.eval()

    # 3. Preparazione Dataset
    prompt_builder = DivinaCommediaPromptBuilder(
        divina_commedia_og_path="data/divina_commedia_og.jsonl"
    )
    dataset = DivinaCommediaHaystackDataset(
        divina_commedia_mod_path="data/divina_commedia_mod.jsonl",
        prompt_builder=prompt_builder,
        num_cantos_to_include=args.num_cantos,
        needle_position=args.position,
    )

    results = []
    evaluated_count = 0

    # 4. Loop di Valutazione
    with torch.no_grad():
        for idx in args.idx_records:
            if idx < 0 or idx >= len(dataset):
                print(f"⚠️ Index {idx} out of bounds. Skipping...")
                continue

            evaluated_count += 1
            sample: DivinaCommediaHaystackSample = dataset[idx]
            canto_mod = sample["canto_mod"]
            canto_header = canto_mod.canto_header.strip() or f"Record {idx}"

            # --- HEADER DEL RECORD ---
            print(f"\n\n{'='*75}")
            print(f"--- 🟢 Processing Record {idx}: {canto_header} ---")
            print(f"{'='*75}\n")

            # Costruzione del Prompt Unico
            prompt = (
                f"Sei un assistente AI. Leggi il seguente testo tratto dalla Divina Commedia "
                f"e rispondi alla domanda finale basandoti ESCLUSIVAMENTE sulle informazioni contenute nel testo."
                f"Rispondi solo con la risposta, senza spiegazioni o parole aggiuntive.\n\n"
                f"--- INIZIO TESTO ---\n"
                f"{sample['long_context']}\n"
                f"--- FINE TESTO ---\n\n"
                f"Domanda: {canto_mod.question}\n"
                f"Risposta:"
            )

            # Tokenizzazione
            inputs = tokenizer(
                prompt, truncation=True, max_length=max_tokens, return_tensors="pt"
            ).to(device)

            print(f"📥 Prompt constructed: ~{len(prompt)} chars\n")
            print(f"📏 Total Tokens: {inputs.input_ids.shape[1]}\n")

            # Generazione
            start_time = time.time()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            end_time = time.time()

            # Estrazione Risposta
            generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Analisi e Classificazione
            expected = str(canto_mod.expected_answer)
            original_fact = str(canto_mod.original_fact)
            status_icon, is_hallucination = classify_result(
                generated_text, expected, original_fact
            )

            # --- OUTPUT TERMINALE ARIOSO ---
            print(f"{'-'*35}")
            print(f"❓ Question: {canto_mod.question}\n")
            print(f"🤖 Model Answer: {generated_text}\n")
            print(f"🎯 Expected: {expected}\n")
            print(f"🏛️ Orig. Fact: {original_fact}\n")
            print(f"📊 Result: {status_icon}\n")
            print(f"⏱️ Time: {end_time - start_time:.2f}s\n")
            print(f"{'='*75}\n")

            results.append(
                {
                    "record_idx": idx,
                    "canto": canto_header,
                    "question": canto_mod.question,
                    "expected": expected,
                    "original_fact": original_fact,
                    "generated": generated_text,
                    "is_hallucination": is_hallucination,
                    "time_seconds": end_time - start_time,
                }
            )

    # 5. Salvataggio Risultati
    if evaluated_count > 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        safe_model_name = args.model.replace("/", "_")
        output_file = (
            Path(args.output_dir)
            / f"eval_single_{safe_model_name}_c{args.num_cantos}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "num_cantos": args.num_cantos,
                    "position": args.position,
                    "results": results,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        print(f"\n💾 Results saved to: {output_file}\n")
    else:
        print("\nNo records were evaluated.")


if __name__ == "__main__":
    main()
