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

# Zittisce i warning superflui
transformers.utils.logging.set_verbosity_error()


def classify_result(generated_text: str, expected: str, original_fact: str):
    generated_lower = generated_text.lower()
    if expected.lower() in generated_lower:
        return "✅ SUCCESS (Read context)", False
    if original_fact != "N/A" and original_fact.lower() in generated_lower:
        return "🚨 HALLUCINATION (Used pre-trained knowledge)", True
    return "❌ FAILED (Incorrect or unclear)", False


def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Evaluation with Chat Templates"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--idx_records", type=int, nargs="+", default=[0])
    parser.add_argument("--num_cantos", type=int, default=5)
    parser.add_argument("--position", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="experiments/logs")
    args = parser.parse_args()

    device = get_device()
    print(f"\n🚀 Device: {device} | Loading: {args.model}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device)
    model.eval()

    prompt_builder = DivinaCommediaPromptBuilder("data/divina_commedia_og.jsonl")
    dataset = DivinaCommediaHaystackDataset(
        divina_commedia_mod_path="data/divina_commedia_mod.jsonl",
        prompt_builder=prompt_builder,
        num_cantos_to_include=args.num_cantos,
        needle_position=args.position,
    )

    results = []

    with torch.no_grad():
        for idx in args.idx_records:
            if idx < 0 or idx >= len(dataset):
                continue

            sample: DivinaCommediaHaystackSample = dataset[idx]
            canto_mod = sample["canto_mod"]

            # --- HEADER DEL RECORD ---
            print(f"\n\n{'='*70}")
            print(f"--- 🟢 Processing Record {idx}: {canto_mod.canto_header} ---")
            print(f"{'='*70}\n")

            # --- PHASE 1: CONTEXT INJECTION ---
            messages = [
                {
                    "role": "system",
                    "content": "Sei un assistente preciso che si basa solo sui testi forniti.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Leggi attentamente il seguente testo della Divina Commedia. "
                        f"Memorizza ogni dettaglio perché tra poco ti farò una domanda specifica. "
                        f"Se hai capito e sei pronto, rispondi esclusivamente con la parola 'OK'.\n\n"
                        f"--- TESTO ---\n{sample['long_context']}\n--- FINE TESTO ---"
                    ),
                },
            ]

            context_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            print(f"📥 Injecting context: {context_prompt[:120]}...\n")
            print(f"📏 Text Length: {len(sample['long_context'])} chars\n")

            context_inputs = tokenizer(context_prompt, return_tensors="pt").to(device)
            start_pref = time.time()

            context_outputs = model.generate(
                **context_inputs,
                max_new_tokens=10,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            kv_cache = context_outputs.past_key_values
            prefilled_len = context_outputs.sequences.shape[1]

            injection_response_ids = context_outputs.sequences[0][
                context_inputs.input_ids.shape[1] :
            ]
            injection_response_text = tokenizer.decode(
                injection_response_ids, skip_special_tokens=True
            ).strip()

            end_pref = time.time()
            print(f"🤖 Model Response: {injection_response_text}\n")
            print(f"⏱️ Prefill Time: {end_pref - start_pref:.2f}s\n")

            # --- PHASE 2: QUESTION ANSWERING ---
            print(f"{'-'*30}")
            print(f"❓ Question: {canto_mod.question}\n")

            question_messages = [
                {
                    "role": "user",
                    "content": (
                        f"Basandoti SOLO sul testo precedente: {canto_mod.question}"
                        "Rispondi solo con la risposta, senza spiegazioni o parole aggiuntive.\n\n"
                    ),
                }
            ]

            q_template = tokenizer.apply_chat_template(
                question_messages, tokenize=False, add_generation_prompt=True
            )
            q_text_final = q_template.split("<|im_start|>user")[-1]
            q_text_final = "<|im_start|>user" + q_text_final

            q_inputs = tokenizer(
                q_text_final, return_tensors="pt", add_special_tokens=False
            ).to(device)
            q_len = q_inputs.input_ids.shape[1]

            cache_position = torch.arange(
                prefilled_len, prefilled_len + q_len, device=device
            )
            full_attention_mask = torch.ones((1, prefilled_len + q_len), device=device)

            start_gen = time.time()
            gen_outputs = model.generate(
                input_ids=q_inputs.input_ids,
                attention_mask=full_attention_mask,
                past_key_values=kv_cache,
                cache_position=cache_position,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            end_gen = time.time()

            full_decoded = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            if "assistant" in full_decoded:
                generated_text = full_decoded.split("assistant")[-1].strip()
            else:
                generated_text = full_decoded.strip()

            expected = str(canto_mod.expected_answer)
            original_fact = str(canto_mod.original_fact)
            status_icon, is_hallucination = classify_result(
                generated_text, expected, original_fact
            )

            print(f"🤖 Model Answer: {generated_text}\n")
            print(f"🎯 Expected: {expected}\n")
            print(f"🏛️ Orig. Fact: {original_fact}\n")
            print(f"📊 Result: {status_icon}\n")
            print(f"⏱️ Gen Time: {end_gen - start_gen:.2f}s\n")
            print(f"{'='*70}\n")

            results.append(
                {
                    "record_idx": idx,
                    "canto": canto_mod.canto_header,
                    "generated": generated_text,
                    "is_hallucination": is_hallucination,
                    "prefill_time": end_pref - start_pref,
                    "gen_time": end_gen - start_gen,
                }
            )

    if results:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        out_file = (
            Path(args.output_dir)
            / f"kv_chat_eval_{args.model.replace('/', '_')}_c{args.num_cantos}.json"
        )
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()
