#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "TheBloke/LLaMA-70B-GPTQ"  # Beispiel: quantisierte Version für 70B
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Lade das Modell in 4-bit für Speicherersparnis
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    prompt = "Antworte in einem Satz auf die folgende Frage: Wer wurde letztes Jahr Meister in der Bundesliga?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = output_text[len(prompt):].strip()
    
    print("Prompt:\n", prompt)
    print("\nAntwort:\n", answer)

if __name__ == "__main__":
    main()
