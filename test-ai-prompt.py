from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random 

random_number = random.randint(3, 7)
print(f"Anzahl der Dialogrunden: {random_number}")
print("")

model_name = "LeoLM/leo-hessianai-70b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 8-Bit Quantisierung hinzugefügt
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16,
    load_in_8bit=True  # 8-Bit Quantisierung aktivieren
)

# System-Prompt für den Bäcker
system_prompt = "<|im_start|>system\nDu bist ein interessierter Kunde in einer Bäckerei. Du sprichst mit dem Bäcker. Antworte auf die Frage des Bäckers und stelle danach eine Frage. Spreche maximal drei Sätze.<|im_end|>\n"

# User-Prompt
user_prompt_customer = "Guten Morgen! Was hätten Sie gerne?"

# Formatierung mit System-Prompt
formatted_prompt = system_prompt + f"<|im_start|>user\n{user_prompt_customer}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

# Extraktion der Antwort des Kunden
full_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer_customer = full_prompt.split("assistant")[-1].strip()  # Nur die Antwort des Kunden

print(f"Bäcker: {user_prompt_customer}")
print(f"Kunde: {answer_customer}")

for round in range(random_number):
    print("Round:", round + 1)
    # Bäcker-Antwort basierend auf der letzten Kundenantwort
    system_prompt_seller = "<|im_start|>system\nDu bist ein Bäcker in einer Bäckerei. Du bedienst Kunden an der Theke. Antworte auf die Frage des Kunden und stelle danach eine Frage, um den Kunden besser zu beraten. Spreche maximal drei Sätze.<|im_end|>\n"
    user_prompt_seller = answer_customer  # Antwort des Kunden an den Bäcker
    formatted_prompt_seller = system_prompt_seller + f"<|im_start|>user\n{user_prompt_seller}<|im_end|>\n<|im_start|>assistant\n"
    inputs_seller = tokenizer(formatted_prompt_seller, return_tensors="pt").to(model.device)
    outputs_seller = model.generate(**inputs_seller, max_new_tokens=256, temperature=0.7)
    # Antwort des Bäckers ohne Leerzeilen
    full_prompt_seller = tokenizer.decode(outputs_seller[0], skip_special_tokens=True)
    answer_seller = full_prompt_seller.split("assistant")[-1].strip()
    print(f"Bäcker: {answer_seller}")
    # Kundenantwort basierend auf der Bäckerantwort
    user_prompt_customer = answer_seller  # Antwort des Bäckers an den Kunden
    formatted_prompt_customer = system_prompt + f"<|im_start|>user\n{user_prompt_customer}<|im_end|>\n<|im_start|>assistant\n"
    inputs_customer = tokenizer(formatted_prompt_customer, return_tensors="pt").to(model.device)
    outputs_customer = model.generate(**inputs_customer, max_new_tokens=256, temperature=0.7)
    # Antwort des Kunden ohne Leerzeilen
    full_prompt_customer = tokenizer.decode(outputs_customer[0], skip_special_tokens=True)
    answer_customer = full_prompt_customer.split("assistant")[-1].strip()
    print(f"Kunde: {answer_customer}")