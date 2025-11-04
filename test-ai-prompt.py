from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Modellname
model_name = "TheBloke/openchat_v3.2_super-GGUF"

# Tokenizer & Modell laden (GPU + fp16)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Mock "client" wie OpenAI API
class LocalClient:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    class chat:
        @staticmethod
        def completions_create(model, messages, max_tokens=200):
            # Wir nehmen die letzte User-Nachricht
            user_message = messages[-1]["content"]
            inputs = tokenizer(user_message, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Rückgabe in OpenAI-ähnlichem Format
            return {"choices": [{"message": {"role": "assistant", "content": answer}}]}

# Client initialisieren
client = LocalClient(model, tokenizer, device)

# Beispielaufruf
completion = client.chat.completions_create(
    model="openchat_v3.2_super",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(completion["choices"][0]["message"]["content"])
