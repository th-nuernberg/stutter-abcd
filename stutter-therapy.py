import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import random
import re
import time
import json
import os
import datetime
from collections import Counter

random_number = random.randint(3, 7)

# Configuration
model_name = "LeoLM/leo-hessianai-70b-chat"
stutter_prob = 0.15
max_new_tokens = 120
temperature = 0.8
top_p = 0.9
repetition_penalty = 1.1
num_rounds = random_number
context_window = 4

UNWANTED_WORDS = [
    'hallo', 'hello', 'hi ', 'hey ', 'guten tag', 'guten morgen', 'guten abend',
    'assistant', 'user', 'assistent', 'benutzer', 'ki', 'ai', 'sprachmodell',
    'modell', 'bot', 'chatbot', 'ich bin ein', 'ich bin eine', 'willkommen'
]

# Ursprüngliche System-Prompts
STUTTER_AI_SYSTEM_PROMPT = """<|im_start|>system
Du bist ein interessierter Kunde in einer Bäckerei. Du sprichst mit dem Bäcker. Antworte auf die Frage des Bäckers und stelle danach eine Frage. Wenn der Bäcker keine Frage stellt, beziehe sich auf seine Aussage als Kunde. Spreche maximal drei Sätze.<|im_end|>
"""

NORMAL_AI_SYSTEM_PROMPT = """<|im_start|>system
Du bist ein Bäcker in einer Bäckerei. Du bedienst Kunden an der Theke. Antworte auf die Frage des Kunden und stelle danach eine Frage, um den Kunden besser zu beraten. Spreche maximal drei Sätze.<|im_end|>
"""

# Metadata collection
metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "model_used": model_name,
    "ai_behavior_commands": {
        "stutter_ai_system_prompt": STUTTER_AI_SYSTEM_PROMPT,
        "normal_ai_system_prompt": NORMAL_AI_SYSTEM_PROMPT
    },
    "parameters": {
        "stutter_probability": stutter_prob,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "conversation_rounds": num_rounds,
        "context_window": context_window
    },
    "system_info": {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else 0
    },
    "stutter_statistics": {
        "total_words": 0,
        "stuttered_words": 0,
        "stutter_rate": 0.0,
        "stutter_types": Counter(),
        "stutter_patterns": []
    },
    "conversation_metrics": {
        "total_messages": 0,
        "average_response_length": 0,
        "response_times": [],
        "conversation_duration": 0
    },
    "conversation_data": []
}

print(f"Anzahl der Dialogrunden: {random_number}")
print("")

print("Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Lade 70B Modell mit Quantisierung...")
model_load_start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)
model_load_time = time.time() - model_load_start

metadata["model_load_time_seconds"] = round(model_load_time, 2)
metadata["quantization_used"] = True
metadata["quantization_config"] = "8bit"

print("Modell geladen!")

def add_stutter(text, stutter_prob=stutter_prob):
    words = text.split()
    stuttered_text = []
    stutter_count = 0
    current_stutter_details = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        metadata["stutter_statistics"]["total_words"] += 1
        
        if len(clean_word) > 2 and random.random() < stutter_prob:
            stutter_type = random.choice(['repeat', 'partial', 'block'])
            metadata["stutter_statistics"]["stutter_types"][stutter_type] += 1
            stutter_count += 1
            
            if stutter_type == 'repeat' and len(clean_word) > 1:
                repeat_part = clean_word[:random.randint(1, min(2, len(clean_word)//2))]
                stuttered_word = f"{repeat_part}-{repeat_part}-{word}"
                pattern = f"repeat_{repeat_part}"
                metadata["stutter_statistics"]["stutter_patterns"].append(pattern)
                current_stutter_details.append({
                    "original_word": word,
                    "stuttered_word": stuttered_word,
                    "type": "repeat",
                    "pattern": pattern
                })
            elif stutter_type == 'partial':
                if len(clean_word) >= 3:
                    partial = clean_word[:random.randint(1, 2)]
                    stuttered_word = f"{partial}... {word}"
                    pattern = f"partial_{partial}"
                    metadata["stutter_statistics"]["stutter_patterns"].append(pattern)
                    current_stutter_details.append({
                        "original_word": word,
                        "stuttered_word": stuttered_word,
                        "type": "partial",
                        "pattern": pattern
                    })
                else:
                    stuttered_word = word
            else:
                stuttered_word = f"... {word}"
                pattern = "block"
                metadata["stutter_statistics"]["stutter_patterns"].append(pattern)
                current_stutter_details.append({
                    "original_word": word,
                    "stuttered_word": stuttered_word,
                    "type": "block",
                    "pattern": pattern
                })
            
            stuttered_text.append(stuttered_word)
        else:
            stuttered_text.append(word)
    
    metadata["stutter_statistics"]["stuttered_words"] += stutter_count
    return ' '.join(stuttered_text), current_stutter_details

def remove_stutter(text):
    text = re.sub(r'(\w+)-\1-\1', r'\1', text)
    text = re.sub(r'(\w+)\.\.\.\s*', '', text)
    text = re.sub(r'\.\.\.', '', text)
    text = re.sub(r'(\w)-\1', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_unwanted_words(text, message_count, is_stuttering=False):
    original_text = text
    text_lower = text.lower()
    
    if is_stuttering and message_count <= 2:
        pass 
    else:
        for word in UNWANTED_WORDS:
            if text_lower.startswith(word):
                pattern = r'^' + re.escape(word) + r'\s*[,.!?]*\s*'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                text_lower = text.lower()
    
    for word in ['assistant', 'user', 'assistent', 'benutzer', 'ki', 'ai', 'sprachmodell', 'modell', 'bot', 'chatbot']:
        if word in text_lower:
            pattern = r'\b' + re.escape(word) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,.!?]\s*', '', text)
    
    if len(text.strip()) < 5:
        return original_text
    
    return text

def get_conversation_context(conversation_history, current_speaker, context_size=context_window):
    other_speaker = "normal_ai" if current_speaker == "stuttering_ai" else "stuttering_ai"
    
    relevant_messages = []
    for msg in reversed(conversation_history):
        if msg["speaker"] == other_speaker:
            relevant_messages.append(msg)
        if len(relevant_messages) >= context_size:
            break
    
    relevant_messages.reverse()
    return relevant_messages

def generate_response(conversation_history, system_prompt, is_stuttering=False, max_new_tokens=max_new_tokens, temperature=temperature):
    response_start = time.time()
    
    current_speaker = "stuttering_ai" if is_stuttering else "normal_ai"
    context_messages = get_conversation_context(conversation_history, current_speaker)
    
    # Formatierung im ursprünglichen Stil
    formatted_prompt = system_prompt
    
    if context_messages:
        for msg in context_messages:
            speaker_name = "Bäcker" if msg["speaker"] == "normal_ai" else "Kunde"
            formatted_prompt += f"<|im_start|>user\n{speaker_name}: {msg['message']}<|im_end|>\n"
    
    last_message = conversation_history[-1]["message"] if conversation_history else ""
    speaker_name = "Bäcker" if conversation_history[-1]["speaker"] == "normal_ai" else "Kunde" if conversation_history else ""
    formatted_prompt += f"<|im_start|>user\n{speaker_name}: {last_message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    
    response_time = time.time() - response_start
    metadata["conversation_metrics"]["response_times"].append(response_time)
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_response = full_text.split("assistant")[-1].strip()
    
    filter_steps = {
        "raw_response": raw_response,
        "after_basic_cleaning": "",
        "after_word_filtering": ""
    }
    
    step1_response = re.sub(r'^(assistant|user)\s*', '', raw_response, flags=re.IGNORECASE).strip()
    step1_response = re.sub(r'<\|.*?\|>', '', step1_response).strip()
    filter_steps["after_basic_cleaning"] = step1_response
    
    total_messages_so_far = len(conversation_history)
    final_response = filter_unwanted_words(step1_response, total_messages_so_far, is_stuttering)
    filter_steps["after_word_filtering"] = final_response
    
    if is_stuttering:
        stuttered_answer, stutter_details = add_stutter(final_response)
        return stuttered_answer, {
            "system_prompt_used": system_prompt,
            "full_prompt_used": formatted_prompt,
            "user_input": last_message,
            "clean_response": final_response,
            "stuttered_response": stuttered_answer,
            "response_time_seconds": round(response_time, 2),
            "stutter_details": stutter_details,
            "context_messages": [msg["message"] for msg in context_messages],
            "generation_parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            },
            "ai_type": "stuttering_ai",
            "filter_processing_steps": filter_steps
        }
    else:
        return final_response, {
            "system_prompt_used": system_prompt,
            "full_prompt_used": formatted_prompt,
            "user_input": last_message,
            "clean_response": final_response,
            "response_time_seconds": round(response_time, 2),
            "context_messages": [msg["message"] for msg in context_messages],
            "generation_parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            },
            "ai_type": "normal_ai",
            "filter_processing_steps": filter_steps
        }

def simulate_typing(text, is_stutter=False):
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)

# Main - Ursprüngliche Struktur
print("\n" + "="*70)
print("KI UNTERHALTUNG: BÄCKEREI - BÄCKER FRAGT ZUERST, KUNDE STOTTERT")
print("="*70)
print("Bäcker  |  Stotternder Kunde\n")
print(f"Kontext-Fenster: Letzte {context_window} Nachrichten werden berücksichtigt\n")

conversation_start = time.time()

# Start-Prompt wie im Original
start_message = "Guten Morgen! Was hätten Sie gerne?"

conversation_history = [{
    "speaker": "normal_ai",
    "message": start_message
}]

print(f"Bäcker: {start_message}")

# Erste Kundenantwort generieren (mit Stottern)
response_stutter, stutter_metadata = generate_response(
    conversation_history, 
    STUTTER_AI_SYSTEM_PROMPT, 
    is_stuttering=True
)

print(f"Kunde: {response_stutter}")

conversation_history.append({
    "speaker": "stuttering_ai",
    "message": response_stutter
})

metadata["conversation_data"].append({
    "round": 0,
    "speaker": "normal_ai",
    "message": start_message,
    "clean_message": start_message,
    "system_prompt_used": "initial_message",
    "full_prompt_used": "initial_message",
    "user_input": "none",
    "response_time_seconds": 0,
    "is_stuttered": False,
    "ai_type": "normal_ai",
    "filter_processing_steps": {
        "raw_response": start_message,
        "after_basic_cleaning": start_message,
        "after_word_filtering": start_message
    }
})

metadata["conversation_data"].append({
    "round": 1,
    "speaker": "stuttering_ai",
    "message": response_stutter,
    "clean_message": stutter_metadata["clean_response"],
    "system_prompt_used": stutter_metadata["system_prompt_used"],
    "full_prompt_used": stutter_metadata["full_prompt_used"],
    "user_input": stutter_metadata["user_input"],
    "response_time_seconds": stutter_metadata["response_time_seconds"],
    "is_stuttered": True,
    "stutter_details": stutter_metadata["stutter_details"],
    "context_messages": stutter_metadata["context_messages"],
    "generation_parameters": stutter_metadata["generation_parameters"],
    "ai_type": "stuttering_ai",
    "filter_processing_steps": stutter_metadata["filter_processing_steps"]
})

# Dialog-Schleife wie im Original
for round_num in range(num_rounds):
    print(f"\nRound: {round_num + 1}")
    
    # Bäcker-Antwort
    response_normal, normal_metadata = generate_response(
        conversation_history,
        NORMAL_AI_SYSTEM_PROMPT, 
        is_stuttering=False
    )
    print(f"Bäcker: {response_normal}")
    
    conversation_history.append({
        "speaker": "normal_ai",
        "message": response_normal
    })
    
    metadata["conversation_data"].append({
        "round": round_num + 2,
        "speaker": "normal_ai", 
        "message": response_normal,
        "clean_message": normal_metadata["clean_response"],
        "system_prompt_used": normal_metadata["system_prompt_used"],
        "full_prompt_used": normal_metadata["full_prompt_used"],
        "user_input": normal_metadata["user_input"],
        "response_time_seconds": normal_metadata["response_time_seconds"],
        "is_stuttered": False,
        "context_messages": normal_metadata["context_messages"],
        "generation_parameters": normal_metadata["generation_parameters"],
        "ai_type": "normal_ai",
        "filter_processing_steps": normal_metadata["filter_processing_steps"]
    })
    
    if any(word in response_normal.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break
    
    # Kundenantwort
    response_stutter, stutter_metadata = generate_response(
        conversation_history, 
        STUTTER_AI_SYSTEM_PROMPT, 
        is_stuttering=True
    )
    print(f"Kunde: {response_stutter}")
    
    conversation_history.append({
        "speaker": "stuttering_ai",
        "message": response_stutter
    })
    
    metadata["conversation_data"].append({
        "round": round_num + 2,
        "speaker": "stuttering_ai",
        "message": response_stutter,
        "clean_message": stutter_metadata["clean_response"],
        "system_prompt_used": stutter_metadata["system_prompt_used"],
        "full_prompt_used": stutter_metadata["full_prompt_used"],
        "user_input": stutter_metadata["user_input"],
        "response_time_seconds": stutter_metadata["response_time_seconds"],
        "is_stuttered": True,
        "stutter_details": stutter_metadata["stutter_details"],
        "context_messages": stutter_metadata["context_messages"],
        "generation_parameters": stutter_metadata["generation_parameters"],
        "ai_type": "stuttering_ai",
        "filter_processing_steps": stutter_metadata["filter_processing_steps"]
    })

    if any(word in response_stutter.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break

# Abschluss wie im Original
if not any(word in metadata["conversation_data"][-1]["message"].lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
    print(f"\nKunde: V-V-Vielen Dank! A-A-Auf Wiedersehen!")
    print(f"Bäcker: Gerne wieder! Einen schönen Tag noch!")

print("\nEnde des Dialogs")

# Metadaten speichern (wie im Original)
conversation_duration = time.time() - conversation_start
metadata["conversation_metrics"]["conversation_duration"] = round(conversation_duration, 2)
metadata["conversation_metrics"]["total_messages"] = len(metadata["conversation_data"])

if metadata["stutter_statistics"]["total_words"] > 0:
    metadata["stutter_statistics"]["stutter_rate"] = round(
        metadata["stutter_statistics"]["stuttered_words"] / metadata["stutter_statistics"]["total_words"] * 100, 2
    )

if metadata["conversation_metrics"]["response_times"]:
    metadata["conversation_metrics"]["average_response_time"] = round(
        sum(metadata["conversation_metrics"]["response_times"]) / len(metadata["conversation_metrics"]["response_times"]), 2
    )

response_lengths = []
for msg in metadata["conversation_data"]:
    if "clean_message" in msg:
        response_lengths.append(len(msg["clean_message"].split()))
    else:
        response_lengths.append(len(msg["message"].split()))

if response_lengths:
    metadata["conversation_metrics"]["average_response_length"] = round(sum(response_lengths) / len(response_lengths), 1)
    metadata["conversation_metrics"]["min_response_length"] = min(response_lengths)
    metadata["conversation_metrics"]["max_response_length"] = max(response_lengths)

json_path = '/home/neuendankbe92700/stutter-abcd/complete_metadata_with_prompts.json'

if os.path.exists(json_path):
    with open(json_path, "r", encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            try:
                all_data = json.loads(content)
            except json.JSONDecodeError:
                all_data = []
        else:
            all_data = []
else:
    all_data = []

all_data.append(metadata)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\nKomplette Metadaten gespeichert in: {json_path}")

# Zusammenfassung anzeigen
print("\n" + "="*70)
print("METADATEN ZUSAMMENFASSUNG:")
print("="*70)
print(f"Gesprächsdauer: {metadata['conversation_metrics']['conversation_duration']}s")
print(f"Nachrichten gesamt: {metadata['conversation_metrics']['total_messages']}")
if 'average_response_time' in metadata['conversation_metrics']:
    print(f"Durchschnittliche Antwortzeit: {metadata['conversation_metrics']['average_response_time']}s")
print(f"Stotter-Rate: {metadata['stutter_statistics']['stutter_rate']}%")
print(f"Stotter-Typen: {dict(metadata['stutter_statistics']['stutter_types'])}")
print(f"Kontext-Fenster: {context_window} Nachrichten")
print("="*70)