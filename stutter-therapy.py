import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import random
import re
import time

# Use different models for each AI
model_name_stutter = "LeoLM/leo-hessianai-70b"
model_name_normal = "LeoLM/leo-hessianai-13b"

print("Lade Tokenizer...")
tokenizer_stutter = AutoTokenizer.from_pretrained(model_name_stutter)
tokenizer_normal = AutoTokenizer.from_pretrained(model_name_normal)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Lade Stotternde KI (70B)...")
model_stutter = AutoModelForCausalLM.from_pretrained(
    model_name_stutter,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

print("Lade Normale KI (13B)...")
model_normal = AutoModelForCausalLM.from_pretrained(
    model_name_normal,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)


def add_stutter(text, stutter_prob=0.15):
    """
    Add stuttering to text
    """
    words = text.split()
    stuttered_text = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)

        # Different stutter types
        if len(clean_word) > 2 and random.random() < stutter_prob:
            stutter_type = random.choice(['repeat', 'partial', 'block'])
            
            if stutter_type == 'repeat' and len(clean_word) > 1:
                repeat_part = clean_word[:random.randint(1, min(2, len(clean_word)//2))]
                stuttered_word = f"{repeat_part}-{repeat_part}-{word}"
            elif stutter_type == 'partial':
                if len(clean_word) >= 3:
                    partial = clean_word[:random.randint(1, 2)]
                    stuttered_word = f"{partial}... {word}"
                else:
                    stuttered_word = word
            else:
                stuttered_word = f"... {word}"
            
            stuttered_text.append(stuttered_word)
        else:
            stuttered_text.append(word)
    
    return ' '.join(stuttered_text)

def remove_stutter(text):
    """
    Remove stutter patterns from text to create clean version for normal AI
    """
    # Remove repeated patterns like "w-w-wort" -> "wort"
    text = re.sub(r'(\w+)-\1-\1', r'\1', text)
    # Remove partial patterns like "wo... wort" -> "wort"
    text = re.sub(r'(\w+)\.\.\.\s*', '', text)
    # Remove standalone "..."
    text = re.sub(r'\.\.\.', '', text)
    # Remove single letter repetitions like "w-wort" -> "wort"
    text = re.sub(r'(\w)-\1', r'\1', text)
    
    # Clean up any extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def limit_to_three_sentences(text):
    """
    Limit text to maximum 3 sentences
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take max 3 sentences
    if len(sentences) > 3:
        sentences = sentences[:3]
        limited_text = '. '.join(sentences) + '.'
    else:
        limited_text = text
    
    return limited_text

def generate_stutter_response(prompt, max_new_tokens=120, temperature=0.7):
    """
    Generate response with stuttering using 70B model
    """
    formatted_prompt = f"""Führe ein natürliches Gespräch auf Deutsch über Kalifornien. Antworte nur auf Deutsch und stelle auch Fragen:

{prompt}

Antwort auf Deutsch:"""
    
    inputs = tokenizer_stutter(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model_stutter.device) for key, value in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        outputs = model_stutter.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer_stutter.eos_token_id,
            eos_token_id=tokenizer_stutter.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    full_text = tokenizer_stutter.decode(outputs[0], skip_special_tokens=True)
    
    if "Antwort auf Deutsch:" in full_text:
        answer = full_text.split("Antwort auf Deutsch:")[-1].strip()
    else:
        answer = full_text.replace(formatted_prompt, "").strip()
    
    answer = answer.split('\n')[0]
    answer = limit_to_three_sentences(answer)
    
    # Add stutter
    answer = add_stutter(answer)
    
    return answer

def generate_normal_response(prompt, max_new_tokens=120, temperature=0.7):
    """
    Generate normal response using 13B model
    """
    formatted_prompt = f"""Führe ein natürliches Gespräch auf Deutsch über Kalifornien. Antworte nur auf Deutsch in flüssigem, korrektem Deutsch:

{prompt}

Antwort auf Deutsch:"""
    
    inputs = tokenizer_normal(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model_normal.device) for key, value in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        outputs = model_normal.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer_normal.eos_token_id,
            eos_token_id=tokenizer_normal.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    full_text = tokenizer_normal.decode(outputs[0], skip_special_tokens=True)
    
    if "Antwort auf Deutsch:" in full_text:
        answer = full_text.split("Antwort auf Deutsch:")[-1].strip()
    else:
        answer = full_text.replace(formatted_prompt, "").strip()
    
    answer = answer.split('\n')[0]
    answer = limit_to_three_sentences(answer)
    
    return answer

def simulate_typing(text, is_stutter=False):
    """
    Simulate typing with different speeds
    """
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)
        
        if is_stutter:
            # Stuttering AI has longer irregular pauses
            if random.random() < 0.4:
                pause = random.uniform(0.2, 0.8)
                time.sleep(pause)
            elif random.random() < 0.1:
                # Longer pauses for block stuttering
                time.sleep(1.0)
        else:
            # Normal AI has shorter and more regular pauses
            if random.random() < 0.2:
                pause = random.uniform(0.1, 0.3)
                time.sleep(pause)


print("\n" + "="*70)
print("KI UNTERHALTUNG: NATÜRLICHES GESPRÄCH ÜBER KALIFORNIEN")
print("="*70)
print("Normale KI (13B)  |  Stotternde KI (70B)\n")

conversation_history = []

# Static opening message
start_message = "Hallo! Schön, mit dir zu sprechen. Ich war letztes Jahr in Kalifornien und war total begeistert. Warst du schon mal dort?"
current_topic = start_message

print(f"Normale KI: {start_message}")
conversation_history.append(f"Normale KI: {start_message}")
time.sleep(2)

# Number of conversation rounds (including goodbye)
num_rounds = 8

for round_num in range(num_rounds):
    print(f"\n--- Runde {round_num + 1} ---")
    
    print(f"Stotternde KI: ", end="", flush=True)
    response_stutter = generate_stutter_response(current_topic)
    simulate_typing(response_stutter, is_stutter=True)
    conversation_history.append(f"Stotternde KI: {response_stutter}")
    print()
    time.sleep(1)

    # Check for goodbye
    if any(word in response_stutter.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break
    
    # Remove stutter from the response before giving it to normal AI
    clean_prompt_for_normal = remove_stutter(response_stutter)
    
    print(f"Normale KI: ", end="", flush=True)
    response_normal = generate_normal_response(clean_prompt_for_normal)
    simulate_typing(response_normal, is_stutter=False)
    conversation_history.append(f"Normale KI: {response_normal}")
    print()
    time.sleep(1)
    
    # Check for goodbye
    if any(word in response_normal.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break
    
    # Update topic for next round
    current_topic = response_normal

# Add natural goodbye if not already done
if not any(word in conversation_history[-1].lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
    print(f"\nStotternde KI: ", end="", flush=True)
    goodbye_stutter = "Also... ich m-muss jetzt langsam los. War schön mit dir zu reden! Bis bald!"
    simulate_typing(goodbye_stutter, is_stutter=True)
    conversation_history.append(f"Stotternde KI: {goodbye_stutter}")
    print()
    time.sleep(1)
    
    print(f"Normale KI: ", end="", flush=True)
    goodbye_normal = "Ja, mir hat das Gespräch auch viel Spaß gemacht! Pass auf dich auf und bis zum nächsten Mal!"
    simulate_typing(goodbye_normal, is_stutter=False)
    conversation_history.append(f"Normale KI: {goodbye_normal}")
    print()

# Conversation summary
print("\n" + "="*70)
print("GESPRÄCHSZUSAMMENFASSUNG:")
print("="*70)
for i, line in enumerate(conversation_history, 1):
    speaker = "Normale KI" if "Normale KI" in line else "Stotternde KI"
    message = line.split(": ", 1)[1]
    print(f"{i:2d}. {speaker}: {message}")

print(f"\nGespräch erfolgreich beendet! Dauer: {len(conversation_history)} Nachrichten")
print("="*70)