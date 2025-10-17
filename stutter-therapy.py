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

# Configuration
model_name = "LeoLM/leo-hessianai-70b-chat"
stutter_prob = 0.15
max_new_tokens = 120
temperature = 0.8
top_p = 0.9
repetition_penalty = 1.1
num_rounds = 5
context_window = 4

# Unwanted words to be filtered (after the first message)
UNWANTED_WORDS = [
    'hallo', 'hello', 'hi ', 'hey ', 'guten tag', 'guten morgen', 'guten abend',
    'assistant', 'user', 'assistent', 'benutzer', 'ki', 'ai', 'sprachmodell',
    'modell', 'bot', 'chatbot', 'ich bin ein', 'ich bin eine'
]

# AI Behavior Commands
STUTTER_AI_SYSTEM_PROMPT = """You are a friendly conversation partner. Have a natural conversation in German about California. Respond and ask questions about California."""

NORMAL_AI_SYSTEM_PROMPT = """You are a friendly conversation partner. Have a natural conversation in German about California. Respond and ask questions about California."""

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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure 8-bit quantization to reduce memory usage for the 70B model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading 70B model with quantization...")
model_load_start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across available GPUs
    quantization_config=quantization_config,
    torch_dtype=torch.float16  # Use half precision to save memory
)
model_load_time = time.time() - model_load_start

metadata["model_load_time_seconds"] = round(model_load_time, 2)
metadata["quantization_used"] = True
metadata["quantization_config"] = "8bit"

print("Model loaded!")

def add_stutter(text, stutter_prob=stutter_prob):
    """
    Add stuttering effects to text by modifying words with various stutter patterns.
    
    Args:
        text (str): Input text to stutter
        stutter_prob (float): Probability of stuttering for each word
    
    Returns:
        tuple: (stuttered_text, stutter_details) with the modified text and metadata
    """
    words = text.split()
    stuttered_text = []
    stutter_count = 0
    current_stutter_details = []
    
    for word in words:
        # Remove punctuation for stuttering analysis
        clean_word = re.sub(r'[^\w]', '', word)
        metadata["stutter_statistics"]["total_words"] += 1
        
        # Apply stuttering to words longer than 2 characters with given probability
        if len(clean_word) > 2 and random.random() < stutter_prob:
            stutter_type = random.choice(['repeat', 'partial', 'block'])
            metadata["stutter_statistics"]["stutter_types"][stutter_type] += 1
            stutter_count += 1
            
            if stutter_type == 'repeat' and len(clean_word) > 1:
                # Repeat the first 1-2 characters of the word
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
                # Partial word with ellipsis
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
                # Block stutter with ellipsis
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
    """
    Remove stuttering patterns from text to recover clean version.
    
    Args:
        text (str): Text with stuttering patterns
    
    Returns:
        str: Clean text without stuttering
    """
    text = re.sub(r'(\w+)-\1-\1', r'\1', text)  # Remove repeated patterns like "wo-wo-word"
    text = re.sub(r'(\w+)\.\.\.\s*', '', text)  # Remove partial stutters
    text = re.sub(r'\.\.\.', '', text)  # Remove block stutters
    text = re.sub(r'(\w)-\1', r'\1', text)  # Remove any remaining single repeats
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def limit_to_three_sentences(text):
    """
    Limit text to maximum three sentences for more natural conversation flow.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text limited to three sentences
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 3:
        sentences = sentences[:3]
        limited_text = '. '.join(sentences) + '.'
    else:
        limited_text = text
    
    return limited_text

def filter_unwanted_words(text, message_count, is_stuttering=False):
    """
    Filter unwanted words and phrases from the text based on message count and AI type.
    
    Args:
        text (str): Input text to filter
        message_count (int): Current message count in conversation
        is_stuttering (bool): Whether this is from the stuttering AI
    
    Returns:
        str: Filtered text
    """
    original_text = text
    text_lower = text.lower()
    
    # Special rule for stuttering AI: First response can contain greetings
    if is_stuttering and message_count <= 2:
        # For stuttering AI in first response: Allow greetings
        pass  # No filtering for greetings
    else:
        # For all other cases: Remove unwanted words at the beginning
        for word in UNWANTED_WORDS:
            if text_lower.startswith(word):
                # Replace unwanted word at the beginning
                pattern = r'^' + re.escape(word) + r'\s*[,.!]*\s*'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                text_lower = text.lower()
    
    # Always remove unwanted AI-related words (for both AIs)
    for word in ['assistant', 'user', 'assistent', 'benutzer', 'ki', 'ai', 'sprachmodell', 'modell', 'bot', 'chatbot']:
        if word in text_lower:
            pattern = r'\b' + re.escape(word) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation at the beginning
    text = re.sub(r'^[,.!]\s*', '', text)
    
    # If text became too short, use original
    if len(text.strip()) < 5:
        return original_text
    
    return text

def get_conversation_context(conversation_history, current_speaker, context_size=context_window):
    """
    Get the last n messages from the conversation for context.
    
    Args:
        conversation_history (list): List of previous messages
        current_speaker (str): Current speaker identifier
        context_size (int): Number of previous messages to include
    
    Returns:
        list: Relevant context messages
    """
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
    """
    Generate AI response with given parameters and context.
    
    Args:
        conversation_history (list): Previous conversation messages
        system_prompt (str): System prompt for the AI
        is_stuttering (bool): Whether to apply stuttering
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature
    
    Returns:
        tuple: (response_text, metadata_dict)
    """
    response_start = time.time()
    
    current_speaker = "stuttering_ai" if is_stuttering else "normal_ai"
    context_messages = get_conversation_context(conversation_history, current_speaker)
    
    # Build the prompt with system instruction and context
    formatted_prompt = f"{system_prompt}\n\n"
    
    if context_messages:
        formatted_prompt += "Previous conversation:\n"
        for msg in context_messages:
            speaker_name = "Friend" if msg["speaker"] == "normal_ai" else "You"
            formatted_prompt += f"{speaker_name}: {msg['message']}\n"
        formatted_prompt += "\n"
    
    last_message = conversation_history[-1]["message"] if conversation_history else ""
    formatted_prompt += f"Current message: {last_message}\n\n"
    formatted_prompt += "Your response:"
    
    # Tokenize and prepare input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Generate response
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
    
    # Decode and clean response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.replace(formatted_prompt, "").strip()
    
    # Basic cleaning
    answer = re.sub(r'^(assistant|user)\s*', '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'<\|.*?\|>', '', answer).strip()
    
    # Limit to three sentences
    answer = limit_to_three_sentences(answer)
    
    # Filter unwanted words based on message count and AI type
    total_messages_so_far = len(conversation_history)
    answer = filter_unwanted_words(answer, total_messages_so_far, is_stuttering)
    
    # Apply stuttering if needed and return with metadata
    if is_stuttering:
        stuttered_answer, stutter_details = add_stutter(answer)
        return stuttered_answer, {
            "system_prompt_used": system_prompt,
            "full_prompt_used": formatted_prompt,
            "user_input": last_message,
            "clean_response": answer,
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
            "ai_type": "stuttering_ai"
        }
    else:
        return answer, {
            "system_prompt_used": system_prompt,
            "full_prompt_used": formatted_prompt,
            "user_input": last_message,
            "clean_response": answer,
            "response_time_seconds": round(response_time, 2),
            "context_messages": [msg["message"] for msg in context_messages],
            "generation_parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            },
            "ai_type": "normal_ai"
        }

def simulate_typing(text, is_stutter=False):
    """
    Simulate typing effect by printing words with delays.
    
    Args:
        text (str): Text to type out
        is_stutter (bool): Whether to simulate stuttering typing
    """
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)

# Main program
print("\n" + "="*70)
print("AI CONVERSATION: NATURAL DIALOGUE ABOUT CALIFORNIA")
print("="*70)
print("Stuttering AI  |  Normal AI\n")
print(f"Context window: Last {context_window} messages are considered\n")

conversation_start = time.time()

# Start message
start_message = "Hello! Nice to talk to you. I was in California last year and was totally excited. Have you ever been there?"

# Initialize conversation history
conversation_history = [{
    "speaker": "normal_ai",
    "message": start_message
}]

print(f"Normal AI: {start_message}")

# Store first message
metadata["conversation_data"].append({
    "round": 0,
    "speaker": "normal_ai",
    "message": start_message,
    "system_prompt_used": "initial_message",
    "full_prompt_used": "initial_message",
    "user_input": "none",
    "response_time_seconds": 0,
    "is_stuttered": False,
    "ai_type": "normal_ai"
})

# Main conversation loop
for round_num in range(num_rounds):
    print(f"\n--- Round {round_num + 1} ---")
    
    # Stuttering AI responds
    print(f"Stuttering AI: ", end="", flush=True)
    response_stutter, stutter_metadata = generate_response(
        conversation_history, 
        STUTTER_AI_SYSTEM_PROMPT, 
        is_stuttering=True
    )
    simulate_typing(response_stutter, is_stutter=True)
    print()
    
    # Add stuttering response to history
    conversation_history.append({
        "speaker": "stuttering_ai",
        "message": response_stutter
    })
    
    # Store stuttering response
    metadata["conversation_data"].append({
        "round": round_num + 1,
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
        "ai_type": stutter_metadata["ai_type"]
    })

    # Check for goodbye
    if any(word in response_stutter.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break
    
    # Normal AI responds
    print(f"Normal AI: ", end="", flush=True)
    response_normal, normal_metadata = generate_response(
        conversation_history,
        NORMAL_AI_SYSTEM_PROMPT, 
        is_stuttering=False
    )
    simulate_typing(response_normal, is_stutter=False)
    print()
    
    # Add normal response to history
    conversation_history.append({
        "speaker": "normal_ai",
        "message": response_normal
    })
    
    # Store normal response
    metadata["conversation_data"].append({
        "round": round_num + 1,
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
        "ai_type": normal_metadata["ai_type"]
    })
    
    # Check for goodbye
    if any(word in response_normal.lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
        break

# Add goodbye if not already done
if not any(word in metadata["conversation_data"][-1]["message"].lower() for word in ['tschüss', 'auf wiedersehen', 'bis bald', 'ciao', 'bye']):
    print(f"\nStuttering AI: ", end="", flush=True)
    goodbye_stutter = "So... I n-need to go now. It was nice talking to you! See you soon!"
    simulate_typing(goodbye_stutter, is_stutter=True)
    metadata["conversation_data"].append({
        "round": num_rounds + 1,
        "speaker": "stuttering_ai",
        "message": goodbye_stutter,
        "clean_message": "So I need to go now. It was nice talking to you! See you soon!",
        "system_prompt_used": "manual_goodbye",
        "full_prompt_used": "manual_goodbye",
        "user_input": "none",
        "response_time_seconds": 0,
        "is_stuttered": True,
        "ai_type": "stuttering_ai"
    })
    print()
    
    print(f"Normal AI: ", end="", flush=True)
    goodbye_normal = "Yes, I also enjoyed the conversation a lot! Take care and see you next time!"
    simulate_typing(goodbye_normal, is_stutter=False)
    metadata["conversation_data"].append({
        "round": num_rounds + 1,
        "speaker": "normal_ai",
        "message": goodbye_normal,
        "clean_message": goodbye_normal,
        "system_prompt_used": "manual_goodbye",
        "full_prompt_used": "manual_goodbye",
        "user_input": "none",
        "response_time_seconds": 0,
        "is_stuttered": False,
        "ai_type": "normal_ai"
    })
    print()

# Calculate metrics
conversation_duration = time.time() - conversation_start
metadata["conversation_metrics"]["conversation_duration"] = round(conversation_duration, 2)
metadata["conversation_metrics"]["total_messages"] = len(metadata["conversation_data"])

# Calculate stutter statistics
if metadata["stutter_statistics"]["total_words"] > 0:
    metadata["stutter_statistics"]["stutter_rate"] = round(
        metadata["stutter_statistics"]["stuttered_words"] / metadata["stutter_statistics"]["total_words"] * 100, 2
    )

# Calculate response times
if metadata["conversation_metrics"]["response_times"]:
    metadata["conversation_metrics"]["average_response_time"] = round(
        sum(metadata["conversation_metrics"]["response_times"]) / len(metadata["conversation_metrics"]["response_times"]), 2
    )

# Calculate response lengths
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

# Save JSON
json_path = '/home/neuendankbe92700/stutter-abcd/complete_metadata_with_prompts.json'

# Load existing data if available
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

print(f"\nComplete metadata saved in: {json_path}")

# Display summary
print("\n" + "="*70)
print("METADATA SUMMARY:")
print("="*70)
print(f"Conversation duration: {metadata['conversation_metrics']['conversation_duration']}s")
print(f"Total messages: {metadata['conversation_metrics']['total_messages']}")
if 'average_response_time' in metadata['conversation_metrics']:
    print(f"Average response time: {metadata['conversation_metrics']['average_response_time']}s")
print(f"Stutter rate: {metadata['stutter_statistics']['stutter_rate']}%")
print(f"Stutter types: {dict(metadata['stutter_statistics']['stutter_types'])}")
print(f"Context window: {context_window} messages")
print("="*70)

# Display complete conversation
print("\n" + "="*70)
print("COMPLETE CONVERSATION:")
print("="*70)
for i, msg in enumerate(metadata["conversation_data"]):
    speaker = "Normal AI" if msg["speaker"] == "normal_ai" else "Stuttering AI"
    print(f"{i+1:2d}. {speaker}: {msg['message']}")
print("="*70)