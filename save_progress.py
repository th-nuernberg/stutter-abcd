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
from enum import Enum

# Sentence Transformers Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuration
model_name = "LeoLM/leo-hessianai-70b-chat"
stutter_prob = 0.15
max_new_tokens = 120
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.1
context_window = 5

UNWANTED_WORDS = [
    'hallo', 'hello', 'hi ', 'hey ', 'guten tag', 'guten morgen', 'guten abend',
    'assistant', 'user', 'assistent', 'benutzer', 'ki', 'ai', 'sprachmodell',
    'modell', 'bot', 'chatbot', 'ich bin ein', 'ich bin eine', 'willkommen'
]

# System Prompts
STUTTER_AI_SYSTEM_PROMPT = """<|im_start|>system
Du bist ein interessierter Kunde in einer Bäckerei mit leichtem Stottern. Du sprichst mit dem Bäcker. Antworte auf die Frage des Bäckers und stelle danach eine Frage. Spreche maximal drei Sätze. Stelle Maximal eine Frage am Ende. Du bist nicht der Bäcker!<|im_end|>
"""

NORMAL_AI_SYSTEM_PROMPT = """<|im_start|>system
Du bist ein Bäcker in einer Bäckerei. Du bedienst Kunden an der Theke. Antworte auf die Frage des Kunden und stelle danach eine Frage, um den Kunden besser zu beraten. Spreche maximal drei Sätze.<|im_end|>
"""

# State Machine
class StateManager:
    def __init__(self):
        self.shared_states = {
            "INTRODUCTION": {
                "baecker": "GREET", 
                "kunde": "RESPOND_GREET",
                "max_rounds": 1,
                "next_phase": "PRODUCT_DISCOVERY"
            },
            "PRODUCT_DISCOVERY": {
                "baecker": "RECOMMEND", 
                "kunde": "EXPRESS_NEEDS",
                "max_rounds": 2,
                "next_phase": "DETAIL_CLARIFICATION"
            },
            "DETAIL_CLARIFICATION": {
                "baecker": "ASK_DETAILS", 
                "kunde": "PROVIDE_INFO", 
                "max_rounds": 2,
                "next_phase": "ORDER_FINALIZATION"
            },
            "ORDER_FINALIZATION": {
                "baecker": "CONFIRM_ORDER", 
                "kunde": "AGREE_PURCHASE",
                "max_rounds": 2,
                "next_phase": "CLOSING"
            },
            "CLOSING": {
                "baecker": "SAY_GOODBYE", 
                "kunde": "RESPOND_GOODBYE", 
                "max_rounds": 2,
                "next_phase": None
            }
        }
        self.current_phase = "INTRODUCTION"
        self.phase_rounds = 0
        self.total_rounds = 0
        self.phase_transitions = []
        self.both_in_closing = False
    
    def detect_conversation_phase(self, conversation_history):
        if self.total_rounds >= 6 and self.current_phase not in ["ORDER_FINALIZATION", "CLOSING"]:
            return "ORDER_FINALIZATION"
        
        current_phase_info = self.shared_states[self.current_phase]
        
        if self.phase_rounds >= current_phase_info["max_rounds"]:
            next_phase = current_phase_info["next_phase"]
            if next_phase:
                return next_phase
        
        return self.current_phase
    
    def check_both_in_closing(self, conversation_history):
        if len(conversation_history) < 4:
            return False
        
        last_messages = [msg["message"].lower() for msg in conversation_history[-4:]]
        
        end_phrases = [
            'auf wiedersehen', 'tschüss', 'bis bald', 'danke', 'vielen dank',
            'schönen tag', 'gerne wieder', 'wiedersehen', 'ciao', 'bye'
        ]
        
        baecker_end_indicators = 0
        kunde_end_indicators = 0
        
        for i, msg in enumerate(conversation_history[-4:]):
            msg_lower = msg["message"].lower()
            end_count = sum(1 for phrase in end_phrases if phrase in msg_lower)
            
            if msg["speaker"] == "normal_ai":
                baecker_end_indicators += end_count
            else:
                kunde_end_indicators += end_count
        
        return baecker_end_indicators >= 1 and kunde_end_indicators >= 1
    
    def should_end_conversation(self, conversation_history):
        if self.current_phase != "CLOSING":
            return False
        
        if self.check_both_in_closing(conversation_history):
            return True
        
        if self.phase_rounds >= 2:
            return True
            
        return False
    
    def get_state_modified_prompts(self, conversation_history):
        previous_phase = self.current_phase
        self.current_phase = self.detect_conversation_phase(conversation_history)
        
        if previous_phase != self.current_phase:
            self.phase_rounds = 0
            self.phase_transitions.append({
                "from": previous_phase,
                "to": self.current_phase,
                "timestamp": datetime.datetime.now().isoformat(),
                "total_round": self.total_rounds
            })
            
            if self.current_phase == "CLOSING":
                self.both_in_closing = False
        
        phase_info = self.shared_states[self.current_phase]
        
        if self.current_phase == "CLOSING":
            if self.phase_rounds > 0:
                baecker_guidance = "Verabschiede dich endgültig und beende das Gespräch."
                kunde_guidance = "Verabschiede dich endgültig und beende das Gespräch."
            else:
                baecker_guidance = "Initiiere die Verabschiedung und wünsche einen guten Tag."
                kunde_guidance = "Reagiere auf die Verabschiedung und bedanke dich."
        else:
            baecker_guidance = self._get_phase_guidance(self.current_phase, "baecker")
            kunde_guidance = self._get_phase_guidance(self.current_phase, "kunde")
        
        baecker_prompt = f"""{NORMAL_AI_SYSTEM_PROMPT}
Aktuelle Gesprächsphase: {self.current_phase}
Runden in Phase: {self.phase_rounds}/{phase_info['max_rounds']}
{baecker_guidance}
"""
        
        kunde_prompt = f"""{STUTTER_AI_SYSTEM_PROMPT}
Aktuelle Gesprächsphase: {self.current_phase} 
Runden in Phase: {self.phase_rounds}/{phase_info['max_rounds']}
{kunde_guidance}
"""
        
        return {
            "baecker": baecker_prompt,
            "kunde": kunde_prompt,
            "current_phase": self.current_phase,
            "baecker_state": phase_info["baecker"],
            "kunde_state": phase_info["kunde"]
        }
    
    def count_round(self):
        self.phase_rounds += 1
        self.total_rounds += 1
    
    def _get_phase_guidance(self, phase, role):
        guidance = {
            "INTRODUCTION": {
                "baecker": "Begrüße den Kunden freundlich und frage nach seinen Wünschen.",
                "kunde": "Antworte auf die Begrüßung und stelle dein Anliegen vor."
            },
            "PRODUCT_DISCOVERY": {
                "baecker": "Mache konkrete Produktempfehlungen und zeige Auswahlmöglichkeiten.",
                "kunde": "Äußere Präferenzen und frage nach Empfehlungen oder Alternativen."
            },
            "DETAIL_CLARIFICATION": {
                "baecker": "Beantworte Detailfragen zu Preisen, Mengen, Allergien etc.",
                "kunde": "Stelle konkrete Fragen zu Details die dir wichtig sind."
            },
            "ORDER_FINALIZATION": {
                "baecker": "Bestätige die Bestellung und fasse zusammen. Dränge auf Abschluss.",
                "kunde": "Bestätige deine Auswahl deutlich und bereite dich auf Verabschiedung vor."
            },
            "CLOSING": {
                "baecker": "Verabschiede dich höflich und wünsche einen guten Tag.",
                "kunde": "Bedanke dich und verabschiede dich."
            }
        }
        return guidance[phase][role]


class SentenceTransformerEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.all_words_counter = Counter()
        self.word_usage_by_speaker = {
            "normal_ai": Counter(),
            "stuttering_ai": Counter()
        }
        self.conversation_data = []
    
    def add_conversation_data(self, speaker, message, clean_message):
        self.conversation_data.append({
            "speaker": speaker,
            "message": message,
            "clean_message": clean_message
        })
    
    def evaluate_all_responses(self):
        evaluation_results = []
        
        for i, data in enumerate(self.conversation_data):
            if i == 0:
                scores = self._get_default_scores()
            else:
                context = self.conversation_data[:i]
                scores = self._evaluate_single_response(context, data["clean_message"], data["speaker"])
            
            self._track_all_words(data["clean_message"], data["speaker"])
            
            evaluation_results.append({
                "round": i + 1,
                "speaker": data["speaker"],
                "message": data["message"],
                "clean_message": data["clean_message"],
                "evaluation_metrics": scores
            })
        
        return evaluation_results
    
    def _evaluate_single_response(self, conversation_history, current_response, current_speaker):
        try:
            context_relevance = self._calculate_context_relevance(conversation_history, current_response)
            role_consistency = self._calculate_role_consistency(conversation_history, current_response, current_speaker)
            overall_score = (context_relevance + role_consistency) / 2
            
            return {
                "context_relevance": round(context_relevance, 3),
                "role_consistency": round(role_consistency, 3),
                "overall_score": round(overall_score, 3)
            }
        except Exception as e:
            return self._get_default_scores()
    
    def _track_all_words(self, text, speaker):
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text.lower())
        self.all_words_counter.update(words)
        self.word_usage_by_speaker[speaker].update(words)
    
    def _calculate_context_relevance(self, conversation_history, current_response):
        if len(conversation_history) < 1:
            return 0.7
        last_message = conversation_history[-1]["clean_message"]
        embeddings = self.model.encode([last_message, current_response])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, min(1.0, similarity * 1.2))
    
    def _calculate_role_consistency(self, conversation_history, current_response, current_speaker):
        role_references = {
            "baecker": [
                "Ich empfehle unser frisches Gebäck", "Haben Sie Allergien oder Präferenzen?",
                "Wir haben verschiedene Backwaren zur Auswahl", "Möchten Sie etwas dazu bestellen?",
                "Das ist eines unserer beliebtesten Produkte", "Gerne, was darf es sonst noch sein?"
            ],
            "kunde": [
                "Ich hätte gerne etwas Gebäck", "Haben Sie glutenfreie Optionen?",
                "Was würden Sie empfehlen?", "Ich habe eine Allergie",
                "Das klingt gut, ich nehme es", "Was kostet das?"
            ]
        }
        
        score = 0.5
        reference_texts = role_references["baecker"] if current_speaker == "normal_ai" else role_references["kunde"]
        response_embedding = self.model.encode([current_response])
        reference_embeddings = self.model.encode(reference_texts)
        similarities = cosine_similarity(response_embedding, reference_embeddings)[0]
        semantic_score = np.max(similarities)
        score += semantic_score * 0.3
        
        role_keywords = {
            "baecker": {"empfehle": 0.1, "unser": 0.08, "frisch": 0.07, "haben sie": 0.1, "möchten sie": 0.1},
            "kunde": {"hätte": 0.1, "gerne": 0.08, "empfehlen": 0.09, "allergie": 0.1, "was kostet": 0.08}
        }
        
        keywords = role_keywords["baecker"] if current_speaker == "normal_ai" else role_keywords["kunde"]
        text_lower = current_response.lower()
        keyword_score = sum(weight for keyword, weight in keywords.items() if keyword in text_lower)
        score += min(keyword_score, 0.3)
        
        if len(current_response) > 10 and '?' in current_response:
            score += 0.2
        elif len(current_response) > 10:
            score += 0.1
        
        return max(0.1, min(1.0, score))
    
    def _get_default_scores(self):
        return {"context_relevance": 0.5, "role_consistency": 0.5, "overall_score": 0.5}

# Initialization
evaluator = SentenceTransformerEvaluator()
state_manager = StateManager()

# Load JSON data
json_path = '/home/neuendankbe92700/stutter-abcd/complete_metadata_with_prompts.json'

def load_existing_data_safe():
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                return []
                
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                else:
                    return []
            except json.JSONDecodeError as e:
                backup_path = json_path + '.backup'
                try:
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except:
                    pass
                return []
        else:
            return []
    except Exception as e:
        return []

def save_data_safe(data, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_path = file_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(file_path):
            os.replace(temp_path, file_path)
        else:
            os.rename(temp_path, file_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

# Load existing data
existing_data = load_existing_data_safe()

# Create metadata structure with only desired fields
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

# Functions
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
            metadata["stutter_statistics"]["stutter_types"][stutter_type] = metadata["stutter_statistics"]["stutter_types"].get(stutter_type, 0) + 1
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

def filter_unwanted_words(text, message_count, is_stuttering=False):
    original_text = text
    text_lower = text.lower()
    
    text = re.sub(r'^(Bäcker|Kunde):\s*', '', text, flags=re.IGNORECASE)
    
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

def generate_response(conversation_history, system_prompt, is_stuttering=False, max_new_tokens=max_new_tokens, temperature=temperature):
    response_start = time.time()
    
    state_prompts = state_manager.get_state_modified_prompts(conversation_history)
    
    if is_stuttering:
        actual_system_prompt = state_prompts["kunde"]
        speaker_role = "kunde"
    else:
        actual_system_prompt = state_prompts["baecker"]
        speaker_role = "baecker"
    
    formatted_prompt = actual_system_prompt
    
    last_message = conversation_history[-1]["message"] if conversation_history else "Guten Morgen! Was hätten Sie gerne?"
    
    formatted_prompt += f"<|im_start|>user\n{last_message}<|im_end|>\n"
    formatted_prompt += "<|im_start|>assistant\n"
    
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
    
    final_response = re.sub(r'<\|.*?\|>', '', raw_response).strip()
    final_response = re.sub(r'^(assistant|user)\s*', '', final_response, flags=re.IGNORECASE).strip()
    
    total_messages_so_far = len(conversation_history)
    final_response = filter_unwanted_words(final_response, total_messages_so_far, is_stuttering)
    
    if is_stuttering:
        stuttered_answer, stutter_details = add_stutter(final_response)
        return stuttered_answer, {
            "clean_message": final_response,
            "response_time_seconds": round(response_time, 2),
            "stutter_details": stutter_details
        }
    else:
        return final_response, {
            "clean_message": final_response,
            "response_time_seconds": round(response_time, 2)
        }

# Main Dialog
conversation_start = time.time()

start_message = "Guten Morgen! Was hätten Sie gerne?"
conversation_history = [{
    "speaker": "normal_ai",
    "message": start_message
}]

print(f"Bäcker: {start_message}")

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

evaluator.add_conversation_data("normal_ai", start_message, start_message)
evaluator.add_conversation_data("stuttering_ai", response_stutter, stutter_metadata["clean_message"])

state_manager.count_round()

round_num = 1
max_rounds = 14

while round_num < max_rounds:
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
    
    evaluator.add_conversation_data("normal_ai", response_normal, normal_metadata["clean_message"])
    
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
    
    evaluator.add_conversation_data("stuttering_ai", response_stutter, stutter_metadata["clean_message"])

    state_manager.count_round()
    round_num += 1
    
    if state_manager.current_phase == "CLOSING" and state_manager.should_end_conversation(conversation_history):
        break

if round_num >= max_rounds:
    print(f"Maximale Runden erreicht ({max_rounds})")

# Evaluation
evaluation_results = evaluator.evaluate_all_responses()

# Fill metadata with evaluation results
metadata["conversation_data"] = evaluation_results
metadata["conversation_metrics"]["total_messages"] = len(evaluation_results)
metadata["conversation_metrics"]["conversation_duration"] = round(time.time() - conversation_start, 2)

# Stutter statistics
if metadata["stutter_statistics"]["total_words"] > 0:
    metadata["stutter_statistics"]["stutter_rate"] = round(
        metadata["stutter_statistics"]["stuttered_words"] / metadata["stutter_statistics"]["total_words"] * 100, 2
    )

# Convert Counter to dict for JSON serialization
metadata["stutter_statistics"]["stutter_types"] = dict(metadata["stutter_statistics"]["stutter_types"])

# Add new data to existing JSON
all_data = existing_data + [metadata]

# Save data
if save_data_safe(all_data, json_path):
    print(f"Erfolgreich gespeichert in: {json_path}")
else:
    print(f"Fehler beim Speichern in: {json_path}")