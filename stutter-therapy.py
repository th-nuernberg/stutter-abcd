import os
import time
import json
import re
import random
import datetime
from random import randint
from collections import Counter
from huggingface_hub import InferenceClient

max_rounds = 10  # Maximale Runden als Fallback
HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(provider="together", api_key=HF_TOKEN)
json_path = "/home/neuendankbe92700/stutter-abcd/conversation_data.json"

metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "model_used": "openai/gpt-oss-120b",
    "parameters": {
        "stutter_probability": 0.15,
        "max_rounds": max_rounds,
        "temperature": 0.7,
        "max_tokens": 120
    },
    "stutter_statistics": {
        "total_words": 0,
        "stuttered_words": 0,
        "stutter_rate": 0.0,
        "stutter_types": Counter(),
    },
    "conversation_metrics": {
        "total_messages": 0,
        "response_times": [],
        "conversation_duration": 0
    },
    "conversation_data": []
}

system_manager = """
Du bist ein KI-Dialog-Manager in einer Bäckerei.
Du bekommst den bisherigen Gesprächsverlauf und den aktuellen Zustand (state).
Entscheide, was der nächste Zustand sein soll.
Antworte IMMER als JSON im Format:
{"next_state": "...", "agreement": false, "reason": "..."}.
Erlaubte States: greeting, details, offer, agreement, closing.
Wenn der Kunde etwas gekauft oder zugestimmt hat, setze agreement = true.
Wenn das Gespräch natürlich beendet werden kann, setze next_state = "closing".
"""

customer_messages = [
    {"role": "system", "content": (
        "Du bist ein Kunde in einer Bäckerei. Begrüße den Bäcker freundlich "
        "und frag nach Brot. Sprich locker, natürlich, maximal zwei vollständigen Sätze."
    )}
]

baker_messages = [
    {"role": "system", "content": (
        "Du bist ein Bäcker. Sei freundlich, locker und menschlich. "
        "Sprich kurz in zwei vollständigen Sätzen, ohne Listen, Sterne, Klammern, Tabellen, Abkürzungen, Gedankenstriche oder Emojis."
    )}
]

evaluator_messages = [
    {"role": "system", "content": (
        "Du bist ein neutraler Evaluator. Beurteile die Antwort des Kunden und Bäckers."
        "Beurteile den Context Relevance Score, Task Success Rate und Role Concistency Score. "
        "Bewerte auf einer Skala von 1 bis 10, wobei 10 die beste Bewertung ist. "
        "Antworte IMMER nur mit JSON: {context_relevance, task_success, role_consistency}."
        "Begründe kurz warum die Bewertungen so sind."
        "Bewerte beide Rollen einzelnd."
    )}
]

def evaluate_response(stuttered_ai, fluent_ai):
    evaluate_ai = stuttered_ai.choices[0].message.content.strip()
    evaluate_fluent = fluent_ai.choices[0].message.content.strip()
    completion_evaluator = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": evaluator_messages[0]["content"]},
            {"role": "user", "content": f"Stotternde Antwort: {evaluate_ai}\nFlüssige Antwort: {evaluate_fluent}"}
        ],
    )
    raw = completion_evaluator.choices[0].message.content.strip()
    print("📝 Evaluator:", raw)
    return None

def get_baker_prompt(state):
    if state == "greeting":
        return "Begrüße den Kunden freundlich und frag, was er möchte. Maximal zwei Sätze."
    elif state == "details":
        return "Reagiere freundlich auf die Bestellung und frag nach Sorte und Menge. Zwei bis drei Sätze."
    elif state == "offer":
        return "Sag den Preis in einem Satz, bestätige, dass das Brot frisch geschnitten werden kann, und frag, ob noch etwas gewünscht wird."
    elif state == "agreement":
        return "Bestätige die Bestellung kurz. Ein kurzer Satz."
    elif state == "closing":
        return "Verabschiede dich freundlich und wünsche einen schönen Tag. Ein Satz genügt."
    else:
        return "Beende das Gespräch höflich in einem Satz."

def add_stutter(text, stutter_prob=0.15):
    words = text.split()
    stuttered_text = []
    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        metadata["stutter_statistics"]["total_words"] += 1
        if len(clean) > 2 and random.random() < stutter_prob:
            t = random.choice(["repeat", "partial", "block"])
            metadata["stutter_statistics"]["stutter_types"][t] += 1
            metadata["stutter_statistics"]["stuttered_words"] += 1
            if t == "repeat":
                r = clean[:random.randint(1, min(2, len(clean)//2))]
                stuttered_text.append(f"{r}-{r}-{word}")
            elif t == "partial":
                p = clean[:random.randint(1, 2)]
                stuttered_text.append(f"{p}... {word}")
            else:
                stuttered_text.append(f"... {word}")
        else:
            stuttered_text.append(word)
    return " ".join(stuttered_text)

def manager_decide(state, history):
    try:
        convo_text = "\n".join([f"{m['speaker']}: {m['clean_message']}" for m in history[-6:]])
        user_input = f"Aktueller State: {state}\nGespräch:\n{convo_text}"
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_manager},
                {"role": "user", "content": user_input},
            ],
        )
        raw = response.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"next_state": state, "agreement": False, "reason": "Parsing-Fallback"}
    except Exception as e:
        print(f"⚠️ Manager-Fehler: {e}")
        return {"next_state": state, "agreement": False, "reason": "Exception-Fallback"}

def save_conversation_data(metadata, path):
    try:
        data = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append(metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")
        return False

# ==========================
# 🔹 Haupt-Dialog
# ==========================
state = "greeting"
conversation_start = time.time()
round_num = 0
conversation_active = True
closing_round_completed = False  # 🔹 NEU: Tracken ob Closing-Runde durchgeführt wurde

while conversation_active and round_num < max_rounds:
    round_num += 1
    print(f"\n=== Runde {round_num} | Aktueller State: {state} ===")

    baker_messages[0]["content"] = "Du bist ein Bäcker. " + get_baker_prompt(state)

    # Kunde antwortet
    try:
        round_start = time.time()
        completion_customer = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=customer_messages
        )
        answer_customer = completion_customer.choices[0].message.content.strip()
        stuttered = add_stutter(answer_customer)
        print("🧍‍♂️ Kunde:", stuttered)

        duration = time.time() - round_start
        metadata["conversation_data"].append({
            "round": round_num,
            "speaker": "kunde",
            "message": stuttered,
            "clean_message": answer_customer,
            "response_time": round(duration, 2)
        })
        metadata["conversation_metrics"]["response_times"].append(duration)

        baker_messages.append({"role": "user", "content": answer_customer})
        customer_messages.append({"role": "assistant", "content": answer_customer})

    except Exception as e:
        print(f"Fehler bei Kunden-Antwort: {e}")
        continue

    # Bäcker antwortet
    try:
        round_start = time.time()
        completion_baker = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=baker_messages
        )
        answer_baker = completion_baker.choices[0].message.content.strip()
        print("👨‍🍳 Bäcker:", answer_baker)

        duration = time.time() - round_start
        metadata["conversation_data"].append({
            "round": round_num,
            "speaker": "baecker",
            "message": answer_baker,
            "clean_message": answer_baker,
            "response_time": round(duration, 2)
        })
        metadata["conversation_metrics"]["response_times"].append(duration)

        customer_messages.append({"role": "user", "content": answer_baker})
        baker_messages.append({"role": "assistant", "content": answer_baker})

    except Exception as e:
        print(f"Fehler bei Bäcker-Antwort: {e}")
        continue

    try:
        print("Evaluator: ")
        evaluation_result = evaluate_response(completion_customer, completion_baker)
    except Exception as e:
        print(f"Fehler bei Evaluator: {e}")

    # Manager entscheidet über nächsten State
    decision = manager_decide(state, metadata["conversation_data"])
    print("🤖 Manager:", decision)

    state = decision.get("next_state", state)
    
    # 🔹 VERBESSERTE LOGIK: Closing-Runde muss komplett durchgeführt werden
    if state == "closing" and not closing_round_completed:
        print("🎯 Closing-State erreicht - führe Verabschiedung durch")
        # Lasse noch eine Runde für die Verabschiedung laufen
        closing_round_completed = True
    elif state == "closing" and closing_round_completed:
        print("✅ Closing-Runde abgeschlossen - beende Gespräch")
        conversation_active = False
    elif decision.get("agreement", False):
        print("✅ Agreement erreicht - wechsle zu Closing")
        state = "closing"

# 🔹 SICHERSTELLEN, dass Daten gespeichert werden
metadata["conversation_metrics"]["total_messages"] = len(metadata["conversation_data"])
metadata["conversation_metrics"]["conversation_duration"] = round(time.time() - conversation_start, 2)
metadata["conversation_metrics"]["total_rounds"] = round_num
if metadata["stutter_statistics"]["total_words"] > 0:
    metadata["stutter_statistics"]["stutter_rate"] = round(
        metadata["stutter_statistics"]["stuttered_words"] / metadata["stutter_statistics"]["total_words"] * 100, 2
    )
metadata["stutter_statistics"]["stutter_types"] = dict(metadata["stutter_statistics"]["stutter_types"])

# 🔹 VERBESSERTES SPEICHERN mit Fehlerbehandlung
max_retries = 3
for attempt in range(max_retries):
    if save_conversation_data(metadata, json_path):
        print(f"✅ Daten gespeichert in: {json_path}")
        break
    else:
        print(f"❌ Versuch {attempt + 1}/{max_retries} fehlgeschlagen")
        if attempt < max_retries - 1:
            time.sleep(1)  # Warte eine Sekunde vor erneutem Versuch
else:
    print(f"❌ Alle Speicherversuche fehlgeschlagen für: {json_path}")

print("\n=== KONVERSATION BEENDET ===")
print(f"📊 Runden: {round_num}")
print(f"💬 Nachrichten: {metadata['conversation_metrics']['total_messages']}")
print(f"🗣️  Stotter-Rate: {metadata['stutter_statistics']['stutter_rate']}%")
print(f"⏱️  Dauer: {metadata['conversation_metrics']['conversation_duration']}s")
print(f"📈 Stotter-Typen: {metadata['stutter_statistics']['stutter_types']}")
print(f"🔚 Beendet durch: {'Closing-State' if closing_round_completed else 'Maximale Runden'}")