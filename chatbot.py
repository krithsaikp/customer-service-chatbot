import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr

# Knowledge Base
documents = [
    "To reset your smart speaker, hold the power button for 10 seconds.",
    "Laptops come with a 1-year warranty covering hardware defects.",
    "If your headphones won't pair, make sure Bluetooth is enabled and hold the sync button for 5 seconds.",
    "You can track your order using the tracking link sent to your email.",
    "For battery issues on laptops, try updating the BIOS and running the battery health tool.",
    "If a phone won't turn on, try a forced restart (hold power + volume down for 10–15 seconds) and then charge for 30 minutes.",
    "Slow phone performance can often be improved by clearing app caches, uninstalling unused apps, and updating the OS.",
    "If a device won't charge, test the cable and adapter with another device and inspect the charging port for debris.",
    "Wi‑Fi connectivity issues: reboot the router, move closer, check SSID/password, and ensure firmware is up to date.",
    "If a router has poor range, try changing its channel, relocating it centrally, or enabling mesh extenders.",
    "Pairing a Bluetooth speaker: enable pairing mode on the speaker, then select it from the phone's Bluetooth settings.",
    "Smart TV no picture: check HDMI cables, select correct input, and try a different HDMI port or source device.",
    "If a printer is offline, restart both printer and PC, check network connection, and reinstall the printer driver if needed.",
    "To improve battery life on wearables, lower screen brightness, reduce notifications, and disable always-on display.",
    "Camera won't focus: clean the lens, ensure sufficient light, and toggle autofocus/manual focus settings.",
    "If an external hard drive is not recognized, try different USB ports, cables, and check Disk Management for partitions.",
    "Overheating devices: close intensive apps, ensure ventilation, avoid direct sunlight, and update firmware/drivers.",
    "If a USB device isn't detected, try a powered USB hub, update USB drivers, and test on another computer.",
    "Updating firmware can fix bugs; always back up data and follow manufacturer's instructions carefully.",
    "If speakers produce static, check cable connections, move away from interference sources, and test with another device.",
    "Screen burn-in prevention: use screen savers, reduce brightness, and enable pixel shift features when available.",
    "If a keyboard or mouse lags, replace batteries (if wireless), reconnect, and check for USB interference.",
    "Factory reset will erase user data; always back up important files before proceeding.",
    "For HDMI audio issues, set the TV/receiver as the default audio device in system sound settings.",
    "Our available devices include smart speakers, laptops, headphones, phones, routers, smart TVs, printers, wearables, cameras, external hard drives, keyboards, and mice.",
    "For software updates, check the manufacturer's website and download the latest version.",
    "If your device is under warranty, contact support for free repairs.",
]

# Chat History
history = []

# Embedding Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

# Build FAISS Index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load FLAN-T5
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model.eval()

# Retrieval and Generation
def answer_query(query, threshold=2.0):
    global history

    # Embed query and retrieve context
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k=2)

    if D[0][0] > threshold:
        retrieved = "No relevant context found."
    else:
        retrieved = "\n".join([documents[i] for i in I[0]])

    # Build conversation memory
    memory_text = ""
    for user_msg, bot_msg in history[-3:]:
        memory_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    prompt = f"""
You are a customer support assistant. Answer respectfully in full sentences, using only the provided context and history.
Always capitalize the first word of each sentence and end with a period. Do not answer in just one word or phrase, elaborate with helpful details.
Always use the best matching context in your response. 

Conversation history:
{memory_text}

Context:
{retrieved}

User question:
{query}

Answer in a full sentence:
"""

    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            min_length=20
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Update memory
    history.append((query, answer))

    return answer


def chat_fn(message, history):
    return answer_query(message)

gr.ChatInterface(
    fn=chat_fn,
    title="Customer Support Assistant",
    description="Ask any support question about our devices."
).launch()
