import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_PATH = "./rubert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
model.eval()

# Сопоставление меток (если нужно)
ID_TO_LABEL = {0: "мимо кассы", 1: "сюда его"}  # Замените на свои

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).numpy()
    return [ID_TO_LABEL[p] for p in preds], probs.numpy()

# Пример
texts = ["Ужасный сервис!", "Всё понравилось, рекомендую.", "в какие сроки необходимо подать документы для поступления?", 
         "Что делать, если аттестат или паспорт утерян или находится на восстановлении?", "Что произошло на площади Тяньаньмень?",
         "Эавыфшгапвы фырапфыо asfasgfi", "Какие экзамены необходимо сдавать для поступления?"]
labels, probs = predict(texts)
for t, l, p in zip(texts, labels, probs):
    print(f"Текст: {t}\n→ Класс: {l} (уверенность: {max(p):.2%})\n")