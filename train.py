import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ================= НАСТРОЙКИ =================
MODEL_NAME = "DeepPavlov/rubert-base-cased"  # Русская BERT
DATA_PATH = "data.csv"                       # Путь к вашему CSV
TEXT_COL = "text"                            # Колонка с текстом
LABEL_COL = "label"                          # Колонка с метками
OUTPUT_DIR = "./rubert-finetuned"
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
# =============================================

# 1. Загрузка данных
df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)

# Разделение на train/eval (80/20)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# 2. Токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_labels = len(set(df[LABEL_COL]))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="single_label_classification",
    use_safetensors=True,
)

# 3. Токенизация датасетов
def tokenize(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", LABEL_COL])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", LABEL_COL])

# 4. Метрики
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }

# 5. Аргументы обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    seed=42,
)


# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 7. Запуск
print("🚀 Начинаю обучение...")
trainer.train()

# Сохранение
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Модель сохранена в {OUTPUT_DIR}")