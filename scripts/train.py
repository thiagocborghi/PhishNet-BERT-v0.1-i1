import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
)
import evaluate

MODEL_DIR = "model/PhishNet-BERT-v0.1-i1"

dataset_dict = load_dataset("shawhin/phishing-site-classification")

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

for name, param in model.base_model.named_parameters():
    param.requires_grad = False
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_data = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    auc = auc_score.compute(prediction_scores=probabilities[:, 1].numpy(), references=labels)["roc_auc"]
    acc = accuracy.compute(predictions=predicted_classes, references=labels)['accuracy']
    
    return {"Accuracy": round(acc, 3), "AUC": round(auc, 3)}


training_args = TrainingArguments(
    output_dir="model/checkpoints",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"Model saved at: {MODEL_DIR}")
