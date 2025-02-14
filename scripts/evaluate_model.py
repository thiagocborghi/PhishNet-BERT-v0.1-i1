import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

MODEL_DIR = "model/PhishNet-BERT-v0.1-i1"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

dataset_dict = load_dataset("shawhin/phishing-site-classification")
tokenized_data = dataset_dict["validation"]

def evaluate_model(model, tokenizer, dataset):
    accuracy = evaluate.load("accuracy")
    auc_score = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        auc = auc_score.compute(prediction_scores=probabilities[:, 1].numpy(), references=labels)["roc_auc"]
        acc = accuracy.compute(predictions=predicted_classes, references=labels)['accuracy']
        
        return {"accuracy": round(acc, 3), "auc": round(auc, 3)}

    predictions = model(**tokenizer(dataset["text"], truncation=True, padding=True, return_tensors="pt"))
    logits = predictions.logits.detach().numpy()
    labels = dataset["label"]

    return compute_metrics((logits, labels))

metrics = evaluate_model(model, tokenizer, tokenized_data)
print(metrics)
