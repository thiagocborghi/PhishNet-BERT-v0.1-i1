import json
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "model/PhishNet-BERT-v0.1-i1"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    phishing_prob = round(probabilities[0][1].item() * 100, 2)
    legit_prob = round(probabilities[0][0].item() * 100, 2)
    
    result = {
        "url": text,
        "classification": "phishing" if predicted_class == 1 else "legitimate",
        "probabilities": {
            "phishing": phishing_prob,
            "legitimate": legit_prob
        }
    }
    
    return json.dumps(result, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict if a URL is phishing or legitimate')
    parser.add_argument('--url', type=str, required=True, help='URL to analyze')
    
    args = parser.parse_args()
    result_json = predict_text(args.url, model, tokenizer)
    print(result_json)
