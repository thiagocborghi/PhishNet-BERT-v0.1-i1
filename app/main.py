import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

MODEL_DIR = "model/PhishNet-BERT-v0.1-i1"

@app.on_event("startup")
def load_model():
    global model, tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

class TextInput(BaseModel):
    text: str

def extract_urls(text):
    url_pattern = re.compile(r"https?://[^\s]+|www\.[^\s]+")
    return url_pattern.findall(text)

def predict_url(url):
    inputs = tokenizer(url, truncation=True, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return {
        "url": url,
        "classification": "phishing" if predicted_class == 1 else "legitimate",
        "probabilities": {
            "phishing": round(probabilities[0][1].item() * 100, 2),
            "legitimate": round(probabilities[0][0].item() * 100, 2)
        }
    }


@app.post("/scan/phishing")
def scan_phishing(input: TextInput):
    try:
        urls = extract_urls(input.text)
        
        if not urls:
            return {"data": "No URLs found in the text."}
        
        results = [predict_url(url) for url in urls]

        return {"data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
