# PhishNet-BERT-v0.1-i1

**PhishNet-BERT** is a fine-tuned BERT-based model for phishing URL detection. It classifies URLs as either **legitimate** or **phishing** using transfer learning techniques on a transformer-based architecture. This model is optimized for accuracy and efficiency in identifying malicious websites.

## Features

- Fine-tuned on the **Phishing Site Classification** dataset.
- Uses **BERT (google-bert/bert-base-uncased)** as the backbone.
- Applies **transfer learning** to adapt a pre-trained transformer model for phishing detection.
- Achieves **high accuracy** and **AUC** in phishing detection.
- Designed for **real-time inference** with a simple API interface.

---

## Project Structure

```
PhishNet-BERT/
│── data/                     # Datasets (raw and processed)
│── model/                    # Trained model and checkpoints
│── scripts/                   # Training and inference scripts
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation script
│   ├── inference.py           # Script for predictions
│── app/                       # API for model deployment
│── README.md                  # Documentation
│── config.json                # Training and model configuration
```

---

## Installation

To set up the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install core dependencies manually:

```bash
pip install datasets transformers torch fastapi uvicorn
```

---

## Training the Model

To train the model from scratch or fine-tune it further, run:

```bash
python scripts/train.py
```

This will train **PhishNet-BERT** using **transfer learning** and save the best model in the `model/` directory.

---

## Evaluating the Model

To evaluate the trained model on a validation dataset, run:

```bash
python scripts/evaluate.py
```

---

## &#x20;Running Inference

Use `inference.py` to predict whether a URL is **phishing** or **legitimate**:

```bash
python scripts/inference.py --url "http://example.com/"
```

Example output (JSON format):

```json
{
    "url": "http://example.com/",
    "classification": "legitimate",
    "probabilities": {
        "phishing": 3.21,
        "legitimate": 96.79
    }
}
```

---

## API Deployment

PhishNet-BERT can be deployed as an API using **FastAPI**. To start the API:

```bash
uvicorn app.main:app --reload
```

Then, access `http://127.0.0.1:8000/docs` for an interactive Swagger UI.

---

## Model Details

- **Base Model:** google-bert/bert-base-uncased
- **Fine-Tuned Dataset:** [Phishing Site Classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)
- **Classes:** `0 = Legitimate`, `1 = Phishing`
- **Training Approach:** Transfer learning with a pre-trained transformer model
- **Training Epochs:** 10
- **Learning Rate:** 2e-4
- **Batch Size:** 8

---

## To-Do & Future Improvements

- &#x20;Initial release of PhishNet-BERT (v0.1).
- &#x20;Optimize inference speed.
- &#x20;Deploy as a **Docker container** for production use.
- &#x20;Expand dataset to include more real-world phishing cases.

---

## Author

**Thiago Cunha**\
GitHub: [thiagocborghi](https://github.com/thiagocborghi)

If you have any questions or suggestions, feel free to **open an issue** or contribute to the project!

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

