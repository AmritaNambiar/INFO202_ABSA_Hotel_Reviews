import torch
import unicodedata
import re
from transformers import BertTokenizerFast, BertForSequenceClassification

# 1. Define Aspects
aspect_cols = [
    "Location", "Room", "Cleanliness", "Service", 
    "Facilities", "Food_and_beverage", "Price", "Safety"
]

id2label = {0: "negative", 1: "neutral", 2: "positive"}

# 2. Load Model & Tokenizer (Cached for speed)
# We use a global variable or a function to load it once, not every click.
model_path = "Amrita28/my-hotel-absa-v1"  # This folder must exist in your repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    print("Loading model...")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback/Debug: Load default if custom path fails (Optional)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# 3. Cleaning Function (From your notebook)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?]){3,}", r"\1\1", text)
    return text

# 4. Inference Function
def predict_aspect_sentiments(review_text: str):
    clean = clean_text(review_text)
    results = {}

    for aspect in aspect_cols:
        aspect_name = aspect.replace("_", " & ")
        
        # Prepare input for BERT
        enc = tokenizer(
            clean,
            aspect_name,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            
            # Calculate probabilities (Confidence)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        label = id2label[pred_id]
        results[aspect_name] = (label, confidence)

    return results
