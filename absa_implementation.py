output_dir = "hotel_absa_model"

aspect_cols = [
    "Location",
    "Room",
    "Cleanliness",
    "Service",
    "Facilities",
    "Food_and_beverage",
    "Price",
    "Safety",
]

# Simple mapping from ID -> label
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Load once at import time
tokenizer = BertTokenizerFast.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def clean_text(text: str) -> str:
    """
    Light, BERT-friendly preprocessing:
    - Normalize unicode (NFKC)
    - Lowercase (for uncased BERT)
    - Remove control characters
    - Normalize whitespace
    - Slightly normalize repeated punctuation
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize Unicode (e.g., fancy quotes, accents) to a consistent form
    text = unicodedata.normalize("NFKC", text)

    # Lowercasing is standard for bert-base-uncased
    text = text.lower()

    # Remove control characters while keeping newlines if you want them
    text = "".join(
        ch
        for ch in text
        if ch == "\n" or unicodedata.category(ch)[0] != "C"
    )

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Reduce long runs of "!!!" or "???", but keep punctuation
    text = re.sub(r"([!?]){3,}", r"\1\1", text)

    return text

def analyze_review(review_text: str):
    """
    Run aspect-based sentiment analysis on a single review.

    Steps:
    - Apply the same preprocessing as training.
    - For each fixed aspect, feed (review, aspect) into BERT.
    - Return a dict: {aspect_name: sentiment_label}.

    This can be used in a UI to show per-aspect pros/cons
    for decision making and transparency.
    """
    clean = clean_text(review_text)
    model.eval()
    results = {}

    for aspect in aspect_cols:
        aspect_name = aspect.replace("_", " & ")

        # Encode (review, aspect) as a BERT sentence pair
        enc = tokenizer(
            clean,
            aspect_name,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Predict sentiment
        with torch.no_grad():
            logits = model(**enc).logits
        pred_id = int(torch.argmax(logits, dim=-1).cpu().item())

        results[aspect_name] = id2label[pred_id]

    return results
