from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load mô hình
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    score = torch.argmax(probs).item() + 1

    if score <= 2:
        return "😠 Câu nói tiêu cực"
    elif score == 3:
        return "😐 Câu nói trung lập"
    else:
        return "😊 Câu nói tích cực"

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["input_text"]
        result = classify_sentiment(text)
    return render_template("index.html", result=result)

# Dùng cho chạy local, không ảnh hưởng khi deploy trên Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
