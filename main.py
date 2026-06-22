from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# モデル読み込み
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# リクエストの型定義
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(input: TextInput):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    label = "positive" if pred == 1 else "negative"
    return {"text": input.text, "label": label}
