from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

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