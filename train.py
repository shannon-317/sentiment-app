import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# データ読み込み
df = pd.read_csv("data/sentiment_data.csv")

X = df["text"]
y = df["label"]

# 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 学習
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 精度確認
pred = model.predict(X_test_vec)
print(f"精度: {accuracy_score(y_test, pred):.2f}")

# 保存
os.makedirs("model", exist_ok=True)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("モデル保存完了")