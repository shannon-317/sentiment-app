import csv
import os

positives = [
    "とても面白いです",
    "簡単で分かりやすい",
    "買ってよかったです",
    "最高の商品です",
    "大満足です",
    "また買いたいです",
    "品質がいいです",
    "おすすめです",
    "使いやすいです",
    "コスパが最高です",
]
negatives = [
    "つまらないです",
    "難しくて挫折しました",
    "お金の無駄でした",
    "最悪です",
    "品質が悪いです",
    "二度と買いません",
    "期待外れでした",
    "失望しました",
    "使いにくいです",
    "返品したいです",
]

dataset = [(text, 1) for text in positives] + [(text, 0) for text in negatives]

os.makedirs("data", exist_ok=True)

with open("data/sentiment_data.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(dataset)

print(f"データ数: {len(dataset)}件")