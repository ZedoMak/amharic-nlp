import urllib.request

base = "https://raw.githubusercontent.com/liyaSileshi/amharic-sentiment-analysis/main/data_preprocess/"

urllib.request.urlretrieve(base + "train.csv", "data/train.csv")
print("✓ train.csv downloaded")

urllib.request.urlretrieve(base + "test.csv", "data/test.csv")
print("✓ test.csv downloaded")