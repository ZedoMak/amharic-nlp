
import pandas as pd
from preprocess import clean_text

train = pd.read_csv("data/train.csv")
print("ORIGINAL SHAPE: ", train.shape)

#remove rows wit missing labels
train = train.dropna(subset=["sentiment"])

#clean tweet
train["tweet"] = train["tweet"].fillna("") 


train["clean_tweet"] = train["tweet"].apply(clean_text)

train = train[train["clean_tweet"].str.len()>0]

print("After cleaning: ", train.shape)

train.to_csv("data/preprocessed_train.csv", index=False)

print(train["clean_tweet"].head(10))

print("Saved preprocessed_train.csv")



