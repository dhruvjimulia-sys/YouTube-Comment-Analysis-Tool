# This file defines the functions to determine the sentiment of a sentence

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch import softmax
import numpy as np
from time import time

nlp = None
tokenizer = None
model = None

# Defines positive and negative emoticons to quickly short circuit computation
# for sentiment
positive_emojis = ("😂", "👍", "🙌", "🤣")
negative_emojis = ("😒", "😑", "🥱", "😕")

# Loads sentiment analysis model
def load_sentiment_model():
    print("Loading Sentiment Analysis Model")
    global nlp, tokenizer, model
    nlp = pipeline("sentiment-analysis")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Returns sentiment of sentence given the sentence
# and a value 0 < neutral_threashold < 1 that decides
# how often the function returns neutral. The higher the
# value, the lower the chance of a sentence being classified
# as neutral
def sentiment(sentence, neutral_threshold):
    for positive_emoji in positive_emojis:
        if positive_emoji in sentence: return "POSITIVE"
    for negative_emoji in negative_emojis:
        if negative_emoji in sentence: return "NEGATIVE"
    
    tokenized = tokenizer.encode_plus(sentence, return_tensors="pt")
    logits = model(**tokenized)[0]
    results = softmax(logits, dim=1).tolist()[0]
    if results[1] > neutral_threshold:
        return "NEUTRAL"
    return nlp(sentence)[0]["label"]

if __name__ == "__main__":
    print("Loading model...")
    start = time()
    load_sentiment_model()
    end = time()
    print(f"Model loaded. Time taken: {end - start}")

    print("Evaluating sentiment...")
    start = time()
    print(sentiment("hilarious", 0.6))
    print(sentiment("Argentina was the first team to play in the Olympics", 0.6))
    print(sentiment("horrible", 0.6))
    end = time()
    print(f"Evaluated sentiment. Time taken: {end - start}")
