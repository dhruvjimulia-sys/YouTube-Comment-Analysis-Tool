# Tested: Sentiment Analysis
# Important functions: load_sentiment_model and sentiment
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch import softmax
import numpy as np
from time import time

nlp = None
tokenizer = None
model = None

classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

positive_emojis = ("😂", "❤", "😘", "💕", "👍", "🙌", "😍", "🤣", "💖")
negative_emojis = ("😒", "😑", "🥱", "😕")
joke_indicators = ("\"", ":", "nobody:", "am i a joke to you", "hold my")

# Loading takes 30 seconds
def load_sentiment_model():
    print("Loading Sentiment Analysis Model")
    global nlp, tokenizer, model
    nlp = pipeline("sentiment-analysis")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def sentiment(sentence, neutral_threshold):
    for positive_emoji in positive_emojis:
        if positive_emoji in sentence: return "POSITIVE"
    for negative_emoji in negative_emojis:
        if negative_emoji in sentence: return "NEGATIVE"
    for joke_indicator in joke_indicators:
        if joke_indicator in sentence: return "POSITIVE"
    
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
