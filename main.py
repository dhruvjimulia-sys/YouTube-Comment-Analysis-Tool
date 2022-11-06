# This is the main file which analyzes comments of a YouTube video
# This file imports relevant functions from files in the nlp/ directory
# and uses them to solve the larger problem of processing YouTube Comments

from nlp.semantic_textual_similarity.main import predict_paraphrase, load_paraphrase_model, simple_paraphrase_check, model_paraphrase_check
from nlp.suggestion_detection.main import is_suggestion
from nlp.question_detection.main import is_question
from nlp.sentiment_analysis.main import sentiment, load_sentiment_model

from collections import deque
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
from os import getenv
import requests
import sys

# Loads YOUTUBE_API_SECRET_KEY stored in .env file
load_dotenv()
YOUTUBE_API_SECRET_KEY = getenv("YOUTUBE_API_SECRET_KEY")

# Loads the sentiment analysis model and the semantic similarity model
def load_models():
    load_paraphrase_model()
    load_sentiment_model()

# Given a list of comments, partitions the comments into categories
# such that one category contains comments that are semantically similar
# Returns the comments, divided into categories
def distribute_by_paraphrase(comments):
    categories = []
    first_comments_exception = False
    for comment in comments:
        if first_comments_exception:
            first_comments_exception = False
            continue
        if not categories:
            if len(comments) == 0:
                break
            elif len(comments) == 1:
                categories.append(deque([comment]))
            else:
                first_comments_exception = True
                curr_comment = comments[0]
                next_comment = comments[1]
                if predict_paraphrase(pd.DataFrame(data=[(curr_comment, next_comment)], columns=['first_sentence', 'second_sentence']))[0] == 1:
                    categories.append(deque([curr_comment, next_comment]))
                    continue
                else:
                    categories.append(deque([curr_comment]))
                    categories.append(deque([next_comment]))
        else:
            done_adding = False
            
            # Since the simple 
            for category in categories:
                for category_comment in category:
                    if simple_paraphrase_check(category_comment, comment):
                        category.append(comment)
                        done_adding = True
                        break
                if done_adding: break
            
            # predict_onnx with particular
            comparisons = pd.DataFrame(data=((category[0], comment) for category in categories), columns=['first_sentence', 'second_sentence'])
            comparison_results = model_paraphrase_check(comparisons)
            max_index = np.argmax(comparison_results)
            max_value = comparison_results[max_index]

            if max_value == 0:
                categories.append(deque([comment]))
            else:
                categories[max_index].append(comment)
    return categories

# Highly specialized function to convert list of deques into a list
# of lists. This function is used by process_comments just before
# returning the result so the result is JSON serializable 
def turn_deques_into_lists(final_categories_and_classes):
    for class_item in final_categories_and_classes:
        for index, category in enumerate(class_item):
            class_item[index] = list(category)
    return final_categories_and_classes

# Returns dictionary representing the processed comments, given a list of
# comments. In this function, we check for suggestions, check for questions,
# perform sentiment analysis, and categorize the comments into categories so
# that comments that are semantically similar are kept in the same category
# Finally, we return the processed comments as a dictionary
def process_comments(list_of_comments):
    print("Processing Comments")
    
    # We define 'neutral_threshold', a number greater than or equal to 0 and
    # less than or equal to 1. The lower this 'neutral_threashold', the higher
    # the probability of getting a neutral sentiment result
    neutral_threshold = 0.25
    
    suggestions = []
    questions = []
    positives = []
    negatives = []
    neutrals = []
    for comment in list_of_comments:
        if is_suggestion(comment):
            suggestions.append(comment)
            continue
        if is_question(comment):
            questions.append(comment)
            continue
        predicted_sentiment = sentiment(comment, neutral_threshold)
        if predicted_sentiment == 'POSITIVE':
            positives.append(comment)
        elif predicted_sentiment == 'NEGATIVE':
            negatives.append(comment)
        else:
            neutrals.append(comment)
    comments_by_class = (suggestions, questions, positives, negatives, neutrals)
    final_categories_and_classes = [distribute_by_paraphrase(class_comments) for class_comments in comments_by_class]
    final_categories_and_classes = turn_deques_into_lists(final_categories_and_classes)
    return {
        'suggestions': final_categories_and_classes[0],
        'questions': final_categories_and_classes[1],
        'positives': final_categories_and_classes[2],
        'negatives': final_categories_and_classes[3],
        'neutral': final_categories_and_classes[4]
    }

# Retrieves all comments from a video given a video id
# The video id is the last part of any URL to a YouTube video
def get_comments_from_video(video_id):
    response = requests.get(f"https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId={video_id}&key={YOUTUBE_API_SECRET_KEY}")
    json_data = json.loads(response.content.decode('utf-8'))
    next_page_token =  json_data['nextPageToken'] if 'nextPageToken' in json_data else ""
    all_comments_text = []
    without_replies_comment_counter = json_data['pageInfo']['totalResults']
    with_replies_comment_counter = 0

    while 'nextPageToken' in json_data:
      for item in json_data['items']:
        all_comments_text.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
        with_replies_comment_counter = with_replies_comment_counter + 1
        with_replies_comment_counter += item['snippet']['totalReplyCount']

      response = requests.get(f"https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId={video_id}&key={YOUTUBE_API_SECRET_KEY}&pageToken={next_page_token}")
      json_data = json.loads(response.content.decode('utf-8'))

      without_replies_comment_counter += json_data['pageInfo']['totalResults']
      if 'nextPageToken' in json_data: next_page_token = json_data['nextPageToken']

    for item in json_data['items']:
        all_comments_text.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
        with_replies_comment_counter = with_replies_comment_counter + 1
        with_replies_comment_counter += item['snippet']['totalReplyCount']

    return {
        "comments": all_comments_text,
        "comment_count_without_replies": without_replies_comment_counter,
        "comment_count_with_replies": with_replies_comment_counter,
    }

if __name__ == '__main__':
    load_models()
    if (len(sys.argv) == 1):
        example_comments = []
        with open("examples.txt", "r") as f:
            example_comments = map(lambda x:x.strip(), f.readlines())
        processed_comments = process_comments(example_comments)
    else:
       processed_comments = process_comments(get_comments_from_video(sys.argv[1])['comments'])
    print(json.dumps(processed_comments, indent=4))
