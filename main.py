
from nlp.semantic_textual_similarity.paraphrase_detection_quantized_bert import predict_paraphrase, load_paraphrase_model, simple_paraphrase_check, predict_onnx
from nlp.suggestion_detection.main import is_suggestion
from nlp.question_detection import is_question
from nlp.sentiment_analysis import sentiment, load_sentiment_model

from collections import deque
import pandas as pd
import numpy as np

def load_models():
    load_paraphrase_model()
    load_sentiment_model()

def distribute_by_paraphrase(class_list):
    categories = []
    first_comments_exception = False
    for comment in class_list:
        if first_comments_exception:
            first_comments_exception = False
            continue
        if not categories:
            if len(class_list) == 0:
                break
            elif len(class_list) == 1:
                categories.append(deque([comment]))
            else:
                first_comments_exception = True
                curr_comment = class_list[0]
                next_comment = class_list[1]
                if predict_paraphrase(pd.DataFrame(data=[(curr_comment, next_comment)], columns=['first_sentence', 'second_sentence']))[0] == 1:
                    categories.append(deque([curr_comment, next_comment]))
                    continue
                else:
                    categories.append(deque([curr_comment]))
                    categories.append(deque([next_comment]))
        else:
            done_adding = False
            
            # Simple paraphrase check with all
            for category in categories:
                for category_comment in category:
                    if simple_paraphrase_check(category_comment, comment):
                        category.append(comment)
                        done_adding = True
                        break
                if done_adding: break
            
            # predict_onnx with particular
            comparisons = pd.DataFrame(data=((category[0], comment) for category in categories), columns=['first_sentence', 'second_sentence'])
            comparison_results = predict_onnx(comparisons)
            max_index = np.argmax(comparison_results)
            max_value = comparison_results[max_index]

            if max_value == 0:
                categories.append(deque([comment]))
            else:
                categories[max_index].append(comment)
    return categories

def turn_deques_into_lists(final_categories_and_classes):
    for class_item in final_categories_and_classes:
        for index, category in enumerate(class_item):
            class_item[index] = list(category)
    return final_categories_and_classes

# Comment Processing Pipeline:
# 1. Translate + Transliterate
# 2. Check for Suggestions
# 3. Check for Questions (which are not suggestions)
# 4. If prob of neutral from RoBERTa model > threshold
# 5. If not neutral, then sentiment analysis to classify positive/negative
# 6. Once in categories, find paraphrases
def process_comments(list_of_comments):
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
        predicted_sentiment = sentiment(comment, 0.6)
        if predicted_sentiment == 'POSITIVE':
            positives.append(comment)
        elif predicted_sentiment == 'NEGATIVE':
            negatives.append(comment)
        else:
            neutrals.append(comment)
    classes = (suggestions, questions, positives, negatives, neutrals)
    final_categories_and_classes = [distribute_by_paraphrase(class_list) for class_list in classes]
    return turn_deques_into_lists(final_categories_and_classes)

if __name__ == '__main__':
    load_models()
    print(process_comments(["You should make more videos on injustice", "I would suggest you to make videos about the injustices in India", "Why are you making videos like this", "You are horrible", "Honestly you make my day", "Make more vids on the Indian injustices", "You are amazing"]))