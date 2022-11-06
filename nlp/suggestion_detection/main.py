# This file defines the functions to check whether a
# sentence is a suggestion or not. For the purposes
# of this project, we consider imperatives to be suggestions

from .imperative_detection import contains_imperative

# List of words that indicate suggestions
suggestion_words = ("you should", "please", "you ought to", "petition to", "could", "if i were you", "if i was you", "i would", "i suggest", "advise", "i recommend", "we want", "i want", "have you considered", "why don't you", "why not", "how about", "have you thought about", "recommendation")

# Given a comment, returns true if it is a suggestion
def is_suggestion(comment):
    comment_lower = comment.lower()
    for word in suggestion_words:
        if word in comment_lower: return True
    return contains_imperative(comment_lower)

if __name__ == '__main__':
    print(is_suggestion("You should make more videos about injustices in India"))
    print(is_suggestion("Make more videos about injustices in India"))
    print(is_suggestion("This video is horrible"))