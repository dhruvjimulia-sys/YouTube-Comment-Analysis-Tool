# Suggestion Detection: Tested
# Important Functions: is_suggestion
from .imperative_detection import contains_imperative

suggestion_words = ("you should", "please", "you ought to", "petition to", "could", "if i were you", "if i was you", "i would", "i suggest", "advise", "i recommend", "we want", "i want", "have you considered", "why don't you", "why not", "how about", "have you thought about", "recommendation")

def is_suggestion(comment):
    comment_lower = comment.lower()
    for word in suggestion_words:
        if word in comment_lower: return True
    return contains_imperative(comment_lower)

if __name__ == '__main__':
    print(is_suggestion("You should make more videos about injustices in India"))
    print(is_suggestion("Make more videos about injustices in India"))
    print(is_suggestion("This video is horrible"))