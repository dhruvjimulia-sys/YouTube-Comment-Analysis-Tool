# This file defines the functions to test whether a
# sentence is a question

from nltk.parse import CoreNLPParser
from nltk.tree import Tree

# We define a parser that is connected to the Stanford CoreNLP Parser
parser = CoreNLPParser('http://localhost:9000')

# Function that takes a sentence as an argument and resturns True if and
# only if it identifies the sentence as a question 
def is_question(sentence):
    if '?' in sentence: return True

    # Examine POS tags in syntax tree to check whether sentence is framed as question,
    # even though question mark is not present
    # For more details on POS tags, go to https://gist.github.com/nlothian/9240750
    output = list(parser.raw_parse(sentence))
    tree = output[0]
    subtrees = tree.subtrees()
    labels = [subtree.label() for subtree in subtrees]
    if ('SBARQ' in labels) or ('SQ' in labels): return True
    if 'SBAR' in labels:
        sbartree = []
        for subtree in tree.subtrees():
            if subtree.label() == 'SBAR':
                sbartree = subtree
                break
        for sbarsubtree in sbartree:
            sbarlabel = sbarsubtree.label()
            if sbarlabel == 'WHADJP' or sbarlabel == 'WHADVP' or sbarlabel == 'WHNP' or sbarlabel == 'WHPP': return True
    return False

if __name__ == '__main__':
    print(is_question("Close the door"))
    print(is_question("The cat was dancing."))
    print(is_question("Will you please close the door"))