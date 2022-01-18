# Tested: Question Detection
# Important function: is_question
from nltk.parse import CoreNLPParser
from nltk.tree import Tree

parser = CoreNLPParser('http://localhost:9000')

def is_question(sentence):
    if '?' in sentence: return True
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