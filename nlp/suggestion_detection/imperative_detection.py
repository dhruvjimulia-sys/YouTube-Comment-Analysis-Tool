# This file defines the functions to test whether a
# sentence is a imperative. If it is, then for the purposes of this
# project, we consider those sentences to be suggestions

from nltk import RegexpParser
from nltk.tree import Tree
from nltk.parse import CoreNLPParser

parser = CoreNLPParser('http://localhost:9000')

# Given a clause, resturns a list grammatical phrases based on its POS tags
# For more information on POS tags, check out https://gist.github.com/nlothian/9240750
def get_chunks(tagged_sent):
    chunkgram = r"""VB-Phrase: {<RB><VB>}                       
                    VB-Phrase: {<UH><,>*<VB>}                 
                    VB-Phrase: {<UH><,>*<VBP>}                
                    VB-Phrase: {<PRP><,>*<VB>}                 
                    VB-Phrase: {<NN.?><,><VB>}                 
                    VB-Phrase: {<NN.?><,><VBP>}                
                    VB-Phrase: {<NNP.?>+<,>*<VB>}               
                    VB-Phrase: {<NNP.?>+<,>*<VBP>}"""          
    chunkparser = RegexpParser(chunkgram)
    return chunkparser.parse(tagged_sent)

# Given a list of clauses, returns True if and only if
# any one of the clauses is an imperative
def is_clause_imperative(clauses):
    for clause in clauses:
        if len(clause) != 0 and len(clause[-1]) != 0 and clause[-1][0] != "?":
            if clause[0][1] == "VB" or clause[0][1] == "MD":
                return True
            else:
                chunk = get_chunks(clause)
                if type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase":
                    return True
    return False

# Given a sentence, returns true if the sentence contains an imperative
def contains_imperative(sentence_string):
    tree = list(parser.raw_parse(sentence_string.lower()))[0]
    pos_list = tree.pos()

    clause = []
    clauses = []
    for word_tuple in pos_list:
        if word_tuple[1] == "." or word_tuple[1] == "CC" or word_tuple[1] == "?" or word_tuple[1] == "!":
            clauses.append(clause)
            clause = []
        else:
            clause.append(word_tuple)
            if word_tuple == pos_list[len(pos_list) - 1]:
                clauses.append(clause)
    return is_clause_imperative(clauses)

if __name__ == '__main__':
    print(contains_imperative("I am not working."))
    print(contains_imperative("Do my homework for me."))