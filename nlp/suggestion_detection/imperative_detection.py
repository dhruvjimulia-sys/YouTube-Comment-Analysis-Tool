# Imperative Detection: Tested
from nltk import RegexpParser
from nltk.tree import Tree
from nltk.parse import CoreNLPParser

parser = CoreNLPParser('http://localhost:9000')

# chunks the sentence into grammatical phrases based on its POS-tags
# Adverb + verb e.g. Silently switch off the AC
# Interjection + , + verb e.g. Hey, turn on the lights
# Interjection + , + verb, sing. present, non-3d e.g.
# Noun + , Compulsary + Verb
# Personal Pronoun + Verb: You,
# Noun + , Compulsary + verb, sing. present, non-3d 
# Proper Noun + , + 
# Proper Noun + , + Verb
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

# Main function to check if string contains imperatives
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

# INDEX
# CC coordinating conjunction
# CD cardinal digit
# DT determiner
# EX existential there (like: “there is” … think of it like “there exists”)
# FW foreign word
# IN preposition/subordinating conjunction
# JJ adjective ‘big’
# JJR adjective, comparative ‘bigger’
# JJS adjective, superlative ‘biggest’
# LS list marker 1)
# MD modal could, will
# NN noun, singular ‘desk’
# NNS noun plural ‘desks’
# NNP proper noun, singular ‘Harrison’
# NNPS proper noun, plural ‘Americans’
# PDT predeterminer ‘all the kids’
# POS possessive ending parent’s
# PRP personal pronoun I, he, she
# PRP$ possessive pronoun my, his, hers
# RB adverb very, silently,
# RBR adverb, comparative better
# RBS adverb, superlative 
# VBN verb, past participle taken
# VBP verb, sing. present, non-3d take
# VBZ verb, 3rd person sing. present takes
# WDT wh-determiner which
# WP wh-pronoun who, whatbest
# RP particle give up
# TO, to go ‘to’ the store.
# UH interjection, errrrrrrrm
# VB verb, base form take
# VBD verb, past tense took
# VBG verb, gerund/present participle taking
# WP$ possessive wh-pronoun whose
# WRB wh-abverb where, when
