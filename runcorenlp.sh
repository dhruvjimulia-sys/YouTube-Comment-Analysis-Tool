cd nlp/stanford-corenlp-4.2.2

java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 \
-port 9000 -timeout 15000

cd ../..
