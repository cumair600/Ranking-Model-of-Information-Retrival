import pickle
import sys
import os
from nltk.stem import PorterStemmer
import re
from rank_bm25 import BM25Okapi
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import statistics
import math
import operator

def bm25_score(d, q):
    D = 3495 # total number of docs
    b = 0.75
    k1 = 1.2
    k2 = 500
    avg_doc_len = statistics.mean(doc_lens)
    doc_len = len(docs_words[d])
    K = k1 * ((1 - b) + b * (doc_len / avg_doc_len))
    
    i_score = []
    for i in q:
        _, _, df_i = term_info[i]
        tf_doc = docs_words[d].count(i)
        tf_q = q.count(i) 
        
        x = (D + 0.5) / (df_i + 0.5)
        x = math.log(x,10)
        y = (1 + k1) * (tf_doc) / (K + tf_doc)
        z = (1 + k2) * (tf_q) / (k2 + tf_q)
        ans = x * y * z
        i_score.append(ans)
        
    return sum(i_score)

def dirichlet_score(d, q):
    mu = statistics.mean(doc_lens)
    N = len(docs_words[d])

    i_score = 1
    for i in q:
        tf_doc = docs_words[d].count(i)
        f_doc = len(docs_words[d])
        prob_d = 0
        if f_doc != 0:
            prob_d = tf_doc / f_doc
        vocab = 198244
        _, tf_c, _ = term_info[i]
        prob_c = tf_c / vocab
        x = N / (N + mu) * prob_d
        y = mu / (N + mu) * prob_c
        ans = x + y
        if f_doc < 2:
            ans = 0
        i_score = i_score * ans
    return i_score

# Take argument from cmd line

argv = sys.argv
score = argv[1]
method = argv[2]

with open('Inverted-Index.pickle', 'rb') as handle:
    term_doc = pickle.load(handle)
    
with open('Term-Info.pickle', 'rb') as handle:
    term_info = pickle.load(handle)
    
with open('Docs-Words.pickle', 'rb') as handle:
    docs_words = pickle.load(handle)    

with open('Doc-Info.pickle', 'rb') as handle:
    docs = pickle.load(handle) 

with open('BM25_Scores.pickle', 'rb') as handle:
    b_topics_scores = pickle.load(handle)

with open('Dirichlet_Scores.pickle', 'rb') as handle:
    d_topics_scores = pickle.load(handle)         

inv_index = {}
for word in term_doc.keys():
    inv_index[word] = term_doc[word].keys() 

doc_ids = {}
with open("docids.txt") as f:
    for line in f:
        key, val = line.split()
        doc_ids[int(key)] = val
        
f = open("stoplist.txt", "r") 
stoplist = f.read().splitlines() #Stoplist words 

queries = {}
i = 0;
f = open("topics.xml", "r")
for line in f:
    result = re.search("<query>(.*)</query>", line)
    if result:
        words = result.group(1).split()
        queries[i] = words
        i = i + 1

ps = PorterStemmer()
for key in queries.keys():
    words = []
    for word in queries[key]:
        word = word.lower()
        if word not in stoplist:  
            word = ps.stem(word) 
            word = word.replace("'", "")   
            words.append(word)
    queries[key] = words 

# Average doc length in the corpus
doc_lens = []
for d in docs_words.keys():
    doc_lens.append(len(docs_words[d])) 


if method == "bm25":
	for q in b_topics_scores.keys():
	    rank = 0
	    for key in b_topics_scores[q]:
	        print(str(q) + "\t" + str(docs[b_topics_scores[q][rank][0]]) + "\t" + str(rank + 1) + "\t" + str(b_topics_scores[q][rank][1]) + "\t" + "run 1")
	        rank = rank + 1
	    print("\n")  

if method == "dirichlet":
	for q in d_topics_scores.keys():
	    rank = 0
	    for key in range(100):
	        print(str(q) + "\t" + str(docs[d_topics_scores[q][rank][0]]) + "\t" + str(rank + 1) + "\t" + str(d_topics_scores[q][rank][1]) + "\t" + "run 1")
	        rank = rank + 1
	    print("\n")                         