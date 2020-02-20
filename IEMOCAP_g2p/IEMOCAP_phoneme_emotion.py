"""
emotion label 맞춰야함
1. 쓸 수 있는 (xxx 아닌 것들) 파일의 목록 만들기
2. 해당 파일의 phoneme sequence만 모으기
3. training CNN
"""

import pandas as pd
import glob
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize

trans_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/*/dialog/transcriptions/*.txt"
g_list = glob.glob(trans_path_train)

emo = pd.read_csv("./emotion_classes.csv")
a = emo[emo["EMOTION"] == 'ang']
h = emo[emo["EMOTION"] == 'hap']
s = emo[emo["EMOTION"] == 'sad']
n = emo[emo["EMOTION"] == 'neu']
tot = pd.concat([a,h,s,n], axis=0).reset_index(drop=True)

#c=0
with open("emotion_utt.txt", 'w') as emt:
    for file in g_list:
        with open(file) as ifp:
            line = ifp.readline()
            while line != "":
                file_name = line.split()[0]
                if file_name in list(tot.name):
                    emt.write(line)
                    #c += 1

                line = ifp.readline()
#print(c)

utt_path = "./emotion_utt.txt"
tokenizer = RegexpTokenizer(r'\s+', gaps=True)
punc = r"[.,?!-]"
total_corpus = []
#with open('sent.txt', 'w') as ifp:

# sentence tokenize
with open(utt_path) as sent:
    line = sent.readline()
    while line != "":
        tmp      = line.split(":")[1]
        tom      = re.sub(punc, " [SIL]", tmp)
        #sentence = re.sub("[.]", " [SIL]", tom)
        tok      = tokenizer.tokenize(tom)
        line     = sent.readline()
        total_corpus.append(tok)
        #print(tok)
                
# word_dict --> word_to_phoneme
words_dict = {}
with open("cmudict.txt") as f:
    for line in f:
        #word, _, phonemes = line.split('\t')
        word, phonemes = line.split('\t')
        words_dict[word.lower()] = list(map(lambda phn: re.sub('\d+', '', phn.lower()), phonemes.split()))

garb = ['[LAUGHTER]', '[BREATHING]', '[GARBAGE]', '[SIL]']
def wrd_to_phn(w):
    w = w.lower()
    if (w in words_dict):
        #return ''.join(words_dict[w])
        return ' '.join(words_dict[w])
    elif (w.upper() in garb):
        return "".join(w.upper())
    return ''

result = []
for sentence in total_corpus:
    #print(sentence)
    s = list(map(wrd_to_phn, sentence))
    string = ' '.join(s)
    result.append(string.split())
#print(result)


# load phn2vec model
import gensim
import gensim.models as g 
from gensim.models import Word2Vec 
from sklearn.manifold import TSNE 
import numpy as np

model_name = 'phonemes'
model = g.Doc2Vec.load(model_name)
vocab = list(model.wv.vocab)

tm = []
for i in result:
    seq = []
    for phn in i:
        seq.append(model[phn])
        tm.append(len(seq))

print(np.mean(tm), max(tm), min(tm))