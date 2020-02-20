import glob 
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize

tokenizer = RegexpTokenizer(r'\s+', gaps=True)
trans_path = "/home/gnlenfn/data/corpus/IEMOCAP/*/dialog/transcriptions/*.txt"
#trans_path = "./emotion_utt.txt"
pp = glob.glob(trans_path)
with open("raw1.txt", 'w') as ifp:
    for file in pp:
        with open(file) as wrd:
            line = wrd.readline()
            while line != "":
                sentence = line.split(":")[1]
                tok = tokenizer.tokenize(sentence)
                ifp.write('\n'.join(tok))
                line = wrd.readline()
                
                
punc = r"[.,?!-]"
brac = re.compile("\[")
with open("raw2.txt", 'w') as wrd:
    with open("raw1.txt") as ifp:
        line = ifp.readline()
        while line != "":
            if brac.search(line):
                tmp = re.sub("]", "] ", line)
                tom = re.sub("\[", " [", tmp)
                sen = re.sub(punc, " ", tom)
                wrd.write(sen)
                line = ifp.readline()
            else:
                sen = re.sub(punc, " ", line)
                wrd.write(sen)
                line = ifp.readline()


with open("IEMOCAP_word_list.txt", 'w') as ifp:
    with open("raw2.txt") as wrd:
        line = wrd.readline()
        while line != "":
            #sentence = line.split(":")[1]
            tok = tokenizer.tokenize(line)
            K = '\n'.join(tok)
            #print(K)
            ifp.write(K+"\n")
            line = wrd.readline()
            
# IEMOCAP sentence tokenize
punc = r"[.,?!-]"
total_corpus = []
with open('sent.txt', 'w') as ifp:
    for file in pp:
        with open(file) as sent:
            line = sent.readline()
            while line != "":
                tmp      = line.split(":")[1]
                tom      = re.sub(punc, " [SIL]", tmp)
                #sentence = re.sub("[.]", " [SIL]", tom)
                tok      = tokenizer.tokenize(tom)
                line     = sent.readline()
                total_corpus.append(tok)
                #print(tok)

import re
words_dict = {}
#with open("/Users/gabrielsynnaeve/postdoc/contextual_segmentation/phonology_dict/words.txt") as f:
with open("cmudict.txt") as f:
    for line in f:
        #word, _, phonemes = line.split('\t')
        word, phonemes = line.split('\t')
        words_dict[word.lower()] = list(map(lambda phn: re.sub('\d+', '', phn.lower()), phonemes.split()))
#print(words_dict)

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

import gensim
import gensim.models as g 
import logging 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import pandas as pd 
from gensim.models import Word2Vec 
from sklearn.manifold import TSNE 

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)   

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)

#save Model
model.init_sims(replace=True)
model_name = "phonemes"
model.save(model_name)

# Visulaize
mpl.rcParams['axes.unicode_minus'] = False
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:100, :])

df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
print(df.shape)
print(df)
fig = plt.figure()
fig.set_size_inches(40,20)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.savefig("result.png")