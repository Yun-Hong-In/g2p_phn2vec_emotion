import glob
import logging
import re

import gensim
import gensim.models as g
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from g2p_en import G2p
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)    

extra = re.compile(r"\[[A-Z]+\]")
g2p = G2p()
tokenizer = RegexpTokenizer(r'\w+')
trans_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/*/dialog/transcriptions/*impro*.txt"
g_list = glob.glob(trans_path_train)

result = []
c = 0
for file in g_list:
    print(file)
    c += 1
    with open(file, 'r') as ifp:
        line = ifp.readline()
        while line != "":
            mat = extra.search(line)
            if mat and line[0] == "S":
                line = line.rstrip("\n")
                tmp = line.split(mat.group())
                line = "".join(tmp)
                sign = line.split(":")[1]
                spk = line.split(":")[0].split()[0]
                sign = tokenizer.tokenize(sign)
                sign = " ".join(sign)
                out = g2p(sign)
                for idx, val in enumerate(out):
                    if val == " ":
                        out[idx] = "[SIL]"
                out = " ".join(out)
                out = out.rstrip("\n") +" " + mat.group() + "\n"
                
                result.append(out.split())
                line = ifp.readline()
                continue
            elif line[0] != "S":
                line = ifp.readline()
                continue
            else:
                sign = line.split(":")[1]
                spk = line.split(":")[0].split()[0]
                sign = tokenizer.tokenize(sign)
                sign = " ".join(sign)
                out = g2p(sign)
                for idx, val in enumerate(out):
                    if val == " ":
                        out[idx] = "[SIL]"
                out = " ".join(out)
                result.append(out.split())
                line = ifp.readline()
print(c)

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
"""
sentences : target
size     : dimension of embedding vector
window   : size of context window
min_count: minimum frequency of a word
workers  : number of processes for training
sg       : 0=CBOW, 1=Skip-gram
"""
# save Model
model.init_sims(replace=True)

model_name='test'
model.save(model_name)

# Visualize
mpl.rcParams['axes.unicode_minus'] = False

model_name = 'test'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:100,:])

df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
print(df.shape)

fig = plt.figure()
fig.set_size_inches(40,20)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.savefig("phoneme.png")


# EMBEDDING?
import tensorflow as tf 

phoneme = list(model.wv.vocab)

for x in phoneme:
    print(model[x])