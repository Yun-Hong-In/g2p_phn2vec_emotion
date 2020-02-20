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


tokenizer = RegexpTokenizer(r'\w+')
extra = re.compile(r"\[[A-Z]+\]")
g2p = G2p()

result = []
c = 0
with open("google_phonemes", 'w') as gp:
    with open('google_word_list', 'r') as ifp:
        line = ifp.readline()
        while line != "":
        #while c < 10:
            c += 1
            #mat = extra.search(line)
            #print(mat)
            # #if mat and line[0] == "S":
            # line = line.rstrip("\n")
            # #tmp = line.split(mat.group())
            # line = "".join(tmp)
            # sign = line.split(":")[1]
            # spk = line.split(":")[0].split()[0]
            # sign = tokenizer.tokenize(sign)
            # sign = " ".join(sign)
            out = g2p(line)
            # for idx, val in enumerate(out):
            #     if val == " ":
            #         out[idx] = "[SIL]"
            # out = " ".join(out)
            # out = out.rstrip("\n") +" " + mat.group() + "\n"
            
            result.append(out)
            gp.write(" ".join(out) + "\n") # save g2p result
            line = ifp.readline()
            if c % 100000 == 0:
                print("{} / {}".format(c, 20400000))

# # save g2p result
# with open("google_phonemes", 'w') as ifp:
#     for i in result:
#         ifp.write(" ".join(i)+"\n")
        
# Phn2Vec with google phonemes
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
model.init_sims(replace=True)
model_name='google_1bil_phn2vec'
model.save(model_name)

mpl.rcParams['axes.unicode_minus'] = False

model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:100,:])
df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(40,20)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.savefig("google_phn2vec.png")