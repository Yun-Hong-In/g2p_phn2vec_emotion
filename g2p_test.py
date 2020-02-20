from g2p_en import G2p
from nltk.tokenize import RegexpTokenizer
import glob
from gensim.models import Word2Vec

##### TRAIN #####
trans_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/train/*/dialog/transcriptions/*.txt"
wav_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/train/*/sentences/wav/*/*.wav"
g = glob.glob(trans_path_train)
w = glob.glob(wav_path_train)

g2p = G2p()
tokenizer = RegexpTokenizer(r'\w+')

result = []
for file in g:
    with open(file, 'r') as ifp:
        line = ifp.readline()
        while line != "":
            if line[0] != "S":
                line = ifp.readline()
                continue
            sign = line.split(":")[1]
            spk = line.split(":")[0].split()[0]
            sign = tokenizer.tokenize(sign)
            sign = " ".join(sign)
            out = g2p(sign)
            for idx, val in enumerate(out):
                if val == " ":
                    out[idx] = "sil"
            out = " ".join(out)
            #print(out.split())
            result.append(out.split())
            line = ifp.readline()
for l in result[:3]:
    print(l)
    
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
"""
sentences : target
size     : dimension of embedding vector
window   : size of context window
min_count: minimum frequency of a word
workers  : number of processes for training
sg       : 0=CBOW, 1=Skip-gram
"""

model_result = model.wv.most_similar("D")
print(model_result)