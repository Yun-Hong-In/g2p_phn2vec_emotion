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

sentences = []
with open("output.txt") as ifp:
    line = ifp.readline()
    while line != "":
        phn = line.split()
        sentences.append(phn)
        line = ifp.readline()

