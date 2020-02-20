""" Uses the *.wrd files from train.scp (mfc) to build the timit "words"
corpus, and then applied a phonemic dictionary on it to print phonemic output.
"""

        
import re
words_dict = {}
#with open("/Users/gabrielsynnaeve/postdoc/contextual_segmentation/phonology_dict/words.txt") as f:
with open("cmudict.txt") as f:
    for line in f:
        #word, _, phonemes = line.split('\t')
        word, phonemes = line.split('\t')
        words_dict[word.lower()] = list(map(lambda phn: re.sub('\d+', '', phn.lower()), phonemes.split()))
#print(words_dict)

def wrd_to_phn(w):
    w = w.lower()
    if w in words_dict:
        #return ''.join(words_dict[w])
        return ' '.join(words_dict[w])
    return ''
#print words_dict
#print(words_dict)
#map(wrd_to_phn, sentence)
with open("IEMOCAP_word_list.txt") as wrd:
    total_corpus = wrd.readlines()
#print(total_corpus)
for sentence in total_corpus:
    #print(sentence, end="")
    s = ["sil"] + list(map(wrd_to_phn, sentence)) + ["sil"]
    #print(sentence + " / " + ' '.join(s))

print(words_dict['fine'])
#print(list(map(wrd_to_phn, words_dict['fine'])))

garb = ['[LAUGHTER]', '[BREATHING]', '[GARBAGE]']
with open('IEMOCAP_phonemes.txt', 'w') as ifp:
    for i in total_corpus:
        if i.strip() in garb:
            ifp.write(i)
            continue
        tmp = i.strip()
        k = wrd_to_phn(tmp)
        print(k)
        if k == '\n':
            pass
        #ifp.write("".join(k)+"\n")