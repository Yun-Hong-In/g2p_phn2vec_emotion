from g2p_en import G2p
from nltk.tokenize import RegexpTokenizer
import glob

##### TRAIN #####
trans_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/train/*/dialog/transcriptions/*.txt"
wav_path_train = "/home/gnlenfn/data/corpus/IEMOCAP/train/*/sentences/wav/*/*.wav"
g = glob.glob(trans_path_train)
w = glob.glob(wav_path_train)
#print(w)

g2p = G2p()
tokenizer = RegexpTokenizer(r'\w+')
# text
with open("./data/train/text", 'w') as f:
    for file in g:
        with open(file, 'r') as ifp:
            line = ifp.readline()
            while line != "":
                if line[0] != "S":
                    #print(line)
                    line = ifp.readline()
                    continue
                sign = line.split(":")[1]
                spk = line.split(":")[0].split()[0]
                sign = tokenizer.tokenize(sign)
                sign = " ".join(sign)
                #print(sign)
                out = g2p(sign)
                #print(out)
                for idx, val in enumerate(out):
                    if val == ' ':
                        out[idx] = "sil"
                out = " ".join(out)
                #print(out)
                f.write(spk + " ")
                f.write(out + "\n")
                line = ifp.readline()
                
                
# spk2gender
with open("./data/train/spk2gender", 'w') as ifp:
    for file in g:
        with open(file, 'r') as f:
            line = f.readline()
            while line != "":
                if line[0] != "S":
                    line = f.readline()
                    continue
                spk = line.split(":")[0].split()[0]
                spk = spk.split("_")
                spk = "_".join(spk[:-1])
                ifp.write(spk + " " + spk[5] + "\n")
                line = f.readline()
                
# wav.scp
with open("./data/train/wav.scp", 'w') as ifp:
    for file in w:
        utt = file.split("/")[-1].split(".")[0]
        ifp.write(utt + " " + file + "\n")
        
# utt2spk
with open("./data/train/utt2spk", 'w') as ifp:
    for file in w:
        utt = file.split("/")[-1].split(".")[0]
        
        
##### TEST #####
trans_path_test = "/home/gnlenfn/data/corpus/IEMOCAP/test/*/dialog/transcriptions/*.txt"
wav_path_test = "/home/gnlenfn/data/corpus/IEMOCAP/test/*/sentences/wav/*/*.wav"
g = glob.glob(trans_path_test)
w = glob.glob(wav_path_test)
g2p = G2p()
tokenizer = RegexpTokenizer(r'\w+')

# text
with open("./data/test/text", 'w') as f:
    for file in g:
        with open(file, 'r') as ifp:
            line = ifp.readline()
            while line != "":
                if line[0] != "S":
                    #print(line)
                    line = ifp.readline()
                    continue
                sign = line.split(":")[1]
                spk = line.split(":")[0].split()[0]
                sign = tokenizer.tokenize(sign)
                sign = " ".join(sign)
                #print(sign)
                out = g2p(sign)
                #print(out)
                for idx, val in enumerate(out):
                    if val == ' ':
                        out[idx] = "sil"
                out = " ".join(out)
                #print(out)
                f.write(spk + " ")
                f.write(out + "\n")
                line = ifp.readline()
                
                
# spk2gender
with open("./data/test/spk2gender", 'w') as ifp:
    for file in g:
        with open(file, 'r') as f:
            line = f.readline()
            while line != "":
                if line[0] != "S":
                    line = f.readline()
                    continue
                spk = line.split(":")[0].split()[0]
                spk = spk.split("_")
                spk = "_".join(spk[:-1])
                ifp.write(spk + " " + spk[5] + "\n")
                line = f.readline()
                
# wav.scp
with open("./data/test/wav.scp", 'w') as ifp:
    for file in w:
        utt = file.split("/")[-1].split(".")[0]
        ifp.write(utt + " " + file + "\n")
        
# utt2spk
with open("./data/test/utt2spk", 'w') as ifp:
    for file in w:
        utt = file.split("/")[-1].split(".")[0]
        
##########################################################################
##########################################################################
##########################################################################

### language data ###
# lexicon.txt

# nonsilence_phones.txt
# CMU phone set

# silence_phones.txt
with open("./data/local/silence_phones.txt", 'w') as f:
    f.write("sil")
    
# optional_silence.txt
with open("./data/local/optional_silence.txt", 'w') as f:
    f.write("sil")