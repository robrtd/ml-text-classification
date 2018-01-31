from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import load, dump
import re
import string

# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


def save_dataset(dataset, filename):
   dump(dataset, open(filename, 'wb'))
   print('Saved: %s' % filename)

def print_dataset(lines, nr=10):
    for i in range(min(nr, len(lines))):
        print(len(lines[i]))
        print(lines[i])

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# pre-process dataset
def pre_process(docs, tokenizer=None, length=None):
    docs = [' '.join(x) for x in docs]
    print(len(docs))
    print_dataset(docs)

    if not tokenizer:
        tokenizer = create_tokenizer(docs)

    if not length:
        length = max_length(docs)
    docsX = encode_text(tokenizer, docs, length)

    return docsX, tokenizer, length




# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

pass
