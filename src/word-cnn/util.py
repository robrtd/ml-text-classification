from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import load, dump
import gzip


def _get_filename(file_identifier, prefix, path_prefix=None):
    if path_prefix:
        path_prefix += '/'
    else:
        path_prefix = ''
    return path_prefix + 'data/' + prefix + '_' + file_identifier + '.pkl.gz'

# load a clean dataset
def load_dataset(file_identifier, prefix='data', path_prefix=None):
    filename = _get_filename(file_identifier, prefix, path_prefix)
    print("  Loading file: %s" % filename)
    return load(gzip.open(filename, 'rb'))

def save_dataset(dataset, file_identifier, prefix='data'):
    filename = _get_filename(file_identifier, prefix)
    dump(dataset, gzip.open(filename, 'wb'))
    print('Saved: %s' % filename)

def print_dataset(lines, nr=10):
    for i in range(min(nr, len(lines))):
        print(len(lines[i]))
        print(lines[i])

# fit a tokenizer
def create_tokenizer(lines):
    num_words_limit = None
    if max([len(d) for d in lines]) > 30:
        num_words_limit = 10000
    tokenizer = Tokenizer(num_words=num_words_limit)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# pre-process dataset
def pre_process(docs, tokenizer=None, length=None):
    #doc_str_list = []
    #for doc in docs:
    #    if isinstance(doc, list):
    #        doc_str_list.append(' '.join(doc))
    #    else:
    #        doc_str_list.append(doc)
    #print("  Document length: %d" % len(doc_str_list))
    print("  Document length: %d" % len(docs))

    if not tokenizer:
        #tokenizer = create_tokenizer(doc_str_list)
        tokenizer = create_tokenizer(docs)

    if not length:
        #length = max_length(doc_str_list)
        length = max_length(docs)
        if length > 600:
            length = 600
    #docsX = encode_text(tokenizer, doc_str_list, length)
    docsX = encode_text(tokenizer, docs, length)

    return docsX, tokenizer, length


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = np.array(pad_sequences(encoded, maxlen=length, padding='post'))
    return padded


# classify a text as negative or positive
def predict_sentiment(text, tokenizer, max_length, model):
    # pre_process text
    padded, _, _ = pre_process([text], tokenizer, max_length)
    # predict sentiment
    yhat = model.predict([padded, padded, padded], verbose=1)
    # retrieve predicted percentage and label
    class_description = 'NEGATIVE'
    if yhat[0,0] >= 0.5:
        class_description = 'POSITIVE'
    return yhat, class_description

pass


