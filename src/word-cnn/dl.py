from nltk.corpus import stopwords
import string
import re
from os import listdir
from pickle import dump

# load doc into memory
def load_doc(filename):
    # open the file
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def process_docs(directory, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

def load_clean_dataset(is_train):
    neg = process_docs('data/txt_sentoken/neg', is_train)
    pos = process_docs('data/txt_sentoken/pos', is_train)
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

def save_dataset(dataset, filename):
   dump(dataset, open(filename, 'wb'))
   print('Saved: %s' % filename)

#
train_docs, ytrain = load_clean_dataset(True)
test_docs, ytest = load_clean_dataset(False)
save_dataset([train_docs, ytrain], 'data/train.pkl')
save_dataset([test_docs, ytest], 'data/test.pkl')

