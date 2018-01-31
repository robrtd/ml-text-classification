# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import tensorflow as tf
from os import listdir
from pickle import dump
import MaildirParser
import util
import codecs

# load doc into memory
MAIL_INBOX_ = '/home/robert/Tng/mail/INBOX/'


def handle_question(text):
    # TODO: '?\\r\\n\\r\\nViele' -> 'rnrnViele'
    # Fragezeichen wird nicht erkannt, falls Space zu Beginn
    # \r \n nicht richtig behandelt

    # TODO: Mime encoding behandeln
    # Content-Type: text/plain; charset="utf-8"
    # Content-Transfer-Encoding: base64
    # MIME-Version: 1.0
    re_q = re.compile('\s([A-Za-z]+)(\?)(?:\s+|$)')
    return re_q.sub(' \\1 FRAGEZEICHEN ', text)


def clean_doc(doc):
    tokens = handle_question(doc).split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('german'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [codecs.encode(word, 'rot_13') for word in tokens]
    return tokens

def load_doc(filename):
    # open the file
    text = "Hallo Welt"
    with open(filename, 'r') as file:
    #with codecs.open(filename,'r',encoding='utf8') as file:
        # read first N lines
        try:
            #head = [next(file) for x in range(10)]
            text = file.read()
        except UnicodeDecodeError: 
            print('Skipping Unicode-Error file: %s' % filename)
    return text

def process_mails(folder, answered, is_read):
    maildir = MaildirParser.MaildirParser(folder)
    docs = maildir.getMessages(answered, is_read)
    documents = []
    for doc in docs:
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

def process_docs(directory, check_start, omit_value):
    documents = list()
    for filename in listdir(directory):
        if (check_start and filename.startswith(omit_value)) or \
                (not check_start and filename.endswith(omit_value)): 
            #filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents


def load_clean_mails(is_train):
    docs_replied   = process_mails(MAIL_INBOX_, answered=True, is_read=is_train)
    docs_unreplied = process_mails(MAIL_INBOX_, answered=False, is_read=is_train)
    docs = docs_replied + docs_unreplied
    labels = [1 for _ in range(len(docs_replied))] + [0 for _ in range(len(docs_unreplied))]
    return docs, labels

def load_clean_dbpedia(is_train):
    # Prepare training and testing data
    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia', test_with_fake_data=False)
    if is_train:
        x = dbpedia.train.data
        y = dbpedia.train.target
    else:
        x = dbpedia.test.data
        y = dbpedia.test.target
    documents = [clean_doc(doc[1]) for doc in x]
    return documents, y

def load_clean_dataset(is_train, dataset=None):
    if dataset == 'mail':
        return load_clean_mails(is_train)
    if dataset == 'dbpedia':
        return load_clean_dbpedia(is_train)
    else:
        neg = process_docs('data/txt_sentoken/neg', is_train)
        pos = process_docs('data/txt_sentoken/pos', is_train)
        docs = neg + pos
        labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels



# choose a dataset
dataset = 'mail'
train_docs, ytrain = load_clean_dataset(is_train=True, dataset=dataset)
test_docs, ytest = load_clean_dataset(is_train=False, dataset=dataset)
print([test_docs[i] for i in range(min(len(test_docs), 10))])

util.save_dataset([train_docs, ytrain], 'data/train.pkl')
util.save_dataset([test_docs, ytest], 'data/test.pkl')

trainX, tokenizer, length = util.pre_process(train_docs)
util.save_dataset([trainX, ytrain], 'data/trainXy.pkl')
util.save_dataset([tokenizer, length], 'data/tokenizer.pkl')

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)


testX, tokenizer, length = util.pre_process(test_docs, tokenizer, length)
util.save_dataset([testX, ytest], 'data/testXy.pkl')
