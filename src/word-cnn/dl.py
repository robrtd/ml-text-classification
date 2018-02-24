# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import tensorflow as tf
from os import listdir
import codecs
import importlib
mdp = importlib.import_module('MaildirParser')
util = importlib.import_module('util')


# TODO
# [x] include a fixed-token for each mail-reference
#       to allow a measure for thread-length
# [x] include DATE in MinimalHeader
# [ ] Blog-Post schreiben
# [x] Use an even distribution of prediction-categories for mails
# [x] Ignore quoted reply text in message-body using EmailReplyParser

# load doc into memory
# MAIL_INBOX_ = '/home/robert/Tng/mail/10_PARTNER/'


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
    #tokens = [codecs.encode(word, 'rot_13') for word in tokens]
    return " ".join(tokens)


def load_doc(filename):
    # open the file
    text = "Hallo Welt"
    with open(filename, 'rb') as file:
        # with codecs.open(filename,'r',encoding='utf8') as file:
        # read first N lines
        try:
            # head = [next(file) for x in range(10)]
            text = file.read()
        except UnicodeDecodeError:
            print('Skipping Unicode-Error file: %s' % filename)
    return text


def process_mails(folder, answered, is_read, limit=None):
    maildir = mdp.MaildirParser(folder)
    docs = maildir.get_messages(answered, is_read, limit=limit)
    return docs


def process_docs(directory, check_start, omit_value):
    documents = list()
    for filename in listdir(directory):
        if (check_start and filename.startswith(omit_value)) or \
                (not check_start and filename.endswith(omit_value)):
            # filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

def load_clean_mails(is_train, mail_inbox, limit=None):
    assert (mail_inbox)

    if is_train:
        # only process answered read mails, not answered unread mails...
        docs_replied = process_mails(folder=mail_inbox, answered=True, is_read=is_train, limit=limit)
    else:
        docs_replied = mdp.MailInputList()
    docs_unreplied = process_mails(folder=mail_inbox, answered=False, is_read=is_train, limit=limit)

    if is_train:
        docs_replied.balance(docs_unreplied)

    docs, labels = docs_replied.convert_and_concat(docs_unreplied)
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


def load_clean_dataset(is_train, dataset=None, mail_inbox=None, limit=None):
    if dataset == 'imap-mail':
        return load_clean_mails(is_train, mail_inbox, limit=limit)
    if dataset == 'dbpedia':
        return load_clean_dbpedia(is_train)
    else:
        neg = process_docs('data/txt_sentoken/neg', is_train)
        pos = process_docs('data/txt_sentoken/pos', is_train)
        docs = neg + pos
        labels = [[0, 1] for _ in range(len(neg))] + [[1, 0] for _ in range(len(pos))]
    return docs, labels




#### SWITCH between big training_run and small run with unreads mail only
#### Do not generate a tokenizer for small data samples (i.e. unread mails)
###is_training_run = True
###
#### choose a dataset
###dataname = 'imap-mail'
###model_dataname = 'imap-mail'
###
###if is_training_run:
###    train_docs = []
###    ytrain = []
###
###    # hack to rerun data-generator without parsing mails again...
###    if False:
###        (train_docs, ytrain) = util.load_dataset(file_identifier=dataname, prefix='docs')
###    elif True:
###        MAIL_INBOX = '/home/robert/Tng/mail/INBOX/'
###        (train_docs, ytrain) = load_clean_mails(is_train=is_training_run, mail_inbox=MAIL_INBOX, limit=2000)
###    else:
###        for MAIL_INBOX in ['/home/robert/Tng/mail/INBOX/'\
###           , '/home/robert/Tng/mail/99_READ_10/'\
###           , '/home/robert/Tng/mail/10_PARTNER/']:
###           x_train, y_train = load_clean_dataset(is_train=is_training_run, dataset=dataname, mail_inbox=MAIL_INBOX)
###
###           train_docs += x_train
###           ytrain += y_train
###else:
###    MAIL_INBOX='/home/robert/Tng/mail/INBOX/'
###    train_docs, ytrain = load_clean_dataset(is_train=is_training_run, dataset=dataname, mail_inbox=MAIL_INBOX)
###
###prefix = 'docs'
###if not is_training_run:
###    prefix = 'unread'
###    model_dataname = dataname
###
###util.save_dataset([train_docs, ytrain], file_identifier=dataname, prefix=prefix)
###
###print(str([(ytrain[i], train_docs[i]) for i in range(min(len(train_docs), 10))]))
###
###if model_dataname:
###    (tokenizer, length) = util.load_dataset(file_identifier=model_dataname, prefix='tokenizer')
###    trainX, tokenizer, length = util.pre_process(train_docs, tokenizer=tokenizer, length=length)
###else:
###    # because of memory-limitations, use a random 30% sample of the data for tokenization only
###    #sampleX, _, ytrain, _ = train_test_split(train_docs, ytrain, test_size=0.2)
###    trainX, tokenizer, length = util.pre_process(train_docs)
###    util.save_dataset([tokenizer, length], file_identifier=dataname, prefix='tokenizer')
###
###util.save_dataset([trainX, ytrain], file_identifier=dataname)
###
###print(' Document count: %d' % len(train_docs))
###print(' Max document length: %d' % length)
###vocab_size = len(tokenizer.word_index) + 1
###print(' Tokenizer / Vocabulary size: %d / %d' % (tokenizer.num_words, vocab_size))
###