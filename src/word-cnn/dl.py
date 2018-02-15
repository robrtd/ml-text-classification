# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import tensorflow as tf
from os import listdir
import codecs
from sklearn.model_selection import train_test_split
import MaildirParser
import util


# TODO
# [x] include a fixed-token for each mail-reference
#       to allow a measure for thread-length
# [x] include DATE in MinimalHeader
# [ ] Blog-Post schreiben
# [x] Use an even distribution of prediction-categories for mails

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
    tokens = [codecs.encode(word, 'rot_13') for word in tokens]
    return tokens


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
            # filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents


def load_clean_mails(is_train, mail_inbox):
    assert (mail_inbox)
    docs_replied = process_mails(mail_inbox, answered=True, is_read=is_train)
    docs_unreplied = process_mails(mail_inbox, answered=False, is_read=is_train)
    docs = docs_replied + docs_unreplied

    if is_train:
        # ensure a balanced distribution of read/unread mails
        num_replied = len(docs_replied)
        num_unreplied = len(docs_unreplied)
        num_used = min(num_replied, num_unreplied)*1.0
        docs_replied, _, labels_replied, _ = train_test_split(docs_replied, [[1, 0] for x in docs_replied], test_size=1-(num_used/num_replied))
        docs_unreplied, _, labels_unreplied, _ = train_test_split(docs_unreplied, [[0, 1] for x in docs_unreplied], test_size=1-(num_used/num_unreplied))
        print("Using %d of %d replies" % (len(docs_replied), num_replied))
        print("Using %d of %d non-replies" % (len(docs_unreplied), num_unreplied))
        docs = docs_replied + docs_unreplied
        labels = labels_replied + labels_unreplied
    else:
        labels = [[1, 0] for _ in range(len(docs_replied))] + [[0, 1] for _ in range(len(docs_unreplied))]
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


def load_clean_dataset(is_train, dataset=None, mail_inbox=None):
    if dataset == 'imap-mail':
        return load_clean_mails(is_train, mail_inbox)
    if dataset == 'dbpedia':
        return load_clean_dbpedia(is_train)
    else:
        neg = process_docs('data/txt_sentoken/neg', is_train)
        pos = process_docs('data/txt_sentoken/pos', is_train)
        docs = neg + pos
        labels = [[0, 1] for _ in range(len(neg))] + [[1, 0] for _ in range(len(pos))]
    return docs, labels


# SWITCH between big training_run and small run with unreads mail only
# Do not generate a tokenizer for small data samples (i.e. unread mails)
is_training_run = True

# choose a dataset
dataname = 'imap-mail'
model_dataname = None

if is_training_run:
   train_docs = []
   ytrain = []
   for MAIL_INBOX in ['/home/robert/Tng/mail/INBOX/'\
       , '/home/robert/Tng/mail/99_READ_10/'\
       , '/home/robert/Tng/mail/10_PARTNER/']:
       x_train, y_train = load_clean_dataset(is_train=is_training_run, dataset=dataname, mail_inbox=MAIL_INBOX)

       train_docs += x_train
       ytrain += y_train
else:
    MAIL_INBOX='/home/robert/Tng/mail/INBOX/'
    train_docs, ytrain = load_clean_dataset(is_train=is_training_run, dataset=dataname, mail_inbox=MAIL_INBOX)

prefix = 'docs'
if not is_training_run:
    prefix = 'unread'
    model_dataname = dataname

util.save_dataset([train_docs, ytrain], file_identifier=dataname, prefix=prefix)

print(str([(ytrain[i], train_docs[i]) for i in range(min(len(train_docs), 10))]))

if model_dataname:
    (tokenizer, length) = util.load_dataset(file_identifier=model_dataname, prefix='tokenizer')
    trainX, tokenizer, length = util.pre_process(train_docs, tokenizer=tokenizer, length=length)
else:
    # because of memory-limitations, use a random 30% sample of the data for tokenization only
    sampleX, _, _, _ = train_test_split(train_docs, ytrain, test_size=0.7) 
    trainX, tokenizer, length = util.pre_process(sampleX)
    util.save_dataset([tokenizer, length], file_identifier=dataname, prefix='tokenizer')

util.save_dataset([trainX, ytrain], file_identifier=dataname)

print(' Document count: %d' % len(train_docs))
print(' Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print(' Tokenizer / Vocabulary size: %d / %d' % (tokenizer.num_words, vocab_size))
