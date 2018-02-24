import configparser
import json
import numpy as np

import importlib
dl = importlib.import_module('dl')
util = importlib.import_module('util')

config = configparser.ConfigParser()
config.read('imap-cnn.ini')

# SWITCH between big training_run and small run with unreads mail only
# Do not generate a tokenizer for small data samples (i.e. unread mails)
is_training_run = config['DEFAULT'].getboolean('is_training_run')

# choose a dataset
dataname = 'imap-mail'
model_dataname = None

do_parse_mailbox = config['DEFAULT'].getboolean('do_parse_mailbox')

x_docs = []
y_labels = []

if not do_parse_mailbox:
    (x_docs, y_labels) = util.load_dataset(file_identifier=dataname, prefix='docs')
else:
    for mailbox in json.loads(config['DEFAULT']['imap_folder_list']):
        print("Parsing folder: %s" % mailbox)
        mailbox_dir = config['DEFAULT']['imap_folder_path'] + mailbox
        x_, y_ = dl.load_clean_dataset(is_train=is_training_run, dataset=dataname, mail_inbox=mailbox_dir, limit=config['DEFAULT'].getint('document_limit'))

        if len(x_docs) == 0:
            x_docs = x_
        else:
            x_docs = [a + b for (a, b) in zip(x_docs, x_)]
        y_labels += y_

prefix = 'docs'
if not is_training_run:
    prefix = 'unread'
    model_dataname = dataname

if do_parse_mailbox:
    util.save_dataset([x_docs, y_labels], file_identifier=dataname, prefix=prefix)

for i in range(min(len(x_docs[1]), 10)):
    print(str(y_labels[i]), str(x_docs[1][i]))


if model_dataname:
    [tokenizers, lengths] = util.load_dataset(file_identifier=model_dataname, prefix='tokenizers')
    #X_header, tokenizer, length = util.pre_process([x[1] for x in x_docs], tokenizer=tokenizer, length=length)
    #X_body, tokenizer, length = util.pre_process([x[2] for x in x_docs], tokenizer=tokenizer, length=length)
    #X = [X_id, X_header, X_body]
else:
    tokenizers = []
    lengths = []
    in_tokenizer = None
    in_length = None
x_token = [x_docs[0]]

for i in range(1, len(x_docs)):
    if model_dataname:
        in_tokenizer = tokenizers[i]
        in_length = lengths[i]
    x, tok, length = util.pre_process(x_docs[i], tokenizer=in_tokenizer, length=in_length)
    x_token.append(x)
    print(str(x[0:10]))
    tokenizers.append(tok)
    lengths.append(length)
    print('Data-Index: %d' % i)
    print(' Document count: %d' % len(x))
    print(' Max document length: %d' % length)
    vocab_size = len(tok.word_index) + 1
    print(' Tokenizer / Vocabulary size: %d / %d' % (tok.num_words, vocab_size))

util.save_dataset([tokenizers, lengths], file_identifier=dataname, prefix='tokenizers')

util.save_dataset([x_token, y_labels], file_identifier=dataname, prefix='eval')


