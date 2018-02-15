# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import csv
import email
import util
from MailParser import MailParser

csv.field_size_limit(500 * 1024)

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
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    #tokens = [codecs.encode(word, 'rot_13') for word in tokens]
    return tokens

def get_mailcontent_from_mail(mail, size=None):
    try:
        header = MailParser.getMinimalMessageHeader(mail)
        body = MailParser.parseMessageParts(mail)
        if size:
            body = body[:size] + (body[size:] and 'VeryLongMail')
        email_str = str(header) + "\nBODYSTART\n" + str(body)

    except UnicodeDecodeError:
        print('Unicode-Decode-Error file: %s' % filename)
        raise ValueError('Problem with Unicode-Encoding')
    except LookupError:
        print('LookupError Encoding-lookup-Error file: %s' % filename)
        raise ValueError('Lookup problem with encoding')

    return email_str

def load_mail_from_file(filename, size=None):
    # open the file
    text = "ERROR"
    with open(filename, 'r') as file:
        try:
            #text = file.read(size)
            mail = email.message_from_file(file)
            email_str = get_mailcontent_from_mail(mail, size)
        except UnicodeDecodeError: 
            print('Unicode-Decode-Error file: %s' % filename)
            raise ValueError('Problem with Unicode-Encoding')
        except LookupError:
            print('LookupError Encoding-lookup-Error file: %s' % filename)
            raise ValueError('Lookup problem with encoding')

    return email_str

def get_label(labelname):
    if labelname.lower()=='ham':
        return [1,0]
    return [0,1]

def process_trec07(index_path):
    filename=index_path+'/index'
    lines = [line.strip().split() for line in open(filename)]
    documents = []
    labels = []
    for line in lines:

        try:
            mail = load_mail_from_file(index_path + '/' + line[1], size=4096)

        except ValueError:
            continue

        tokens = clean_doc(mail)
        documents.append(tokens)

        labels.append(get_label(line[0]))
    return documents, labels

def process_kaggle(index_path):
    filename=index_path+'spam-mail.tr.label'
    lines = [line.strip().split(',') for line in open(filename)]
    documents = []
    labels = []
    for line in lines[1:]:

        try:
            mailfile = index_path + '/TR/TRAIN_' + str(line[0]) + '.eml'
            mail = load_mail_from_file(mailfile, size=4096)

        except ValueError:
            continue

        tokens = clean_doc(mail)
        documents.append(tokens)

        label = [1, 0]
        if line[1] == '0':
            # SPAM
            label = [0, 1]
        labels.append(label)
    return documents, labels


def process_enron(filename):
    documents = []
    labels = []
    with open(filename, newline='') as csvfile:
        mailfilereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # ignore header
        next(mailfilereader, None)
        i = 0
        for row in mailfilereader:
            if i > 10000:
                break
            try:
                mail = email.message_from_string(row[1])
                email_str = get_mailcontent_from_mail(mail)
            except ValueError:
                continue

            tokens = clean_doc(email_str)
            documents.append(tokens)

            label = [1, 0]
            labels.append(label)
            i += 1
    return documents, labels


def load_clean_dataset(dataset):
    if dataset == 'trec07':
        filepath = '/media/data/ml/trec07p/partial'
        docs, labels = process_trec07(index_path=filepath)
        return docs, labels

    if dataset == 'kaggle':
        index_path = 'data/kaggle/'
        docs, labels = process_kaggle(index_path=index_path)
        return docs, labels

    if dataset == 'enron':
        index_path = 'data/enron/emails.csv'
        docs, labels = process_enron(filename=index_path)
        return docs, labels

# choose a dataset
#dataname = 'trec07'
dataname = 'enron'
model_name = 'kaggle'
docs, labels = load_clean_dataset(dataset=dataname)
print([docs[i] for i in range(min(len(docs), 10))])
util.save_dataset([docs, labels], file_identifier=dataname, prefix='docs')

if model_name:
    # use existing tokenizer for model <model_name>
    (tokenizer, length) = util.load_dataset(file_identifier=model_name, prefix='tokenizer')
    trainX, tokenizer, length = util.pre_process(docs, tokenizer=tokenizer, length=length)
else:
    # generate new tokenizer
    trainX, tokenizer, length = util.pre_process(docs)
    util.save_dataset([tokenizer, length], file_identifier=dataname, prefix='tokenizer')

util.save_dataset([trainX, labels], file_identifier=dataname)

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print( ' Tokenizer / Vocabulary size: %d / %d' % (tokenizer.num_words, vocab_size))
print('Data items:' + str(len(docs)))

# TEST_1823 Michael Moore