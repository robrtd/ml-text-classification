# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import email
import util
from MailParser import MailParser


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

def load_mail_from_file(filename, size=None):
    # open the file
    text = "ERROR"
    with open(filename, 'r') as file:
        try:
            #text = file.read(size)
            mail = email.message_from_file(file)
            header = MailParser.getMessageHeader(mail)
            body = MailParser.parseMessageParts(mail)
            email_str = str(header) + "\n" + str(body)

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


def load_clean_trec07(index_type='partial'):
    index_path = '/media/data/ml/trec07p/' + index_type
    docs, labels = process_trec07(index_path=index_path)
    return docs, labels


# choose a dataset
dataname = 'trec07'
docs, labels = load_clean_trec07()
print([docs[i] for i in range(min(len(docs), 10))])
util.save_dataset([docs, labels], file_identifier=dataname, prefix='docs')

trainX, tokenizer, length = util.pre_process(docs)
util.save_dataset([trainX, labels], file_identifier=dataname)
util.save_dataset([tokenizer, length], file_identifier=dataname, prefix='tokenizer')

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

