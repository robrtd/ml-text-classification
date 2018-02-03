# code based on https://machinelearningmastery.com/
from nltk.corpus import stopwords
import string
import re
import codecs
import util


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

def load_doc(filename, size=None):
    # open the file
    text = "ERROR"
    with open(filename, 'r') as file:
    #with codecs.open(filename,'r',encoding='utf8') as file:
        # read first N lines
        try:
            #head = [next(file) for x in range(10)]
            text = file.read(size)
        except UnicodeDecodeError: 
            print('Unicode-Error file: %s' % filename)
            raise
    return text

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
            mail = load_doc(index_path + '/' + line[1], size=4096)
        except UnicodeDecodeError:
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
dataset = 'trec07'
docs, labels = load_clean_trec07()
print([docs[i] for i in range(min(len(docs), 10))])
util.save_dataset([docs, labels], dataset=dataset, prefix='docs')

trainX, tokenizer, length = util.pre_process(docs)
util.save_dataset([trainX, labels], dataset)
util.save_dataset([tokenizer, length], dataset, prefix='tokenizer')

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

