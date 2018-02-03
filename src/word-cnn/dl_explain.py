# following this site
# https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html

# todos
# [ ] limit the number of words used
# [x] print actual label along with the predicted one
# [ ] parse e-mails using the logic in MaildirParser

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import make_pipeline
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import time, codecs
import util

class Preprocess:

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def transform(self, docs):
        if not isinstance(docs, list):
            docs = [docs][0:1]
        encoded = tokenizer.texts_to_sequences(docs)
        padded = pad_sequences(encoded, maxlen=self.maxlen, padding='post')
        return [padded, padded, padded]

    def fit(self):
        return self

class Prep:
    def transform(self, docs):
        #return [docs, docs, docs]
        return [docs, docs, docs]

    def fit(self):
        return self



startTime=time.clock()
explainer = LimeTextExplainer(class_names=['POSITIVE', 'NEGATIVE'])

dataname='trec07'
testMatrix, testLabels = util.load_dataset(file_identifier=dataname, prefix='docs')
testLines = [' '.join(x) for x in testMatrix]

[testX, testLabels] = util.load_dataset(file_identifier=dataname)
# load tokenizer
[tokenizer, length] = util.load_dataset(file_identifier=dataname, prefix='tokenizer')

print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Vocabulary size: %d ' % vocab_size)

# load the model
model = load_model( 'data/model-'+dataname+'.h5' )


# Prepare the pipeline
prepro = Preprocess(length)
pipe = make_pipeline(prepro, model)

#prep2 = Prep()
#pipe2 = make_pipeline(prep2, model)
#print(pipe2.predict(testX[0:1]))


for idx in range(10, 20):
    print("DocumentId: %d" % idx)
    res = pipe.predict(testLines[idx])
    print([round(_, 1) for _ in res[0]])
    print("Label: " + str(testLines[idx]))
    exp = explainer.explain_instance(testLines[idx], pipe.predict, labels=(0,1), num_features=6)
    for x in exp.as_list():
        print('%s: %8.4f' % (codecs.encode(x[0], 'rot_13'), x[1]))

print('execution-time (cpu): %d' % (time.clock()-startTime))