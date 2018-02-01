# following this site
# https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import make_pipeline
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import util

explainer = LimeTextExplainer(class_names=['POSITIVE', 'NEGATIVE'])

testMatrix, testLabels = util.load_dataset( 'data/test.pkl' )
testLines = [' '.join(x) for x in testMatrix]

[testX, testLabels] = util.load_dataset('data/testXy.pkl')
# load tokenizer
[tokenizer, length] = util.load_dataset('data/tokenizer.pkl')

print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Vocabulary size: %d ' % vocab_size)

# load the model
model = load_model( 'data/model.h5' )


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


# Prepare the pipeline
prepro = Preprocess(length)
pipe = make_pipeline(prepro, model)

#prep2 = Prep()
#pipe2 = make_pipeline(prep2, model)
#print(pipe2.predict(testX[0:1]))


print(pipe.predict(testLines[0]))
#percentage, class_description = util.predict_sentiment([testLines[0:1]], tokenizer, length, model)
exp = explainer.explain_instance(testLines[0], pipe.predict, labels=(0,1), num_features=6)
print("DocumentId: %d" % 0)
#print('Probability (POSITIVE): %d' % percentage[0])
print(exp.as_list())