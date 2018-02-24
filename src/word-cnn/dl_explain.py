# following this site
# https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html

# todos
# [x] limit the number of words used (was already in place)
# [x] print actual label along with the predicted one
# [ ] parse e-mails using the logic in MaildirParser
# [ ] Use adversarial techniques to avoid overfitting: https://arxiv.org/pdf/1605.07725.pdf

import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine import InputLayer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import time, codecs
from random import randrange
import util


class Preprocess:

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def transform(self, docs):
        if not isinstance(docs, list):
            docs = [docs][0:1]
        encoded = tokenizer.texts_to_sequences(docs)
        padded = pad_sequences(encoded, maxlen=self.maxlen, padding='post')
#        return [padded, padded, padded]
        return padded

    def fit(self):
        return self

class Prep:
    def transform(self, docs):
        return [docs, docs, docs]

    def fit(self):
        return self


is_training_run = False

startTime=time.clock()
explainer = LimeTextExplainer(class_names=['POSITIVE', 'NEGATIVE'])

test_dataname='imap-mail'
model_name= 'imap-mail'

if is_training_run:
    prefix = 'docs'
else:
    prefix = 'unread'
testMatrix, testLabels = util.load_dataset(file_identifier=test_dataname, prefix=prefix)
testLines = [' '.join(x) for x in testMatrix]

[testX, testLabels] = util.load_dataset(file_identifier=test_dataname, prefix='eval')
# load tokenizer
[tokenizer, length] = util.load_dataset(file_identifier=model_name, prefix='tokenizer')

if length > 600:
    # reduce the document length to the first 600 words
    old_length = length
    length = 600
    print("Reducing document-length from %d to %d" % (old_length, length))
    testX = testX[:, 0:length]

print( ' Document count: %d ' % len(testMatrix))
print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Tokenizer / Vocabulary size: %d / %d' % (tokenizer.num_words, vocab_size))

# load the model
model = load_model('data/model-' + model_name + '.h5')
model.summary()


# Prepare the pipeline
prepro = Preprocess(length)
pipe = make_pipeline(prepro, model)

#prep2 = Prep()
#pipe2 = make_pipeline(prep2, model)
#print(pipe2.predict(testX[0:1]))

layer_list = []
for layer in model.layers:
    layer_list.append(layer)


idx = 0
l_in = {}
for l in layer_list:
    if isinstance(l, InputLayer):
        l_in[l.input] = testX[idx].reshape(1, testX.shape[1])

    print("Layer-Name: %s" % l.name)
    if isinstance(l.input, list):
        # see https://github.com/keras-team/keras/issues/2876
        extended_input = [K.learning_phase()] + [*l.input]
        layer_f = K.function(extended_input, [l.output])
        input_tensor = [0] + [l_in[li] for li in l.input]
    else:
        extended_input = [K.learning_phase()] + [l.input]
        layer_f = K.function(extended_input, [l.output])
        input_tensor = [0] + [l_in[l.input]]
    l_out = layer_f(input_tensor)
    l_in[l.output] = l_out[0]

    if not isinstance(l, InputLayer):
        plt.title("Layer: %s" % l.name)
        #plt.imshow(l_out[0][0][0:300])
        #plt.show()

prob = {}
max_doc_explain = 30
max_doc_explain = len(testLines)
random_indices = [randrange(0, len(testLines)) for _ in range(max_doc_explain)]
for idx in random_indices:
    res = pipe.predict(testLines[idx])
    prob[idx] = res[0][0]


for idx in sorted(prob, key=prob.get, reverse=True):
    #if testLabels[idx][1] > 0.5:
    #    continue
    classification = '** DO-REPLY **'
    if testLabels[idx][1] > 0.5:
        classification = 'INFO'
    res = pipe.predict(testLines[idx])
    prediction = 'OK'
    probability = res[0][0]
    if (res[0][1] > 0.5 and testLabels[idx][1] <= 0.5) or (res[0][1] <= 0.5 and testLabels[idx][1] > 0.5):
        prediction = '*** WRONG ***'
    print("DocumentId: %d (len: %d) Classification: %s Prediction: %s" % (idx, len(testLines[idx]), classification, prediction))
    print("  Probability: %4.2f" % probability)
    print("  [%s]" % ', '.join([str(codecs.encode(_, 'rot_13')) for _ in testMatrix[idx][0:29]]))
    exp = explainer.explain_instance(testLines[idx], pipe.predict, labels=(0, 1), num_features=6)
    for x in exp.as_list():
        print('  %s: %8.6f' % (codecs.encode(x[0], 'rot_13'), x[1]))
    print()

print('execution-time (cpu): %d' % (time.clock()-startTime))
