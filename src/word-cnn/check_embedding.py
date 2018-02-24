import codecs

import util
import numpy as np
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import PCA

dataset='imap-mail'
model_name = dataset
#[X, y_labels] = util.load_dataset(file_identifier=dataset)
[tokenizer, length] = util.load_dataset(file_identifier=model_name, prefix='tokenizer')
vocab_size = len(tokenizer.word_index) + 1
tokenizer_size = tokenizer.num_words
print( ' Tokenizer / Vocabulary size: %d / %d' % (tokenizer.num_words, vocab_size))

# get all words
print("hallo")
#all_words = tokenizer.word_index.keys()
all_words = np.array([[x for x in range(v, v+400)] for v in range(0,12000,400)])
all_words = np.array([[x for x in range(v, v+400)] for v in range(1)])

# load the model
model = load_model( 'data/model-' + 'imap-mail-embedonly' + '.h5')
model.summary()

pred = model.predict(all_words)

layer_list = []
for layer in model.layers:
    layer_list.append(layer)
    print(layer.name)

# retrieve the embedding-layer

l = model.get_layer('embedding_1')
extended_input = [K.learning_phase()] + [l.input]
layer_f = K.function(extended_input, [l.output])
input_tensor = [0] + [all_words]
l_out = layer_f(input_tensor)

idx = 0

plt.title("Layer: %s" % l.name)
plt.imshow(l_out[0][0][0:400])
plt.show()

vocab_vec = l_out[0].reshape(400, 30)
pca = PCA(n_components=6)
result = pca.fit_transform(vocab_vec)
plt.scatter(result[:, 0], result[:, 1])
plt.show()

# get extreme words
wordlist = { idx: val for (idx, val) in enumerate(result[:, 0]) }

axis1 = sorted(wordlist, key=wordlist.get, reverse=True)[0:30]
axis2 = sorted(wordlist, key=wordlist.get, reverse=True)[-30:-1]
small1 = []
small2 = []
for idx in axis1 + axis2 + \
           [tokenizer.word_index[codecs.encode('myfrom', 'rot13')]] + \
           [tokenizer.word_index[codecs.encode('mysubject', 'rot13')]] + \
           [tokenizer.word_index[codecs.encode('myto', 'rot13')]]:
    w = [k for k in tokenizer.word_index if tokenizer.word_index[k] == idx][0]
    print(codecs.encode(w, 'rot_13'), wordlist[idx])
    small1.append(result[idx,0])
    small2.append(result[idx,1])
print("xxxxxxxxxxxx")
plt.scatter(small1, small2)
plt.show()


print("Top-Axis 0: ", axis0[0:10])
max0 = np.argmax(result[:, 0])
min0 = np.argmin(result[:, 0])
max1 = np.argmax(result[:, 1])
min1 = np.argmin(result[:, 1])
wmax0 = [ k for k in tokenizer.word_index if tokenizer.word_index[k] == max0 ][0]
wmin0 = [ k for k in tokenizer.word_index if tokenizer.word_index[k] == min0 ][0]
wmax1 = [ k for k in tokenizer.word_index if tokenizer.word_index[k] == max1 ][0]
wmin1 = [ k for k in tokenizer.word_index if tokenizer.word_index[k] == min1 ][0]
print("Max, min Axis-0: ", codecs.encode(wmax0, 'rot_13'), codecs.encode(wmin0, 'rot_13'))
print("Max, min Axis-1: ", codecs.encode(wmax1, 'rot_13'), codecs.encode(wmin1, 'rot_13'))



