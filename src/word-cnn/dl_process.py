START_FROM_SCRATCH = False

#from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau

import util
from pickle import dump

def define_model(length, vocab_size):
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #plot_model(model, show_shapes=True, to_file='multichannel.png')

    return model

[trainX, trainLabels] = util.load_dataset('data/trainXy.pkl')
[tokenizer, length] = util.load_dataset('data/tokenizer.pkl')

util.print_dataset(trainX)

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)

if START_FROM_SCRATCH:
    model = define_model(length, vocab_size)
else:
    model = load_model('data/model-aws-600.h5')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
reduce_lr = ReduceLROnPlateau(monitor='loss', verbose=1)
model.fit([trainX, trainX, trainX], trainLabels, epochs=300, batch_size=16, callbacks=[tensorboard, reduce_lr])
model.save('data/model-aws-600.h5')

