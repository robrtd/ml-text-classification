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
from sklearn.model_selection import train_test_split

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
    outputs = Dense(2, activation='sigmoid')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #plot_model(model, show_shapes=True, to_file='multichannel.png')

    return model

#[trainX, trainLabels] = util.load_dataset('data/trainXy.pkl')
#[tokenizer, length] = util.load_dataset('data/tokenizer.pkl')

#dataset='trec07'
dataset='imap-mail'
model_name=dataset

[X, y_labels] = util.load_dataset(file_identifier=dataset)
[tokenizer, length] = util.load_dataset(file_identifier=model_name, prefix='tokenizer')

util.print_dataset(X)

print('Max document length: %d' % length)
vocab_size = len(tokenizer.word_index) + 1
tokenizer_size = tokenizer.num_words
print('Tokenizer/Vocabulary size: %d / %d ' % (tokenizer_size, vocab_size))

trainX, testX, trainLabels, testLabels = train_test_split(X, y_labels, test_size=0.1)

if START_FROM_SCRATCH:
    model = define_model(length, vocab_size)
else:
    model = load_model('data/model-'+model_name+'.h5')


tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', verbose=1)
model.fit([trainX, trainX, trainX], trainLabels, epochs=20, batch_size=128, callbacks=[tensorboard, reduce_lr], validation_data=([testX, testX, testX], testLabels))
model.save('data/model-'+model_name+'.h5')
