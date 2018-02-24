from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, GaussianNoise
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import importlib
util = importlib.import_module('util')

START_FROM_SCRATCH = True


def define_model(lengths, tokenizers):
    inputs = []
    embeddings = []
    noises = []
    flattens = []
    convs = []
    drops = []
    pools = []
    for (length, tokenizer) in zip(lengths, tokenizers):
        vocab_size = len(tokenizer.word_index) + 1
        inputs.append(Input(shape=(length,)))
        embeddings.append(Embedding(vocab_size, min(vocab_size, 100))(inputs[-1]))
        noises.append(GaussianNoise(stddev=0.1)(embeddings[-1]))

        if length > 50:
            convs.append(Conv1D(filters=32, kernel_size=4, activation='relu')(noises[-1]))
            drops.append(Dropout(0.5)(convs[-1]))
            convs.append(Conv1D(filters=32, kernel_size=4, activation='relu')(drops[-1]))
            drops.append(Dropout(0.5)(convs[-1]))
            pools.append(MaxPooling1D(pool_size=2)(drops[-1]))
            flattens.append(Flatten()(pools[-1]))
        else:
            flat = Flatten()(noises[-1])
            flattens.append(Dense(10, activation='relu')(flat))



    #inputs2 = Input(shape=(length,))
    #embedding1 = Embedding(vocab_size, 100)(inputs1)
    #inputs2 = Input(shape=(length,))
    #embedding2 = Embedding(vocab_size, 100)(inputs2)
    #conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    #drop2 = Dropout(0.5)(conv2)
    #pool2 = MaxPooling1D(pool_size=2)(drop2)
    #flat2 = Flatten()(pool2)

    #inputs3 = Input(shape=(length,))
    #embedding3 = Embedding(vocab_size, 100)(inputs3)
    #conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    #drop3 = Dropout(0.5)(conv3)
    #pool3 = MaxPooling1D(pool_size=2)(drop3)
    #flat3 = Flatten()(pool3)

    #merged = concatenate([flat1, flat2, flat3])
    merged = concatenate(flattens)
    dense1 = Dense(20, activation='relu')(merged)
    outputs = Dense(2, activation='sigmoid')(dense1)

    #model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='nn_multichannel.png')

    return model

#dataset='trec07'
dataset='imap-mail'
model_name = dataset

[X, y_labels] = util.load_dataset(file_identifier=dataset, prefix='eval')
[tokenizers, lengths] = util.load_dataset(file_identifier=model_name, prefix='tokenizers')

x_input = []
for i in range(0, len(tokenizers)):
    if lengths[i] > 600:
        # reduce the document length to the first N words
        old_length = lengths[i]
        lengths[i] = 600
        print("Reducing document-length from %d to %d" % (old_length, lengths[i]))
        print("Shape: ", X[i+1].shape)
        x = X[i+1][:,0:600]
#        x.resize(600)
        print("New Shape: ", x.shape)
        x_input.append(x)
    else:
        x_input.append(np.array(X[i+1]))

    print('Max document length: %d' % lengths[i])
    vocab_size = len(tokenizers[i].word_index) + 1
    tokenizer_size = tokenizers[i].num_words
    print('Tokenizer/Vocabulary size: %d / %d ' % (tokenizer_size, vocab_size))

y_labels = np.array(y_labels)
for idx in range(min(10, len(X[0]))):
    print("Doc-ID  :", X[0][idx])
    print("Body    :", X[1][idx])
    print("Header-1:", X[2][idx])


split_output = train_test_split(*x_input, y_labels, test_size=0.05)
x_train = split_output[0:2*len(x_input):2]
x_test = split_output[1:2*len(x_input)+1:2]
y_train = split_output[-2]
y_test = split_output[-1]

if START_FROM_SCRATCH:
    model = define_model(lengths, tokenizers)
else:
    model = load_model('data/model-'+model_name+'.h5')


tensorboard = TensorBoard(log_dir="logs_mail_channels/{}".format(time()), histogram_freq=1, write_graph=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', verbose=1)
#model.fit([trainX, trainX, trainX], trainLabels, epochs=20, batch_size=128, callbacks=[tensorboard, reduce_lr], validation_data=([testX, testX, testX], testLabels))
#model.fit(trainX, trainLabels, epochs=15, batch_size=64, callbacks=[tensorboard, reduce_lr], validation_data=(testX, testLabels))
#model.fit([trainH, trainBody], trainLabels, epochs=15, batch_size=64, callbacks=[tensorboard, reduce_lr], validation_data=(testX, testLabels))
model.fit(x_train, y_train, epochs=8, batch_size=64, callbacks=[tensorboard, reduce_lr], validation_data=(x_test, y_test))
model.save('data/model-'+model_name+'.h5')
