from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import util

# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb' ))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding= 'post' )
    return padded



# classify a review as negative or positive
def predict_sentiment(review, tokenizer, max_length, model):
    # clean review
    line = util.clean_doc(review)
    line = " ".join(line)
    # encode and pad review
    padded = encode_text(tokenizer, [line], max_length)
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), ' NEGATIVE '
    return percent_pos, ' POSITIVE '

trainLines, trainLabels = load_dataset( 'data/train.pkl' )
trainLines = [' '.join(x) for x in trainLines]

testLines, testLabels = load_dataset( 'data/test.pkl' )
testLines = [' '.join(x) for x in testLines]

# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Vocabulary size: %d ' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
# load the model
model = load_model( 'data/model.h5' )
# evaluate model on training dataset
_, acc = model.evaluate([trainX,trainX,trainX], trainLabels, verbose=0)
print( ' Train Accuracy: %.2f ' % (acc*100))
# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], testLabels, verbose=0)
print( ' Test Accuracy: %.2f ' % (acc*100))


text = ' Everyone will enjoy this film. I love it, recommended! '
percent, sentiment = predict_sentiment(text, tokenizer, max_length, model)
print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
# test negative text
text = ' This is a bad movie. Do not watch it. It sucks. '
percent, sentiment = predict_sentiment(text, tokenizer, max_length, model)
print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
