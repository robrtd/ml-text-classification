from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import util

# classify a review as negative or positive
def predict_sentiment(review, tokenizer, max_length, model):
    # clean review
    line = util.clean_doc(review)
    line = " ".join(line)
    # encode and pad review
    padded = util.encode_text(tokenizer, [line], max_length)
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), ' NEGATIVE '
    return percent_pos, ' POSITIVE '

trainLines, trainLabels = util.load_dataset( 'data/train.pkl' )
trainLines = [' '.join(x) for x in trainLines]

testLines, testLabels = util.load_dataset( 'data/test.pkl' )
testLines = [' '.join(x) for x in testLines]

# create tokenizer
tokenizer = util.create_tokenizer(trainLines)
# calculate max document length
length = util.max_length(trainLines)
print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Vocabulary size: %d ' % vocab_size)
# encode data
trainX = util.encode_text(tokenizer, trainLines, length)
testX = util.encode_text(tokenizer, testLines, length)
# load the model
model = load_model( 'data/model_v1.h5' )
# evaluate model on training dataset
_, acc = model.evaluate([trainX,trainX,trainX], trainLabels, verbose=0)
print('Test-Result vector: ' + _)
print( ' Train Accuracy: %.2f ' % (acc*100))
# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], testLabels, verbose=0)
print( ' Test Accuracy: %.2f ' % (acc*100))

text = ' Everyone will enjoy this film. I love it, recommended! '
percent, sentiment = predict_sentiment(text, tokenizer, length, model)
print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
# test negative text
text = ' This is a bad movie. Do not watch it. It sucks. '
percent, sentiment = predict_sentiment(text, tokenizer, length, model)
print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
