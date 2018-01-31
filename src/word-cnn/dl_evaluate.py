from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import codecs

import util


# classify a review as negative or positive
def predict_sentiment(text, tokenizer, max_length, model):
    # pre_process text
    padded, _, _ = util.pre_process([text], tokenizer, max_length)
    # predict sentiment
    yhat = model.predict([padded, padded, padded], verbose=1)
    # retrieve predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), ' NEGATIVE '
    return percent_pos, ' POSITIVE '

testLines, testLabels = util.load_dataset( 'data/test.pkl' )
#testLines = [' '.join(x) for x in testLines]

[testX, testLabels] = util.load_dataset('data/testXy.pkl')

# load tokenizer
[tokenizer, length] = util.load_dataset('data/tokenizer.pkl')

print( ' Max document length: %d ' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print( ' Vocabulary size: %d ' % vocab_size)

# load the model
model = load_model( 'data/model.h5' )

# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], testLabels, verbose=0)
print( ' Test Accuracy: %.2f ' % (acc*100))

# predict action for first entry
result = {}
for i in range(len(testX)):
    text = testLines[i]
    percent, sentiment = predict_sentiment(text, tokenizer, length, model)
    result[percent] = [sentiment, text]

for r in sorted(result.keys()):
    print(' Text: %s' % ' '.join([codecs.encode(word, 'rot_13') for word in result[r][1]]))
    print( ' Sentiment: %s (%.3f%%) ' % (result[r][0], r*100))
    print()

# test negative text
#text = ' This is a bad movie. Do not watch it. It sucks. '
#percent, sentiment = predict_sentiment(text, tokenizer, length, model)
#print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
