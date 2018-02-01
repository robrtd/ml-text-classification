from keras.models import load_model
import codecs

import util



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
model = load_model( 'data/model-aws-300.h5' )

# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], testLabels, verbose=0)
print( ' Test Accuracy: %.2f ' % (acc*100))

# predict action for first entry
result = {}
for i in range(len(testX)):
    text = testLines[i]
    percent, sentiment = util.predict_sentiment(text, tokenizer, length, model)
    percent = max(percent)
    result[percent] = [sentiment, text]

for r in sorted(result.keys()):
    print(' Text: %s' % ' '.join([codecs.encode(word, 'rot_13') for word in result[r][1]]))
    print( ' Sentiment: %s (%.3f%%) ' % (result[r][0], r*100))
    print()

# test negative text
#text = ' This is a bad movie. Do not watch it. It sucks. '
#percent, sentiment = predict_sentiment(text, tokenizer, length, model)
#print( ' Review: [%s]\nSentiment: %s (%.3f%%) ' % (text, sentiment, percent*100))
