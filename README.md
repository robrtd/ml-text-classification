# ml-text-classification
Example of recurrent neural networks over characters: Character-level Convolutional Networks for Text Classification

(taken from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/learn)
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626

# Steps

dl.py

* load
   * array of array of words for each label
   * train and test data-sets
   * saved to file
   
dl_process.py

* load files and join words by ' '
* generate tokenizer from array of text
* max_len: maximum number of words in text
* text_to_sequence: transform words to integer indices
* pad each sequence with zeros
