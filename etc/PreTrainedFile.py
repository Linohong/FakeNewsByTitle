import gensim
import numpy as np

'''
# README : varPack.py
This file is like a header file in C programming.
It defines and holds variables that are used in several different 
states of the whole process. 
'''


# VOCABULARY
vocab = []
vocab.extend(['_UNSEEN_'])
vocab.extend(['_ZEROS_'])
vocab = set(vocab)



# etc. Funtions
'''
def makeInput(sent) :
    words_in_sent = []
    words_in_sent.extend(sent.split())
    s = len(words_in_sent)
    if (s > max_sent_len):
        print("Longer sentence more than max_sent_len, so skipped")
        return -1

    input = []
    for word in words_in_sent:
        try:
            input.append(word_to_ix[word])
        except KeyError:
            print("No Such Word in dictionary, Give back unknown index")
            input.append(word_to_ix['_UNSEEN_'])
    for _ in range(max_sent_len - s):
        input.append(word_to_ix['_ZEROS_'])

    return input
'''
