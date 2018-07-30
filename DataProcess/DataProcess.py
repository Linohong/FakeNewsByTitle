'''

   (Lino) Following code has a copyright, and I obtained the permission to use
   with Apache license as follows.

Copyright 2017 Abigail See

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''

import Args
import gensim
import tensorflow as tf
import glob
import struct
import numpy as np
from tensorflow.core.example import example_pb2
from nltk.tokenize import sent_tokenize

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.



def example_generator(data_path, max):
    """
      Generates tf.Examples from data files.
  
      Binary data format: <length><blob>. <length> represents the byte size
      of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
      the tokenized article text and summary.
  
    Args:
      data_path:
        Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
      max:
        maximum number of data
  
    Yields:
      Deserialized tf.Example.
    """
    count = 0

    while True:
        if ( count == max ) :
            break
        filelist = glob.glob(data_path)
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty


        for f in filelist:
            if ( count == max ) :
                break
            else :
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]

                    count+=1
                    if (count == max) :
                        break

                    yield example_pb2.Example.FromString(example_str)


def text_generator(example_generator):
    """
        Generates article and abstract text from tf.Example.

        Args:
        example_generator: a generator of tf.Examples from file. See data.example_generator
    """

    for e in example_generator:  # e is a tf.Example
        try:
            article_text = e.features.feature['article'].bytes_list.value[
                0]  # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[
                0]  # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            tf.logging.error('Failed to get article or abstract from example')
            continue
        if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
            tf.logging.warning('Found an example with empty article text. Skipping it.')
        else:
            yield (article_text, abstract_text)


def form_examples(data_path, max=None):
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = text_generator(example_generator(data_path, max))

    for (article, abstract) in input_gen:  # read the next example from file. article and abstract are both strings.
        abstract_list = [sent.strip() for sent in abstract2sents(
            abstract.decode("utf-8"))]  # abstract is byte type in python3, so convert it to a string type
                                        # Use the <s> and </s> tags in abstract to get a list of sentences.
        article_list = sent_tokenize(article.decode("utf-8"))

        yield (article_list, abstract_list)


        # abstract_sentences = [sent.strip() for sent in abstract2sents(abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
        # example = Example(article, abstract_sentences, self._vocab, self._hps)  # Process into an Example.
        # self._example_queue.put(example)  # place the Example in the example queue.


def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.
  
    Args:
      article_words: list of words (strings)
      vocab: Vocabulary object
  
    Returns:
      ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
      oovs:
        A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
  
    Args:
      abstract_words: list of words (strings)
      vocab: Vocabulary object
      article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers
  
    Returns:
      ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).
  
    Args:
      id_list: list of ids (integers)
      vocab: Vocabulary object
      article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)
  
    Returns:
      words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    """
      Splits abstract text from datafile into list of sentences.
  
      Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences
  
      Returns:
          sents: List of sentence strings (no tags)
    """

    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents



class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)"""

  def __init__(self, vocab_file, max_size):
    """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
      max_size: integer. The maximum size of the resulting Vocabulary."""
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break

    print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

  def word2id(self, word):
    """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer)."""
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    """Returns the total size of the vocabulary"""
    return self._count


def load_predefined_embedding(emb_path, Vocab) :
    # Read WordEmbedding from file
    EMBEDDING_DIM = Args.args.embed_dim
    zeros = [0] * EMBEDDING_DIM
    print('Started loading word2vec ...', end='')
    readEmbedding = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=True)
    wordEmbedding = []

    for key in Vocab._word_to_id :
        try :
            wordEmbedding.append(readEmbedding[key])
        except KeyError :
            if key in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING] :
                wordEmbedding.append(zeros)
            wordEmbedding.append(np.random.rand(EMBEDDING_DIM))
    wordEmbedding = np.array(wordEmbedding)
    del readEmbedding
    print('Done !!!')

    return wordEmbedding