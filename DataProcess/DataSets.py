'''
    Following codes are written so that
    train examples can be dealt most easily.

    (TODO) Redefine DataSet class to cutomize.

    Inheriting the Dataset class is explained in details at the site below
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
import Args
import torch
from random import shuffle
import DataProcess.DataProcess as DP
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class NewsDataset(Dataset):
    """CNN/Daily Mail Dataset"""

    def __init__(self, formed_examples, Vocab, reuse):
        """
        Args:
            data_path (string): Directory with all the News Data with binary form.
            formed_examples (generator) : generator of tuples with form : (article_list, abstract_list)
        """

        if reuse is False :
            self.set = 'train' # train/validation/test
            self.portion ={'train':80, 'validation':0, 'test':20} #train/validation/test
            self.startIdx = {'train':0, 'validation':0, 'test':0}
            self.device = torch.device(Args.args.device)
            self.Vocab = Vocab
            self.formed_examples = self.__getProperExamples__(formed_examples, if_shuffle=True) # list of dictionaries of article, abstract pair
            self.setIdxes()  # sets the start indexes of each sets
            self.indexed_examples = self.__index_examples__(self.formed_examples, tensor=True)

            # self.indexed_examples = self.__make_dictionary__(self.indexed_examples) # dictionary version 'train', 'validation', 'test'

        elif reuse is True :
            self.set = 'test'
            self.device = torch.device(Args.args.device)
            self.Vocab = Vocab
            self.indexed_examples = {}
            self.indexed_examples[self.set] = formed_examples # formed_examples should be indexed examples


    def __len__(self):
        return len(self.indexed_examples[self.set])

    def __getitem__(self, doc_idx):
        '''
            - Highly recommended to use this function to
            search out those of which don't meet the constraints
        '''
        example = self.indexed_examples[self.set][doc_idx]
        return example

    def __make_dictionary__(self, indexed_examples) :
        Examples = {'train':[], 'validation':[], 'test':[]}
        length = len(indexed_examples)
        for i in range(0, self.startIdx['validation']) :
            Examples['train'].append(indexed_examples[i])
            indexed_examples[i] = None
        for i in range(self.startIdx['validation'], self.startIdx['test']) :
            Examples['validation'].append(indexed_examples[i])
            indexed_examples[i] = None
        for i in range(self.startIdx['test'], length) :
            Examples['test'].append(indexed_examples[i])
            indexed_examples[i] = None
        # Examples['train'] = indexed_examples[:self.startIdx['validation']]
        # Examples['validation'] = indexed_examples[self.startIdx['validation']:self.startIdx['test']]
        # Examples['test'] = indexed_examples[self.startIdx['test']:]

        indexed_examples = None
        del indexed_examples
        return Examples

    def __getProperExamples__(self, formed_examples, if_shuffle=True) :
        Proper_Examples = []
        for example in formed_examples :
            if self.__is_proper_doc__(example, 'article') is not True :
                continue
            if self.__is_proper_doc__(example,'abstract') is not True :  # if not proper number of sentences in the article nor abstractreturn None
                continue
            if self.__is_proper_sents__(example['article']) is not True :
                continue
            if self.__is_proper_sents__(example['abstract']) is not True :  # if not proper number of words in a sentences of article nor abstract
                continue

            Proper_Examples.append(example)

        if if_shuffle is True :
            shuffle(Proper_Examples)

        return Proper_Examples

    def __is_proper_doc__(self, tensored_example, type) :
        '''
            :param tensored_example:
            :return:
        '''
        if type == 'article' :
            max = Args.args.sent_num
        elif type == 'abstract' :
            max = Args.args.abs_num

        example = tensored_example[type]
        return True if len(example) <= max else False

    def __is_proper_sents__(self, tensored_examples) :
        '''
            :param tensored_example:
            :param type:
            :return:
        '''
        max = Args.args.max_sent
        for sent in tensored_examples :
            if len(sent.split()) > max  :
                return False

        return True

    def __proper_art_sents__(self, tensored_example, max_len) :
        '''
            :param formed_example: formed pair of article and abstract
            :param max_len
            :return: Unexceeding max_len of article sentences
        '''
        proper_art_sents = []

        for sent in tensored_example['article'] :
            if len(sent) <= max_len :
                proper_art_sents.append(sent)

        return proper_art_sents

    def __proper_abs_sents__(self, tensored_example, max_len):
        '''
              :param formed_example: formed pair of article and abstract
              :param max_len
              :return: Unexceeding max_len of abstract sentences
          '''
        proper_abs_sents = []

        for sent in tensored_example['abstract']:
            if len(sent) <= max_len:
                proper_abs_sents.append(sent)

        return proper_abs_sents

    def which(self, ind) :
        '''
            :param ind: current index of examples
            :return: return the name of the data set of which the 'ind' is a part.
        '''
        if ind >= self.startIdx['train'] and ind < self.startIdx['validation'] :
            return 'train'
        elif ind >= self.startIdx['validation'] and ind < self.startIdx['test'] :
            return 'validation'
        else :
            return 'test'

    def __index_examples__(self, Examples, tensor=False) :
        indexed_examples = {'train':[], 'validatoin':[], 'test':[]}
        for i, example in enumerate(Examples):
            article_list = example['article']
            abstract_list = example['abstract']
            label = example['label']  # TODO) label must be added
            id = example['id']

            # tensor article
            indexed_article = []
            for sent in article_list:
                cur_sent = sent.split()
                result = []
                for word in cur_sent:
                    try:
                        result.append(self.Vocab._word_to_id[word])
                    except KeyError:
                        result.append(self.Vocab._word_to_id[DP.UNKNOWN_TOKEN])
                result = self.sent_zero_padding(result, Args.args.max_sent)
                indexed_article.append(result)

            indexed_article = self.doc_zero_padding(indexed_article, 'article', 'back', tensor=False)
            if tensor is True:
                # indexed_article = torch.tensor(indexed_article, requires_grad=True, device=self.device)
                indexed_article = torch.tensor(indexed_article, requires_grad=False, device=self.device)

            # tensor abstract
            indexed_abstract = []
            for sent in abstract_list:
                cur_sent = sent.split()
                result = []
                for word in cur_sent:
                    try:
                        result.append(self.Vocab._word_to_id[word])
                    except KeyError:
                        result.append(self.Vocab._word_to_id[DP.UNKNOWN_TOKEN])
                result = self.sent_zero_padding(result, Args.args.max_sent)
                indexed_abstract.append(result)

            indexed_abstract = self.doc_zero_padding(indexed_abstract, 'abstract', 'front')
            if tensor is True:
                # indexed_abstract = torch.tensor(indexed_abstract, requires_grad=False, device=self.device)
                indexed_abstract = torch.tensor(indexed_abstract, requires_grad=False, device=self.device)

            # tensor label (1/0)
            indexed_label = torch.tensor(label, requires_grad=False, device=self.device) if tensor is True else label  # requires_grad is not necessary for labels

            # put into the result var.
            indexed_examples[self.which(i)].append({'article': indexed_article, 'abstract': indexed_abstract, 'label': indexed_label, 'id': id})

        return indexed_examples


    def sent_zero_padding(self, input, max_len) :
        '''
            :param sent: subject to be padded with zeros
            :return: padded SENTENCE of input
        '''
        result = input
        pad_token = DP.PAD_TOKEN
        length = len(result)
        for _ in range(max_len - length) :
            result.append(self.Vocab._word_to_id[pad_token])

        return result

    def doc_zero_padding(self, tensored_example, type, position, tensor=False) :
        '''
            :param sent_hiddens : subject to be padded.
            :param type : article/ abstract
            :param position : position of the padding. front / back
            :param tensor : tensor or just index
            :return: padded document of sentences
        '''
        sent_num = Args.args.sent_num if type is 'article' else Args.args.abs_num
        max_sent = Args.args.max_sent
        length = len(tensored_example)

        pad_token = DP.PAD_TOKEN
        pads = [self.Vocab._word_to_id[pad_token] for _ in range(max_sent)]

        for _ in range(sent_num - length):
            if (position == 'back'):
                tensored_example.append(pads)
            elif (position == 'front'):
                tensored_example = [pads] + tensored_example


        return tensored_example

    def sortBatch(self, inputs) :
        '''
            :param input:
            :return: return descending order of batch
        '''
        return sorted(inputs, reverse=True)

    def getBatch_length(self, inputs) :
        '''
            :param inputs: descending order of batches
            :return: list of the number of sentences of each batch
        '''
        length = []
        for batch in inputs :
            length.append(len(batch))

        return length

    def setIdxes(self):
        # total = len(self.indexed_examples)
        # num_train = int(total * (self.portion['train'] / 100))
        # num_val = int(total * (self.portion['validation'] / 100))
        # num_test = total - (num_train + num_val)
        #
        # # set start indexes of train, validation, test
        # self.startIdx['validation'] = num_train
        # self.startIdx['test'] = num_train + num_val
        #
        # # set the number of each examples set ( proportion -> actual count )
        # self.portion['train'] = num_train
        # self.portion['validation'] = num_val
        # self.portion['test'] = num_test
        total = len(self.formed_examples)
        num_train = int(total * (self.portion['train'] / 100))
        num_val = int(total * (self.portion['validation'] / 100))
        num_test = total - (num_train + num_val)

        # set start indexes of train, validation, test
        self.startIdx['validation'] = num_train
        self.startIdx['test'] = num_train + num_val

        # set the number of each examples set ( proportion -> actual count )
        self.portion['train'] = num_train
        self.portion['validation'] = num_val
        self.portion['test'] = num_test

    def get_trainloader(self, dataset, set, batch_size) :
        self.set = set # train/validation/test
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        return trainloader

    def get_examples_of(self, set) :
        self.set = set

        start = self.startIdx[set]
        end = start + self.__len__(self)

        return self.formed_examples[start:end],  self.indexed_examples[start:end]
