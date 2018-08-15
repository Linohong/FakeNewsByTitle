'''
    Following codes are written so that 
    train examples can be dealt most easily.

    (TODO) Redefine DataSet class to cutomize.

    Inheriting the Dataset class is explained in details at the site below
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
import Args
import torch
import DataProcess.DataProcess as DP
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class NewsDataset(Dataset):
    """CNN/Daily Mail Dataset"""

    def __init__(self, formed_examples, Vocab):
        """
        Args:
            data_path (string): Directory with all the News Data with binary form.
            formed_examples (generator) : generator of tuples with form : (article_list, abstract_list)
        """
        self.device = torch.device(Args.args.device)
        self.formed_examples = self.__getProperExamples__(formed_examples) # list of dictionaries of article, abstract pair
        self.Vocab = Vocab
        self.indexed_examples = self.index_examples(self.formed_examples, tensor=True)

    def __len__(self):
        return len(self.indexed_examples)

    def __getitem__(self, doc_idx):
        ''' 
            - Highly recommended to use this function to  
            search out those of which don't meet the constraints
        '''
        # example = self.tensored_examples[doc_idx]
        example = self.indexed_examples[doc_idx]
        # example['article'] = example['article'][:2]

        return example


    def __getProperExamples__(self, formed_examples) :
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


    def index_examples(self, Examples, tensor=False) :
        indexed_examples = []
        for example in Examples :
            article_list = example['article']
            abstract_list = example['abstract']
            label = example['label'] # TODO) label must be added

            # tensor article
            indexed_article = []
            for sent in article_list :
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
                indexed_article = torch.tensor(indexed_article, requires_grad=True, device=self.device)


            # tensor abstract
            indexed_abstract = []
            for sent in abstract_list :
                cur_sent = sent.split()
                result = []
                for word in cur_sent :
                    try :
                        result.append(self.Vocab._word_to_id[word])
                    except KeyError :
                        result.append(self.Vocab._word_to_id[DP.UNKNOWN_TOKEN])
                result = self.sent_zero_padding(result, Args.args.max_sent)
                indexed_abstract.append(result)

            indexed_abstract = self.doc_zero_padding(indexed_abstract, 'abstract', 'front')
            if tensor is True :
                indexed_abstract = torch.tensor(indexed_abstract, requires_grad=True, device=self.device)

            # tensor label (1/0)
            indexed_label = torch.tensor(label, requires_grad=False, device=self.device) if tensor is True else label # requires_grad is not necessary for labels

            indexed_examples.append({'article':indexed_article, 'abstract':indexed_abstract, 'label':indexed_label})

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

    def loader(self, data) :
        batch_size = Args.args.batch_size
        return DataLoader(data, batch_size, shuffle=False, num_workers=32)