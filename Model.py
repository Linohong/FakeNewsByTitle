import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import DataProcess.DataProcess as DP
import Args

torch.manual_seed(1)

'''
README : Model.py

    ############ class CnnSentEnc ############  
    It defines CNN Network for encoding the sentence.
    This network is based on the paper from Yoon Kim, 2014.
'''

class CnnEnc(nn.Module) :
    '''
        CNN Encoder
    '''
    def __init__(self, Vocab) :
        super(CnnEnc, self).__init__()
        # embedding layer
        self.embeddings = nn.Embedding(Args.args.vocab_size + 4, Args.args.embed_dim)
        self.feat_size = Args.args.feat_size
        self.embed_dim = Args.args.embed_dim
        self.max_sent = Args.args.max_sent
        self.hidden_size = Args.args.hidden_size

        if Args.args.predefined == True :
            wordEmbedding = DP.load_predefined_embedding(Args.args.embed_path, Vocab)
            self.embeddings.weight.data.copy_(torch.from_numpy(wordEmbedding))  # pretrained vocabulary used.

        # Kernel Operation
        self.conv1 = nn.Conv2d(1, self.feat_size, (3, self.embed_dim) )
        self.conv2 = nn.Conv2d(1, self.feat_size, (4, self.embed_dim) )
        self.conv3 = nn.Conv2d(1, self.feat_size, (5, self.embed_dim) )
        # Dropout & Linear
        self.drp = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.feat_size * 3, self.hidden_size)
        self.Tanh = nn.Tanh()

    def forward(self, x) :
        # Get embeddings first then conv.

        # if (type == 'sent') :
        x = self.embeddings(x).view(-1, 1, self.max_sent, self.embed_dim) # batch size sets to -1 for flexibility

        # Max pooling over a (2, 2) window
        maxed_conv1 = F.max_pool2d(F.relu(self.conv1(x)), (self.max_sent-2, 1) ) # second argument for the max-pooling size
        maxed_conv2 = F.max_pool2d(F.relu(self.conv2(x)), (self.max_sent-3, 1) )
        maxed_conv3 = F.max_pool2d(F.relu(self.conv3(x)), (self.max_sent-4, 1) )

        # If the size is square you can only specify a single number
        maxed_conv1 = maxed_conv1.view(-1, self.num_flat_features(maxed_conv1))
        maxed_conv2 = maxed_conv2.view(-1, self.num_flat_features(maxed_conv2))
        maxed_conv3 = maxed_conv3.view(-1, self.num_flat_features(maxed_conv3))

        x = torch.cat((maxed_conv1, maxed_conv2, maxed_conv3), 1) # becomes : batch_size x 300
        # Fully connected layer with dropout and softmax output
        x = self.drp(x)
        x = self.Tanh(self.fc(x))

        return x

    def num_flat_features(self, x) :
        size = x.size()[1:]
        num_features = 1
        for s in size :
            num_features *= s
        return num_features





class DocEnc(nn.Module) :
    '''
        Document Encoder
    '''
    def __init__(self, Vocab) :
        super(DocEnc, self).__init__()
        self.batch_size = Args.args.batch_size
        self.sent_num = Args.args.sent_num
        self.hidden_size = Args.args.hidden_size
        self.embed_dim = Args.args.embed_dim
        self.attnHidden = torch.empty(self.batch_size, self.hidden_size*2,  1) # hidden_size * 2 for bidirectional.
        nn.init.normal_(self.attnHidden) # declare a hidden vector for scoring using attention mechanism with normal distribution initialization

        self.sentEncoder = CnnEnc(Vocab)
        self.DocEncoder = nn.LSTM(input_size=self.embed_dim,
                                  hidden_size=self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True)


    def forward(self, inputs) :
        sent_hiddens = []
        # inputs = sorted(inputs, reverse=True)

        for i in range(self.sent_num) :
            sent_hiddens.append(self.sentEncoder(inputs[:][i])) # (TODO) must consider notion of mini_batch
        sent_hiddens = torch.stack(sent_hiddens).view(-1, len(sent_hiddens), self.hidden_size) # reshape the sent_hiddens as a tensor - list of tensors    batch_size * sent_num * hidden_size
        # rnn.pack_padded_sequence(sent_hiddens)


        sent_hiddens, _ = self.DocEncoder(sent_hiddens) # encoder_out : [b x sent_num x hid*2]
        doc_hidden = sent_hiddens.transpose(0,1)[-1][:] # tranpose operation necessary for accessing problem in python

        # attention mechanism
        scores = torch.bmm(sent_hiddens, self.attnHidden)
        attn = F.softmax(scores, dim=1) # probability distribution
        attn_sent_hiddens = torch.mul(attn.expand_as(sent_hiddens), sent_hiddens)

        return attn_sent_hiddens, doc_hidden # (TODO) consider if return the last or whole outputs



class ScoringNetwork(nn.Module) :
    '''
        Scoring Network 
    '''
    def __init__(self, Vocab) :
        super(ScoringNetwork, self).__init__()
        self.sent_num = Args.args.sent_num
        self.CnnEnc = CnnEnc(Vocab)
        self.DocEncoder = DocEnc(Vocab)

    def forward(self, articles, abstracts, labels) :
        attn_sent_hiddens, doc_hiddens = self.DocEncoder(articles)
        abs_hiddens = []
        for abs in abstracts :
            abs_hiddens.append(self.CnnEnc(abs))  # TODO) must consider notion of mini_batch
        abs_hiddens = torch.stack(abs_hiddens).view(-1, len(abs_hiddens), self.hidden_size)  # reshape the sent_hiddens as a tensor - list of tensors    batch_size * sent_num * hidden_size

        scores_s = torch.bmm(abs_hiddens.expand_as(attn_sent_hiddens), attn_sent_hiddens)
        scores_d = torch.bmm(abs_hiddens,  doc_hiddens)

        return scores_s + scores_d

