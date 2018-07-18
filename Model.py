import torch
import torch.nn as nn
import torch.nn.functional as F
import etc.PreTrainedFile as D

torch.manual_seed(1)

'''
README : Model.py

    ############ class CnnSentEnc ############  
    It defines CNN Network for encoding the sentence.
    This network is based on the paper from Yoon Kim, 2014.
     
'''

class CnnSentEnc(nn.Module) :
    def __init__(self) :
        super(CnnSentEnc, self).__init__()
        # embedding layer
        # self.embeddings = nn.Embedding(D.vocab_size, D.EMBEDDING_DIM)
        self.embeddings.weight.data.copy_(torch.from_numpy(D.wordEmbedding)) # pretrained vocabulary used.

        # Kernel Operation
        self.conv1 = nn.Conv2d(1, D.FEATURE_SIZE, (3, D.EMBEDDING_DIM) )
        self.conv2 = nn.Conv2d(1, D.FEATURE_SIZE, (4, D.EMBEDDING_DIM) )
        self.conv3 = nn.Conv2d(1, D.FEATURE_SIZE, (5, D.EMBEDDING_DIM) )
        # Dropout & Linear
        self.drp = nn.Dropout(p=0.2)
        self.fc = F.tanh(nn.Linear(D.FEATURE_SIZE * 3, 2))

    def forward(self, x) :
        # Get embeddings first then conv.
        x = self.embeddings(x).view(-1, 1, D.max_sent_len, D.EMBEDDING_DIM) # batch size sets to -1 for flexibility

        # Max pooling over a (2, 2) window
        maxed_conv1 = F.max_pool2d(F.relu(self.conv1(x)), (D.max_sent_len-2, 1) ) # second argument for the max-pooling size
        maxed_conv2 = F.max_pool2d(F.relu(self.conv2(x)), (D.max_sent_len-3, 1) )
        maxed_conv3 = F.max_pool2d(F.relu(self.conv3(x)), (D.max_sent_len-4, 1) )

        # If the size is square you can only specify a single number
        maxed_conv1 = maxed_conv1.view(-1, self.num_flat_features(maxed_conv1))
        maxed_conv2 = maxed_conv2.view(-1, self.num_flat_features(maxed_conv2))
        maxed_conv3 = maxed_conv3.view(-1, self.num_flat_features(maxed_conv3))

        x = torch.cat((maxed_conv1, maxed_conv2, maxed_conv3), 1)
        # Fully connected layer with dropout and softmax output
        x = self.drp(x)
        x = self.fc(x)
        # prob = F.softmax(x, dim=1)

        return x

    def num_flat_features(self, x) :
        size = x.size()[1:]
        num_features = 1
        for s in size :
            num_features *= s
        return num_features

