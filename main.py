import DataProcess.DataProcess as DP
import DataProcess.DataSets as DS
import Args
import Model
import torch


# Create Vocab
Vocab = DP.Vocab(Args.args.voca_path, Args.args.vocab_size)



# Call DataSet
print("Obtaining Dataset .... ", end='')
data_examples_gen = DP.form_examples(Args.args.data_path, Args.args.train_size) # generator of article, abstract pair
dataset = DS.NewsDataset(data_examples_gen, Vocab) # dataset defined
print("Done !!!")


# Define Model
# sentCNN = Model.CnnEnc(Vocab)
sentRNN = Model.DocEnc(Vocab)
# train = []
# label = []
for i in range(5) :
    sample = dataset.__getitem__(i) # (article, abstract)
    if sample == None : continue
    # train.append(sample['article'])
    # label.append(sample['abstract'])
    output = sentRNN(sample['article'])


# train_data = torch.utils.data.TensorDataset(train, label)
# trainloader = torch.utils.data.DataLoader(train_data, batch_size=Args.args.batch_size, shuffle=False, num_workers=32)
#
# for batch_article, batch_abstract in trainloader :
#     # output = sentRNN(sample['article'])
#     output = sentRNN(batch_article)

# Train