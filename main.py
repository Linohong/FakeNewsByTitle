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
DataManager = DS.NewsDataset(data_examples_gen, Vocab) # dataset defined
print("Done !!!")


# Define Model
SCORENET = Model.ScoringNetwork(Vocab)
trainloader = torch.utils.data.DataLoader(DataManager, batch_size=Args.args.batch_size, shuffle=True, num_workers=32)

for mini_batch in trainloader :
    print(mini_batch['article'])
    #print(mini_batch['abstract'][0].shape)
    #print(mini_batch['label'][0].shape)
    # output = SCORENET(batch_articles, batch_abstracts, batch_labels)



# train_data = torch.utils.data.TensorDataset(train, label)
# trainloader = torch.utils.data.DataLoader(train_data, batch_size=Args.args.batch_size, shuffle=False, num_workers=32)
#
# for batch_article, batch_abstract in trainloader :
#     # output = sentRNN(sample['article'])
#     output = sentRNN(batch_article)

# Train