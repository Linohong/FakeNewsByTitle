import DataProcess.DataProcess as DP
import DataProcess.DataSets as DS
import Args
import Model
import torch
import Train
import Evaluation
import etc.TimeElapsed as TE
stime = [None] # elements of list in python which is passed to a function as a parameter can be modified within the function.


# Create Vocab
print("Reading Vocabulary .... ", end='')
Vocab = DP.Vocab(Args.args.voca_path, Args.args.vocab_size)
print("Done !!!")



# Call DataSet
print("Obtaining Dataset .... ", end='')
TE.timeCheck('s', stime)
data_examples_gen = DP.form_examples(Args.args.data_path, Args.args.train_size) # generator of article, abstract pair
DataManager = DS.NewsDataset(data_examples_gen, Vocab) # dataset defined
del data_examples_gen
trainloader = DataManager.get_trainloader('train', Args.args.batch_size)
print("Done !!!", end='     ')
print(DataManager.portion)
TE.timeCheck('e', stime)


# Define Model
print("Calling Model .... ", end = '')
TE.timeCheck('s', stime)
SCORENET = Model.ScoringNetwork(Vocab)
SCORENET = SCORENET.cuda() if Args.args.device == 'cuda' else SCORENET
print("Done !!!", end = '    ')
TE.timeCheck('e', stime)


# Training Model
print("Training Model .... ")
TE.timeCheck('s', stime)
trainloader = DataManager.get_trainloader('test', Args.args.batch_size)
Train.Train(trainloader, SCORENET)
print("Done Training !!!", end= '    ')
TE.timeCheck('e', stime)



# Save Model
print("Saving Model .... ", end='')
torch.save(SCORENET, './SavedModel/' + Args.args.model_name)
print("Done !!!")


# Evaluation
# SCORENET = torch.load('./SavedModel/ScoreAdam')
print("Evaluating .... ")
TE.timeCheck('s', stime)
Evaluation.Evaluation(trainloader, SCORENET)
print("Done Evaluating !!!", end= '    ')
TE.timeCheck('e', stime)

