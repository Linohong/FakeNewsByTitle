import DataProcess.DataProcess as DP
import DataProcess.DataSets as DS
import Args
import Model
import torch
import Train
import Evaluation
import etc.Utils as U
import pickle
stime = [None] # elements of list in python which is passed to a function as a parameter can be modified within the function.
referenced_id = {'trained':[], 'closed':[], 'tested':[]}

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


# Create Vocab
print("Reading Vocabulary .... ", end='')
Vocab = DP.Vocab(Args.args.voca_path, Args.args.vocab_size)
print("Done !!!")


if True :
    # Call DataSet
    print("Obtaining Dataset .... ", end='')
    U.timeCheck('s', stime)
    data_examples_gen = DP.form_examples(Args.args.data_path, Args.args.train_size) # generator of article, abstract pair
    DataManager = DS.NewsDataset(data_examples_gen, Vocab, False) # dataset defined
    # DataManager = DS.NewsDataset(data_examples_gen, Vocab)  # dataset defined
    # U.saveExamples(DataManager.formed_examples)
    # print(U.checkProportion(DataManager.formed_examples))
    del data_examples_gen
    print("Done !!!", end='     ')
    print(DataManager.portion)
    U.timeCheck('e', stime)



if False :
    # Save Test Examples
    U.timeCheck('s', stime)
    U.saveTestExamples()
    U.timeCheck('e', stime)



if False :
    # Get saved test examples
    U.timeCheck('s', stime)
    saved_examples = U.getSavedExamples('BILINEAR0016pre-3MTEST_testdata.p')
    DataManager = DS.NewsDataset(saved_examples, Vocab, True)
    U.timeCheck('e', stime)


if True :
    # Define Model
    print("Calling Model .... ", end = '')
    U.timeCheck('s', stime)
    if 'SCORE' in Args.args.model_name :
        NET = Model.ScoringNetwork(Vocab).to(torch.device(Args.args.device))
        print('ScoreNet Created')
    elif 'BILINEAR' in Args.args.model_name :
        NET = Model.BiLinearNetwork(Vocab).to(torch.device(Args.args.device))
        print('BiLinearNet Created')
    print("Done !!!", end = '    ')
    U.timeCheck('e', stime)



if True :
    # Training Model
    print("Training Model .... ")
    torch.cuda.empty_cache()  # for memory saving
    U.timeCheck('s', stime)
    print('memory allocated (dictionary): ', end=' ')
    print(torch.cuda.memory_allocated(1))
    trainloader = DataManager.get_trainloader(DataManager, 'train', Args.args.batch_size)
    referenced_id['trained'] = Train.Train(trainloader, NET)
    print("Done Training !!!", end= '    ')
    U.timeCheck('e', stime)



if False :
    # Save Model
    print("Saving Model .... ", end='')
    torch.save(NET, Args.args.model_name)
    with open(Args.args.model_name + '_trained.p', 'wb') as fp:
        pickle.dump(referenced_id['trained'], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done !!!")
    exit(0)



if False :
    # Getting the Network for Evaluation
    # SCORENET = torch.load(Args.args.model_name)
    NET = torch.load(Args.args.model_name)
    NET = NET.to(torch.device(Args.args.device))
    with open(Args.args.model_name + '_trained.p', 'rb') as fp:  # read input language
        referenced_id['trained'] = pickle.load(fp)


if False :
    # closed test Evaluation
    print("Evaluating .... ")
    U.timeCheck('s', stime)
    print("closed test")
    trainloader = DataManager.get_trainloader(DataManager, 'train', Args.args.batch_size)
    referenced_id['closed'] = Evaluation.Evaluation(trainloader, NET, referenced_id['trained'], type='closed')


if False :
    # Real test Evaluation
    print("Evaluating .... ")
    print("real test")
    trainloader = DataManager.get_trainloader(DataManager, 'test', Args.args.batch_size)
    referenced_id['tested'] = Evaluation.Evaluation(trainloader, NET, referenced_id['trained'], type='test')
    print("Done Evaluating !!!", end= '    ')
    U.timeCheck('e', stime)


'''
    Below sections are real test evluation section with 
    test examples of pre-saved test examples.
'''

if False :
    # Get saved test examples
    U.timeCheck('s', stime)
    saved_examples = U.getSavedExamples('BILINEAR0016pre-3MTEST_testdata.p')
    DataManager = DS.NewsDataset(saved_examples, Vocab, True)
    U.timeCheck('e', stime)


if False :
    # Real test Evaluation
    print("Evaluating .... ")
    print("real test")
    trainloader = DataManager.get_trainloader(DataManager, 'test', Args.args.batch_size)
    referenced_id['tested'] = Evaluation.Evaluation(trainloader, NET, referenced_id['trained'], type='test')
    print("Done Evaluating !!!", end= '    ')
    U.timeCheck('e', stime)