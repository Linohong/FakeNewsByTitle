import torch.optim as optim
import torch.nn as nn
import Args

def Evaluation(trainloader, SCORENET) :
    samCnt = 0

    for mini_batch in trainloader:
        output = SCORENET(mini_batch['article'], mini_batch['abstract'], mini_batch['label'])
        label = mini_batch['label']
        printResult(output, label)


def printResult(output, label) :
    for i in range(len(output)) :
        print("---------------------------")
        print(output[i])
        print(label[i])
        print("---------------------------")
