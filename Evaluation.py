import torch.optim as optim
import torch.nn as nn
import Args

def Evaluation(trainloader, SCORENET) :
    stats = {'total':0, 'got_right':0}

    for mini_batch in trainloader:
        output = SCORENET(mini_batch['article'], mini_batch['abstract'], mini_batch['label'])
        label = mini_batch['label']
        countResult(output, label, stats)

    print("total %.2f Accuracy" % (stats['got_right'] / stats['total']))
    print("%d out of %d correct" % (stats['got_right'], stats['total']))


def countResult(output, label, stats) :
    for i in range(len(output)) :
        prediction = [1, 0] if float(output[i].data[0]) >= float(output[i].data[1]) else [0, 1]
        lab = [1, 0] if float(label[i].data[0]) >= float(label[i].data[1]) else [0, 1]
        stats['total'] += 1
        if ( prediction == lab ) :  stats['got_right'] += 1





