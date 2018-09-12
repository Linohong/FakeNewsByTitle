import torch.optim as optim
import torch.nn as nn
import Args

trained = []

def Evaluation(trainloader, SCORENET, referenced_id, type) :
    global trained
    trained = referenced_id
    stats = {'total':0, 'got_right':0, 'skipped':0}
    ides = []

    for mini_batch in trainloader:
        output = SCORENET(mini_batch['article'], mini_batch['abstract'], mini_batch['label'])
        label = mini_batch['label']
        id = mini_batch['id']
        countResult(output, label, stats, id, type)

        # id add
        for ex in mini_batch['id']:
            ides.append(int(ex))

    print(stats)
    print("total %.2f Accuracy" % (stats['got_right'] / stats['total']))
    print("%d out of %d correct" % (stats['got_right'], stats['total']))

    return ides


def countResult(output, label, stats, id, type) :
    for i in range(len(output)) :
        if int(id[i]) in trained and type == 'test':
            stats['skipped'] += 1
            continue
        prediction = [1, 0] if float(output[i].data[0]) >= float(output[i].data[1]) else [0, 1]
        lab = [1, 0] if float(label[i].data[0]) >= float(label[i].data[1]) else [0, 1]
        stats['total'] += 1
        if ( prediction == lab ) :  stats['got_right'] += 1





