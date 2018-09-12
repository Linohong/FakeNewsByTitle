import torch.optim as optim
import torch.nn as nn
import Args
import etc.Utils as U
import torch
stime = [None]

def Train(trainloader, NET) :

    lr = Args.args.learning_rate
    if Args.args.optim == 'SGD' :
        optimizer = optim.SGD(NET.parameters(), lr=lr, momentum=0.9)
        print("USING SGD OPTIMIZER : ", end='')
        print(optimizer.defaults['lr'])
    elif Args.args.optim == 'Adam' :
        optimizer = optim.Adam(NET.parameters(), lr=lr)
        print("USING ADAM OPTIMIZER : ", end='')
        print(optimizer.defaults['lr'])
    elif Args.args.optim == 'RMSprop' :
        optimizer = optim.RMSprop(NET.parameters(), lr=lr)
        print("USING RMSprop OPTIMIZER : ", end='')
        print(optimizer.defaults['lr'])
    criterion = nn.BCELoss()

    ides = []

    for epoch in range(Args.args.epoch) :
        torch.cuda.empty_cache()  # for memory saving
        loss_total = 0.0
        samCnt = 0

        U.timeCheck('s', stime)
        for mini_batch in trainloader:
            optimizer.zero_grad()
            output = NET(mini_batch['article'], mini_batch['abstract'], mini_batch['label'])
            label = mini_batch['label']
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # for loss calculation
            cur_batch_size = mini_batch['article'].size()[0]
            samCnt += cur_batch_size
            loss_total += float(loss.data) * cur_batch_size

            # id add
            for ex in mini_batch['id'] :
                ides.append(int(ex))

        print('[%d] epoch with learning_rate : ' % (epoch+1), end= ' ')
        print(optimizer.defaults['lr'])
        print('loss : %.3f .... ' % (loss_total/samCnt), end='    ')
        # optimizer = optim.Adam(NET.parameters(), lr= optimizer.defaults['lr'] * 0.93)
        U.timeCheck('e', stime)

    return ides