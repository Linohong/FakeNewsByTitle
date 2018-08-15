import torch.optim as optim
import torch.nn as nn
import Args
import etc.TimeElapsed as TE
stime = [None]

def Train(trainloader, SCORENET) :
    lr = Args.args.learning_rate
    optimizer = optim.SGD(SCORENET.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(SCORENET.parameters())
    criterion = nn.BCELoss()

    for epoch in range(Args.args.epoch) :
        loss_total = 0.0
        samCnt = 0

        TE.timeCheck('s', stime)
        for mini_batch in trainloader:
            optimizer.zero_grad()
            output = SCORENET(mini_batch['article'], mini_batch['abstract'], mini_batch['label'])
            label = mini_batch['label']
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # for loss calculation
            cur_batch_size = mini_batch['article'].size()[0]
            samCnt += cur_batch_size
            loss_total += float(loss.data) * cur_batch_size

        print('[%d] epoch - loss : %.3f .... ' % (epoch, loss_total/samCnt), end='    ')
        TE.timeCheck('e', stime)