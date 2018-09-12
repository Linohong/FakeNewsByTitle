import time
import Args as Args
import pickle

def timeCheck(type, stime) :
    if type is 's' : # start type
        stime[0] = time.time()

    elif type is 'e' : # end type
        elapsed = time.time() - stime[0]
        print("%.2f seconds elapsed" % (elapsed))


def saveExamples(formed_examples) :
    files = 10
    for i, example in enumerate(formed_examples) :
        if i == files :
            break
        else :
            fw = open('file%d.txt' % (i), 'w')
            fw.write(str(example['label'][0]) + '\n')
            fw.write('\n\n[Article]\n')
            for sent in example['article'] :
                fw.write(sent + '\n')


            fw.write('\n\n\n\n[Abstract]\n')
            for sent in example['abstract'] :
                fw.write(sent + '\n')
            fw.close()

    return

def saveTestExamples(DataManager) :
    print("Saving TEST DATA ... ", end='')
    DataManager.set = 'test'
    with open(Args.args.model_name + '_testdata.p', 'wb') as fp:
        pickle.dump(DataManager.indexed_examples['test'], fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(Args.args.model_name + '_testnum.p', 'wb') as fp :
        pickle.dump(len(DataManager), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done')

    return

def getSavedExamples(filename) :
    print("Obtaining saved test Dataset .... ", end='')
    with open(filename, 'rb') as fp:  # read input language
        saved_examples = pickle.load(fp)

    return saved_examples
    print('Done !!!')


def checkProportion(formed_examples) :
    result = {'real':0, 'fake':0}
    for example in formed_examples :
        if int(example['label'][0]) is 0 :
            result['fake'] += 1
        else :
            result['real'] += 1

    return result