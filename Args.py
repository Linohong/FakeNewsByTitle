import argparse

# ARGUMENT PART
parser = argparse.ArgumentParser(description='FakeNewsByTitle Argument Parser')
# option
parser.add_argument('-device', type=str, default='cuda', help='cpu/cuda')
parser.add_argument('-predefined', type=bool, default=False, help='enable/disable predefined word embedding')
parser.add_argument('-score', type=float, help='score threshold')


# model
parser.add_argument('-hidden_size', type=int, default=300)
parser.add_argument('-max_sent', type=int, default=64, help='max sentence length')
parser.add_argument('-sent_num', type=int, default=64, help='max number of sentences')
parser.add_argument('-abs_num', type=int, default=2, help='max number of abstract sentences')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-model', type=str, default='vanilla')
parser.add_argument('-model_name', type=str, default='NONE')
parser.add_argument('-enc_unit', type=str, default='syll')
parser.add_argument('-dec_unit', type=str, default='syll')


# learning
parser.add_argument('-epoch', type=int, default=1000)
parser.add_argument('-batch_size', type=int, default=1, help='batch size for training [default: 1]')
parser.add_argument('-learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('-kfold', type=int, default=10, help='k-folding size')
parser.add_argument('-early', type=int, default=None)
parser.add_argument('-optim', type=str, default='Adam')
parser.add_argument('-exam_unit', type=str, default='word')

# Data
parser.add_argument('-data_path', type=str, default='./Data/finished_files/chunked/train*', help='data location relative to the main file')
parser.add_argument('-voca_path', type=str, default='./Data/finished_files/vocab', help='vocab location relative to the main file')
parser.add_argument('-embed_path', type=str, default='../GoogleNewsVec/GoogleNews-vectors-negative300.bin')
parser.add_argument('-vocab_size', type=int, default=50000, help='size of the unique vocabulary')
parser.add_argument('-train_size', type=int, default=100, help='train size')
parser.add_argument('-task', type=str, default='train')
parser.add_argument('-files_to_read', type=int, default=10, help='the number of files to read for test data')
parser.add_argument('-eval_size', type=int, default=100000, help='eval_size of inputs to be evaluated')
parser.add_argument('-eval_iter', type=int, default=1, help='the number of iterations of evaluation')
args = parser.parse_args()