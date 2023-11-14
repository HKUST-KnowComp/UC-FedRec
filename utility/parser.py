# This file is based on the NGCF author's implementation
# <https://github.com/xiangwang1223/neural_graph_collaborative_filtering/blob/master/NGCF/utility/parser.py>.

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    parser.add_argument('--log_path', nargs='?', default='/home/qhuaf/graph_pri/logs/',
                        help='Store log path')
    parser.add_argument('--data_path', nargs='?', default='./data/recommendation/',
                        help='Input data path.')
    parser.add_argument('--model_name', type=str, default='two_side_graph',
                        help='model name.')
    parser.add_argument('--save_name', type=str, default='model.pkl',
                        help='saved model name')

    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset from {ml-1m, douban}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--pri_epoch', type=int, default=200,
                        help='Number of privacy part epoch')

    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[128]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout '
                             '(i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, '
                             'indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity'
                             ' levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--local_epoch', type=int, default=10,
                        help='the epoch number for local user model training')
    parser.add_argument('--num_process', type=int, default=4,
                        help='the number of processes for local user simulation')

    parser.add_argument('--local_batch_size', type=int, default=256,
                        help='local item batch size for training')
    parser.add_argument('--user_batch_size', type=int, default=256,
                        help='User batch size for parallel training.')

    parser.add_argument('--num_neighbor', type=int, default=10,
                        help='Specify the number of local rec model neigbor users')
    parser.add_argument('--item_threshold', type=int, default=30, help='Specify the number of predicted item number')
    parser.add_argument('--item_density_threshold', type=float, default=0.01,
                        help='Specify the density of predicted item')
    parser.add_argument('--pretrain_epoch', type=int, default=300,
                        help='the number of epochs for pretraining')
    parser.add_argument('--privacy_protect', type=bool, default=False,
                        help='Whether enable the privacy protection')
    parser.add_argument('--lmbda', type=float, default=1,
                        help='the weight for privacy protection')
    parser.add_argument('--log_name', type=str, default='test',
                        help='the name for log file')

    parser.add_argument('--privacy_ratio', type=float, default=0.2,
                        help='the ratio of private attribute')
    parser.add_argument('--privacy_tradeoff', type=float, default=1,
                        help='the tradeoff of performance and privacy')

    parser.add_argument('--dp_clip', type=float, default=5)
    parser.add_argument('--dp_eps', type=float, default=8.0)
    parser.add_argument('--dp_delta', type=float, default=1e-5)
    parser.add_argument('--dp_mechanism', type=str, default='laplace')
    args = parser.parse_args()

    args.mess_dropout = eval(args.mess_dropout)
    args.layer_size = eval(args.layer_size)
    args.regs = eval(args.regs)

    return args
