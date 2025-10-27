import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.utils import log_string,load_data,count_parameters,setup_seed
from utils.utils import masked_mae_loss
from model.model import STFQET
from model.train import train
from model.test import test

#使用GPU进行计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Traffic_Prediction_2')
parser.add_argument('--time_slot', type=int, default=15,
                    help='a time step is 15 mins')
parser.add_argument('--torch_seed', type=int, default=24,
                    help='the seed of torch')
parser.add_argument('--num_link', type=int, default=223,
                    help='The number of predicted road segments')
parser.add_argument('--num_his', type=int, default=8,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=8,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=2,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=100,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=8,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.003,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='./data/Traffic_speed_(24, 33).csv',
                    help='traffic file')
parser.add_argument('--query_file', default='./data/query_beijing_0201_grid_start_destination_count.npz',
                    help='query_file')
parser.add_argument('--SE_file', default='./data/Gf_grid_24_33_id.txt',
                    help='spatial embedding file')
parser.add_argument('--adj_file', default='./data/edge_list_grid_(24,33)_weight.txt',
                    help='graph file')
parser.add_argument('--row', default=24, help='row')
parser.add_argument('--col', default=33, help='column')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_file = f"./save_model/grid_24_33/Model_GF_test_{timestamp}.pth"
parser.add_argument('--model_file', default=model_file,
                    help='save the model to disk')
parser.add_argument('--log_file', default='./log/grid_24_33/log_test',
                    help='log_ss_sd_ed file')

args = parser.parse_args()
log = open(args.log_file, 'w')
log_string(log, str(args)[10: -1])
log_string(log, 'Using device: %s' % device)

(trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,
query_trainX,query_trainY,query_valX,query_valY,query_testX,query_testY,
periods, strengths, adj_matrix, SE, mean, std) = load_data(args,device)
log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
log_string(log, 'data loaded!')

del trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,query_trainX,query_trainY,query_valX,query_valY,query_testX,query_testY,periods, strengths, mean, std

log_string(log, 'compiling model...')
setup_seed(args.torch_seed)
model = STFQET(SE, args,adj_matrix, bn_decay=0.1).to(device)
loss_criterion = masked_mae_loss
optimizer = optim.Adam(model.parameters(), args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.decay_epoch,
                                      gamma=0.9)
parameters = count_parameters(model)
log_string(log, 'trainable parameters: {:,}'.format(parameters))


if __name__ == '__main__':
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler,device)
    trainPred, valPred, testPred = test(args, log,device)
    end = time.time()
    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()

