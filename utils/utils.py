import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import pickle
from scipy import signal

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / label
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)
    return mae, rmse, mape


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def masked_mae_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def query2instance(data, num_his, num_pred):
    num_step, row,col,dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, row,col,dims)
    y = torch.zeros(num_sample, num_pred, row,col,dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

#利用傅里叶变换，提取周期特征
def extract_spatiotemporal_periods(data, k):
    data = np.nan_to_num(data) 
    data -= np.mean(data, axis=0)

    total_timesteps = data.shape[0]
    global_magnitude = np.zeros(total_timesteps // 2)

    for node_idx in range(data.shape[1]):
        node_fft = np.fft.fft(data[:, node_idx])
        magnitudes = np.abs(node_fft[:total_timesteps // 2]) * 2 / total_timesteps
        global_magnitude += magnitudes

    global_magnitude /= np.max(global_magnitude)

    peaks, _ = signal.find_peaks(global_magnitude[1:], height=0.1)
    peak_indices = peaks + 1

    top_k_indices = peak_indices[np.argsort(-global_magnitude[peak_indices])[:k]]
    periods = total_timesteps / top_k_indices
    strengths = global_magnitude[top_k_indices]

    sorted_indices = np.argsort(-strengths)
    return periods[sorted_indices], strengths[sorted_indices]


def load_adjacency_matrix(file_path,traffic_data):
    adj_list = pd.read_csv(file_path, header=None, names=['node1', 'node2', 'weight'],sep=r'\s+')
    num_nodes = traffic_data.shape[1]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for _, row in adj_list.iterrows():
        i, j, w = int(row['node1']), int(row['node2']), float(row['weight'])
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w
    return adj_matrix


def Time_Embedding(num_timesteps, periods):

    TE = np.zeros((num_timesteps, len(periods)), dtype=np.int32)
    for col_idx, period in enumerate(periods):
        if period <= 0:
            raise ValueError(f"Period must be a positive integer {period}")
        TE[:, col_idx] = [t % period for t in range(num_timesteps)]

    return torch.from_numpy(TE)

def process_query_data(query_data, time_slot=15):
    num_regions_row, num_regions_col, num_timesteps, _ = query_data.shape
    num_new_timesteps = num_timesteps // 3

    processed_data = np.zeros((num_new_timesteps, num_regions_row, num_regions_col, 2))

    for t in range(num_new_timesteps):
        start_idx = t * 3
        end_idx = start_idx + 3
        processed_data[t] = np.sum(query_data[:, :, start_idx:end_idx, :], axis=2)

    return processed_data

def load_data(args,device):
    traffic_speed = pd.read_csv(args.traffic_file)
    traffic_speed = torch.tensor(traffic_speed.values)
    periods, strengths = extract_spatiotemporal_periods(traffic_speed, k=3)

    with open(args.SE_file, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
           temp = line.split(' ')
           index = int(temp[0])
           SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    query_data = np.load(args.query_file)['Query_ST_data']
    query_data = process_query_data(query_data, args.time_slot)
    row, col = args.row, args.col
    query_data = query_data[:,row - 2:row + 3, col - 2:col + 3,:]
    query_data = torch.from_numpy(query_data).float()
    adj_matrix = load_adjacency_matrix(args.adj_file, traffic_speed)

    TE = Time_Embedding(traffic_speed.shape[0], periods)

    num_step = traffic_speed.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train_traffic =traffic_speed[:train_steps]
    val_traffic = traffic_speed[train_steps:train_steps+val_steps]
    test_traffic = traffic_speed[train_steps+val_steps:]

    query_train=query_data[:train_steps]
    query_val = query_data[train_steps:train_steps+val_steps]
    query_test = query_data[train_steps+val_steps:]
    query_trainX, query_trainY = query2instance(query_train, args.num_his, args.num_pred)
    query_valX, query_valY = query2instance(query_val, args.num_his, args.num_pred)
    query_testX, query_testY = query2instance(query_test, args.num_his, args.num_pred)

    trainX, trainY = seq2instance(train_traffic, args.num_his, args.num_pred)
    valX, valY = seq2instance(val_traffic, args.num_his, args.num_pred)
    testX, testY = seq2instance(test_traffic, args.num_his, args.num_pred)

    mean, std = torch.mean(trainX), torch.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    train = TE[: train_steps]
    val = TE[train_steps: train_steps + val_steps]
    test = TE[-test_steps:]
    trainTE = seq2instance(train, args.num_his, args.num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    valTE = seq2instance(val, args.num_his, args.num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    testTE = seq2instance(test, args.num_his, args.num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)

    return [torch.tensor(x).to(device) if isinstance(x, np.ndarray) else x.to(device)
            for x in (trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE,
                      query_trainX, query_trainY, query_valX, query_valY, query_testX, query_testY,
                      periods, strengths, adj_matrix, SE, mean, std)]