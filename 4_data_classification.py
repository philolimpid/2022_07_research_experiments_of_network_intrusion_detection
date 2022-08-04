import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import h5py
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from time import process_time
import importlib
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdate
import argparse

import general_parameters
import pipeline_definition

np.random.seed(general_parameters.random_seed)

plt.rcParams['font.sans-serif'] = ['STSONG']
#plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12

#parse some parameters
parser = argparse.ArgumentParser(description='deliver resampling parameters')
parser.add_argument('data_type',type=str)
parser.add_argument('resampling_type',type=str)
parser.add_argument('resampling_ratio',type=float)
parser.add_argument('version',type=str)
args = parser.parse_args()
print(args)
data_type = args.data_type
resampling_type = args.resampling_type
resampling_ratio = args.resampling_ratio
version = args.version

class class_of_data_block():
    def __init__(self):
        print('done')
        self.train = [0,1]
        self.val = [0,1]
        self.test = [0,1]
def load_data(data_type,resampling_type,resampling_ratio,input_feature_list):
    data = class_of_data_block()
    data_train = pd.read_hdf(general_parameters.project_dir + r'\data\intermediate_data\\' + 'data_train_' + data_type + '_' + resampling_type + '_' + str(
            resampling_ratio) + '.h5',
        key='data_train_' + data_type + '_' + resampling_type + '_' + str(resampling_ratio))
    data_val = pd.read_hdf(
        general_parameters.project_dir + r'\data\intermediate_data\\' + data_type + '_data_val_after_standardization.h5',
        key=data_type + '_data_val_after_standardization')
    data_test = pd.read_hdf(
        general_parameters.project_dir + r'\data\intermediate_data\\' + data_type + '_data_test_after_standardization.h5',
        key=data_type + '_data_test_after_standardization')
    data.train[0],data.train[1] = data_train[input_feature_list],data_train['attack_cat']
    data.val[0], data.val[1] = data_val[input_feature_list], data_val['attack_cat']
    data.test[0], data.test[1] = data_test[input_feature_list], data_test['attack_cat']
    return data
def load_scaled_feature(data_type):
    scaled_feature = pd.read_csv(general_parameters.project_dir+r'\data\intermediate_data\scaled_feature_'+data_type+'.csv')
    return scaled_feature
def one_hot(data_block):
    for i in ['train','val','test']:
        labels_dense = np.asarray(eval('data_block.'+i)[1]).astype(int)
        num_labels = len(eval('data_block.'+i)[1])
        num_classes = len(set(eval('data_block.'+i)[1].to_list()))
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        eval('data_block.'+i)[1] = labels_one_hot
    return data_block


for model in ['ANN', 'DTR']:
    if data_type == 'nb15':
        input_feature_list = ['sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
                              'dttl', 'sloss',
                              'dloss',
                              'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb',
                              'dtcpb',
                              'smeansz',
                              'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt',
                              'Dintpkt', 'tcprtt',
                              'synack',
                              'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst',
                              'ct_dst_ltm',
                              'ct_src_ltm',
                              'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
    elif data_type == 'kdd99':
        input_feature_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                              '13', '14', '15', '16', '17', '18', '20', '21', '22', '23', '24', '25',
                              '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                              '38', '39', '40']

    data_block = load_data(data_type, resampling_type, resampling_ratio, input_feature_list)
    scaled_feature = load_scaled_feature(data_type)
    data_block = one_hot(data_block)

    importlib.reload(pipeline_definition)
    if model == 'ANN':
        pipeline = \
            pipeline_definition.ANN_pipeline(model_name='ANN',
                                             data_block=data_block,
                                             input_feature_list=input_feature_list,
                                             scaled_feature=scaled_feature,
                                             data_type=data_type,
                                             resampling_type=resampling_type,
                                             resampling_ratio=resampling_ratio,
                                             version=version
                                             )
        pipeline.build_and_compile_model()
        pipeline.fit(5)
        pipeline.save_model_and_history()
        pipeline.load_model_and_history()
        pipeline.get_the_classification_results()
        pipeline.evaluate_the_results()
        pipeline.get_training_plot()
    elif model == 'DTR':
        pipeline = pipeline_definition.DTR_pipeline(model_name='DTR',
                                                    data_block=data_block,
                                                    input_feature_list=input_feature_list,
                                                    scaled_feature=scaled_feature,
                                                    data_type=data_type,
                                                    resampling_type=resampling_type,
                                                    resampling_ratio=resampling_ratio,
                                                    version=version)
        pipeline.change_data_block_shape()
        pipeline.build_and_compile_model()
        pipeline.fit()
        pipeline.get_the_classification_results()
        pipeline.evaluate_the_results()
