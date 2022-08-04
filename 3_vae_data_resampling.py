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

'''
#define some parameters
data_type = 'kdd99'
resampling_type = 'vae'
resampling_ratio = 0.01
version = '20220731_1'
'''

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



data_train = pd.read_hdf(general_parameters.project_dir+r'\data\intermediate_data\\'+data_type+'_data_train.h5',
                            key=data_type+'_data_train')
data_val = pd.read_hdf(general_parameters.project_dir+r'\data\intermediate_data\\'+data_type+'_data_val.h5',
                            key=data_type+'_data_val')
data_test = pd.read_hdf(general_parameters.project_dir+r'\data\intermediate_data\\'+data_type+'_data_test.h5',
                            key=data_type+'_data_test')


#read data into a dataframe
if data_type == 'nb15':
    input_feature_list = ['sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                      'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
                      'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
                      'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                      'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
elif data_type == 'kdd99':
    input_feature_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                  '13', '14', '15', '16', '17', '18', '20', '21', '22', '23', '24', '25',
                  '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                  '38', '39', '40']

resampling_ratio_table = pd.DataFrame(data_train.groupby('attack_cat').count()).iloc[:,0].reset_index()
resampling_ratio_table.columns = ['attack_cat','original']
resampling_ratio_table['target'] = int(resampling_ratio_table['original'].max()*resampling_ratio)
resampling_ratio_table['difference'] = resampling_ratio_table['target'] - resampling_ratio_table['original']
resampling_ratio_table = resampling_ratio_table[resampling_ratio_table['difference']>0]
sso = dict(zip(resampling_ratio_table['attack_cat'].to_list(),resampling_ratio_table['target'].to_list()))



def standardization_with_maxmin(train_df, val_df, test_df,input_feature_list):
    scaled_feature = {}
    for e in input_feature_list:
        min_value, max_value = (train_df[e].min(), train_df[e].max())
        scaled_feature[e] = [min_value, max_value]
        train_df.loc[:, e] = (train_df.loc[:, e] - min_value) / (max_value - min_value)
        val_df.loc[:, e] = (val_df.loc[:, e] - min_value) / (max_value - min_value)
        test_df.loc[:, e] = (test_df.loc[:, e] - min_value) / (max_value - min_value)
    return (train_df, val_df, test_df, scaled_feature)

(data_train, data_val, data_test, scaled_feature) = standardization_with_maxmin(
    data_train.copy(),data_val.copy(),data_test.copy(),input_feature_list)
pd.DataFrame(scaled_feature).to_csv(
        general_parameters.project_dir+r'\data\intermediate_data\scaled_feature_'+data_type+'.csv', index=False)
print('data_train_standardization finished')







class class_of_data_block():
    def __init__(self):
        print('done')
        self.train = [0,1]
        self.val = [0,1]
        self.test = [0,1]
def load_data(data_train, data_val, data_test,input_feature_list):
    data = class_of_data_block()
    data_train, data_val, data_test = data_train, data_val, data_test
    data.train[0],data.train[1] = data_train[input_feature_list],data_train['attack_cat']
    data.val[0], data.val[1] = data_val[input_feature_list], data_val['attack_cat']
    data.test[0], data.test[1] = data_test[input_feature_list], data_test['attack_cat']
    return data

def one_hot(data_block):
    for i in ['train','val','test']:
        labels_dense = np.asarray(eval('data_block.'+i)[1]).astype(int)
        num_labels = len(eval('data_block.'+i)[1])
        if data_type == 'kdd99':
            num_classes = 5
        elif data_type == 'nb15':
            num_classes = 10
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        eval('data_block.'+i)[1] = labels_one_hot
    return data_block


dense_data_block = load_data(data_train, data_val, data_test, input_feature_list)

for i in sso:
    print(i)
    #i=4
    data_block = class_of_data_block()
    data_block.train[0] = dense_data_block.train[0][dense_data_block.train[1] == i].copy()
    data_block.train[1] = dense_data_block.train[1][dense_data_block.train[1] == i].copy()
    data_block.test[0] = dense_data_block.test[0][dense_data_block.test[1] == i].copy()
    data_block.test[1] = dense_data_block.test[1][dense_data_block.test[1] == i].copy()
    data_block.val[0] = dense_data_block.val[0][dense_data_block.val[1] == i].copy()
    data_block.val[1] = dense_data_block.val[1][dense_data_block.val[1] == i].copy()

    data_block = one_hot(data_block)

    importlib.reload(pipeline_definition)
    VAE_pipeline = \
        pipeline_definition.VAE_pipeline(model_name='VAE',
                                         data_block=data_block,
                                         input_feature_list=input_feature_list,
                                         scaled_feature=scaled_feature,
                                         data_type=data_type,
                                         resampling_type=resampling_type,
                                         resampling_ratio=resampling_ratio,
                                         version=version
                                         )
    VAE_pipeline.build_model()
    '''
    for i in range(5):
        VAE_pipeline.compile_and_fit_model(i*10)
        VAE_pipeline.generate_new_samples(10)
    '''
    VAE_pipeline.compile_and_fit_model(100)
    VAE_pipeline.save_model_and_history()
    VAE_pipeline.load_model_and_history()
    VAE_pipeline.reconstruction()
    VAE_pipeline.generate_new_samples(sso[i])
    new_samples = pd.DataFrame(VAE_pipeline.new_samples.numpy())
    new_samples.columns = input_feature_list
    new_samples['attack_cat'] = i
    data_train = pd.concat([data_train, new_samples], axis=0)




data_train.to_hdf(
    general_parameters.project_dir+r'\data\intermediate_data\\'+'data_train_'+data_type+'_'+resampling_type+'_'+str(resampling_ratio)+'.h5',
    key='data_train_'+data_type+'_'+resampling_type+'_'+str(resampling_ratio))



