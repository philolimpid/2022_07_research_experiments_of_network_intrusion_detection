#import python packages
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import general_parameters

np.random.seed(general_parameters.random_seed)

#load data
data_path = [r'\data\meta_data\NB15\UNSW-NB15_1.csv', r'\data\meta_data\NB15\UNSW-NB15_2.csv',
             r'\data\meta_data\NB15\UNSW-NB15_3.csv', r'\data\meta_data\NB15\UNSW-NB15_4.csv']
data_list = []
for i in range(4):
    inputdata = pd.read_csv(general_parameters.project_dir+data_path[i],header=None,low_memory=False)
    data_list.append(inputdata)
data = pd.concat(data_list,axis = 0, ignore_index = True)


#add column name
original_feature_list = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
                 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
data.columns = original_feature_list


#change data format, replace, fillna
print(pd.DataFrame(data.dtypes))

abandoned_feature_list = ['srcip', 'dstip', 'Stime', 'Ltime', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'Label']
data = data.drop(abandoned_feature_list, axis=1)

data['attack_cat'].fillna('0', inplace=True)
data['attack_cat'].replace(
    ["Fuzzers", "Reconnaissance", "Shellcode", "Analysis", "Backdoors", "DoS", "Exploits", "Generic", "Worms"],
    ["1", "2", "3", "4", "5", "6", "7", "8", "9"], inplace=True)
data['attack_cat'].replace(
    [" Fuzzers", " Reconnaissance", " Shellcode", " Analysis", " Backdoors", " DoS", " Exploits", " Generic", " Worms"],
    ["1", "2", "3", "4", "5", "6", "7", "8", "9"], inplace=True)

proto_list = list(set(list(data['proto'])))
state_list = list(set(list(data['state'])))
service_list = list(set(list(data['service'])))
proto_number = list(range(len(proto_list)))
state_number = list(range(len(state_list)))
service_number = list(range(len(service_list)))
data['proto'].replace(proto_list, proto_number, inplace=True)
data['state'].replace(state_list, state_number, inplace=True)
data['service'].replace(service_list, service_number, inplace=True)

data['sport'] = pd.to_numeric(data['sport'], errors='coerce')
data['dsport'] = pd.to_numeric(data['dsport'], errors='coerce')
data['attack_cat'] = pd.to_numeric(data['attack_cat'], errors='coerce')
data.dropna(inplace=True)

print('data_preprocessing finished')


#split data into data_train and data_test
data_shuffled1 = shuffle(data)
data_train = data_shuffled1.iloc[:int(data_shuffled1['attack_cat'].count()*general_parameters.kTrainProportion)]
data_val = data_shuffled1.iloc[int(data_shuffled1['attack_cat'].count()*general_parameters.kTrainProportion):
                               int(data_shuffled1['attack_cat'].count()*(
                                       general_parameters.kTrainProportion+general_parameters.kValProportion))]
data_test = data_shuffled1.iloc[int(data_shuffled1['attack_cat'].count()*(
        general_parameters.kTrainProportion+general_parameters.kValProportion)):]
data_train = data_train.reset_index(drop=True)
data_val = data_val.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)


#save the data into hdf5 files
data_train.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb15_data_train.h5',key='nb15_data_train')
data_val.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb15_data_val.h5',key='nb15_data_val')
data_test.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb15_data_test.h5',key='nb15_data_test')