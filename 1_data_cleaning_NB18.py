#import packages
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import general_parameters

np.random.seed(general_parameters.random_seed)

#load data
data_path = general_parameters.project_dir+r'\data\meta_data\NB18\UNSW_2018_IoT_Botnet_Final_10_Best.csv'
data = pd.read_csv(data_path,sep = ';',header='infer', engine = 'python')
data = data.drop('Unnamed: 0',axis=1)

#define some lists of features. Prepare for column selection
feature_and_label_list = list(data.columns)
feature_list = ['proto', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP', 'drate', 'srate', 'max']

proto_list = ['arp', 'icmp', 'ipv6-icmp', 'tcp', 'udp']
subcategory_list = ['Normal', 'Data_Exfiltration', 'HTTP', 'Keylogging', 'OS_Fingerprint', 'Service_Scan', 'TCP', 'UDP']
proto_number = list(range(len(proto_list)))
subcategory_number = list(range(len(subcategory_list)))
data['proto'].replace(proto_list, proto_number, inplace=True)
data['subcategory'].replace(subcategory_list, subcategory_number, inplace=True)
data.rename({'subcategory':'attack_cat'},axis=1,inplace=True)

data.dropna(inplace=True)



#split data into data_training and data_testing
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


data_train.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb18_data_train.h5',key='nb18_data_train')
data_val.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb18_data_val.h5',key='nb18_data_val')
data_test.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb18_data_test.h5',key='nb18_data_test')