#import packages
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import general_parameters

np.random.seed(general_parameters.random_seed)

data_path = [r'\data\meta_data\NB17_ecobee\ecobee_thermostat_benign_traffic.csv',
             r'\data\meta_data\NB17_ecobee\mirai_attacks\ack.csv',
             r'\data\meta_data\NB17_ecobee\mirai_attacks\scan.csv',
             r'\data\meta_data\NB17_ecobee\mirai_attacks\syn.csv',
             r'\data\meta_data\NB17_ecobee\mirai_attacks\udp.csv',
             r'\data\meta_data\NB17_ecobee\mirai_attacks\udpplain.csv',
             r'\data\meta_data\NB17_ecobee\gafgyt_attacks\combo.csv',
             r'\data\meta_data\NB17_ecobee\gafgyt_attacks\junk.csv',
             r'\data\meta_data\NB17_ecobee\gafgyt_attacks\scan.csv',
             r'\data\meta_data\NB17_ecobee\gafgyt_attacks\tcp.csv',
             r'\data\meta_data\NB17_ecobee\gafgyt_attacks\udp.csv',]
data_list = []
for i in range(len(data_path)):
    inputdata = pd.read_csv(general_parameters.project_dir+data_path[i],header='infer',engine='python')
    inputdata['attack_cat'] = i
    data_list.append(inputdata)
data = pd.concat(data_list,axis = 0, ignore_index = True)


feature_and_label_list = list(data.columns)
feature_list = feature_and_label_list[:-1]
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


data_train.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb17_ecobee_data_train.h5',key='nb17_ecobee_data_train')
data_val.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb17_ecobee_data_val.h5',key='nb17_ecobee_data_val')
data_test.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\nb17_ecobee_data_test.h5',key='nb17_ecobee_data_test')