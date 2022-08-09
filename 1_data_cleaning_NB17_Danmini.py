#import packages
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import general_parameters

np.random.seed(general_parameters.random_seed)

#load data
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\danmini_doorbell_benign_traffic.csv'
data0 = pd.read_csv(data_path,header='infer',engine = 'python')
data0['attack_cat'] = 0
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\mirai_attacks\ack.csv'
data1 = pd.read_csv(data_path,header='infer',engine = 'python')
data1['attack_cat'] = 1
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\mirai_attacks\scan.csv'
data2 = pd.read_csv(data_path,header='infer',engine = 'python')
data2['attack_cat'] = 2
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\mirai_attacks\syn.csv'
data3 = pd.read_csv(data_path,header='infer',engine = 'python')
data3['attack_cat'] = 3
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\mirai_attacks\udp.csv'
data4 = pd.read_csv(data_path,header='infer',engine = 'python')
data4['attack_cat'] = 4
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\mirai_attacks\udpplain.csv'
data5 = pd.read_csv(data_path,header='infer',engine = 'python')
data5['attack_cat'] = 5
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\gafgyt_attacks\combo.csv'
data6 = pd.read_csv(data_path,header='infer',engine = 'python')
data6['attack_cat'] = 6
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\gafgyt_attacks\junk.csv'
data7 = pd.read_csv(data_path,header='infer',engine = 'python')
data7['attack_cat'] = 7
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\gafgyt_attacks\scan.csv'
data8 = pd.read_csv(data_path,header='infer',engine = 'python')
data8['attack_cat'] = 8
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\gafgyt_attacks\tcp.csv'
data9 = pd.read_csv(data_path,header='infer',engine = 'python')
data9['attack_cat'] = 9
data_path = r'E:\project\researchwithdb\data\danmini_doorbell\gafgyt_attacks\udp.csv'
data10 = pd.read_csv(data_path,header='infer',engine = 'python')
data10['attack_cat'] = 10

data = pd.concat([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10],axis=0,ignore_index = True)

data_path = [r'\data\meta_data\NB17_danmini\danmini_doorbell_benign_traffic.csv',
             r'\data\meta_data\NB17_danmini\mirai_attacks\ack.csv',
             r'\data\meta_data\NB17_danmini\mirai_attacks\scan.csv',
             r'\data\meta_data\NB17_danmini\mirai_attacks\syn.csv',
             r'\data\meta_data\NB17_danmini\mirai_attacks\udp.csv',
             r'\data\meta_data\NB17_danmini\mirai_attacks\udpplain.csv',
             r'\data\meta_data\NB17_danmini\gafgyt_attacks\combo.csv',
             r'\data\meta_data\NB17_danmini\gafgyt_attacks\junk.csv',
             r'\data\meta_data\NB17_danmini\gafgyt_attacks\scan.csv',
             r'\data\meta_data\NB17_danmini\gafgyt_attacks\tcp.csv',
             r'\data\meta_data\NB17_danmini\gafgyt_attacks\udp.csv',]
data_list = []
for i in range(len(data_path)):
    inputdata = pd.read_csv(general_parameters.project_dir+data_path[i],header='infer',engine='python')
    inputdata['attack_cat'] = i
    data_list.append(inputdata)
data = pd.concat(data_list,axis = 0, ignore_index = True)




label = pd.read_csv(general_parameters.project_dir+r'\data\meta_data\KDD99\training_attack_types.txt', sep=' ', header=None, engine='python')
label[1].replace(['probe', 'dos', 'u2r', 'r2l'], ["1", "2", "3", "4"], inplace=True)
label[0] = label[0]+'.'

#define some lists of features. Prepare for column selection
feature_list1 = []
for i in range(41):
    feature_list1.append(str(i))
feature_list1.append('attack_cat')
feature_list3 = []
for i in range(41):
    feature_list3.append(str(i))
feature_list3.remove('19')

protocol_list = ['tcp', 'udp', 'icmp']

service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
                'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
                'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
                'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
                'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
                'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
                'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']

flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']



#drop some columns, data format conversion, string to int
data.columns = feature_list1
data = data.drop(['19'], axis=1)
# data['attack_cat'].fillna('9',inplace = True)

data['1'].replace(protocol_list, ["0", "1", "2"], inplace=True)
data['2'].replace(service_list,
                  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                   "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
                   "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46",
                   "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
                   "62", "63", "64", "65", "66", "67", "68", "69"], inplace=True)
data['3'].replace(flag_list, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], inplace=True)
# data['attack_cat'].replace(label_list,["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"],inplace=True)
# data['attack_cat'].replace(label[0].tolist(),label[2].tolist(),inplace=True)
data['attack_cat'].replace(label[0].tolist(), label[1].tolist(), inplace=True)
data['attack_cat'].replace('normal.',0,inplace=True)

# data.dtypes
data['1'] = pd.to_numeric(data['1'], errors='coerce')
data['2'] = pd.to_numeric(data['2'], errors='coerce')
data['3'] = pd.to_numeric(data['3'], errors='coerce')
data['attack_cat'] = pd.to_numeric(data['attack_cat'], errors='coerce')

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


data_train.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\kdd99_data_train.h5',key='kdd99_data_train')
data_val.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\kdd99_data_val.h5',key='kdd99_data_val')
data_test.to_hdf(general_parameters.project_dir+r'\data\intermediate_data\kdd99_data_test.h5',key='kdd99_data_test')