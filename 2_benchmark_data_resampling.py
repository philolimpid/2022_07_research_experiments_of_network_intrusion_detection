#import python packages
import pandas as pd
import numpy as np
import argparse

import general_parameters

np.random.seed(general_parameters.random_seed)

#parse some parameters
parser = argparse.ArgumentParser(description='deliver resampling parameters')
parser.add_argument('data_type',type=str)
parser.add_argument('resampling_type',type=str)
parser.add_argument('resampling_ratio',type=float)
args = parser.parse_args()
print(args)
data_type = args.data_type
resampling_type = args.resampling_type
resampling_ratio = args.resampling_ratio



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


def benchmark_resampling(mode,data,input_feature_list,output_feature):
    print(mode+" started")
    label = data[output_feature]
    features = data[input_feature_list]

    if mode == 'random_oversampling':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(sampling_strategy=sso, random_state=general_parameters.random_seed)
        f_resampled, l_resampled = ros.fit_resample(features, label)
    elif mode == 'smote':
        from imblearn.over_sampling import SMOTE
        f_resampled, l_resampled = SMOTE(sampling_strategy=sso).fit_resample(features, label)
    elif mode == 'adasyn':
        from imblearn.over_sampling import ADASYN
        aos = ADASYN(sampling_strategy=sso, random_state=general_parameters.random_seed)
        f_resampled, l_resampled = aos.fit_resample(features, label)
    elif mode == 'borderlinesmote':
        from imblearn.over_sampling import BorderlineSMOTE
        aos = BorderlineSMOTE(sampling_strategy=sso, random_state=general_parameters.random_seed)
        f_resampled, l_resampled = aos.fit_resample(features, label)
    elif mode == 'no_resampling':
        f_resampled = features.copy()
        l_resampled = label.copy()

    from collections import Counter
    print(sorted(Counter(l_resampled).items()))

    data_train_after_resampling = pd.concat([f_resampled, l_resampled], axis=1)
    print(mode+" finished")
    return data_train_after_resampling


data_train_after_resampling = benchmark_resampling(resampling_type,data_train,input_feature_list,output_feature='attack_cat')
print('data_train_preprocessing finished')


data_train_after_resampling.to_hdf(
    general_parameters.project_dir+r'\data\intermediate_data\\'+'data_train_'+data_type+'_'+resampling_type+'_'+str(resampling_ratio)+'.h5',
    key='data_train_'+data_type+'_'+resampling_type+'_'+str(resampling_ratio))
data_val.to_hdf(
    general_parameters.project_dir+r'\data\intermediate_data\\'+data_type+'_data_val_after_standardization.h5',
    key=data_type+'_data_val_after_standardization')
data_test.to_hdf(
    general_parameters.project_dir+r'\data\intermediate_data\\'+data_type+'_data_test_after_standardization.h5',
    key=data_type+'_data_test_after_standardization')
