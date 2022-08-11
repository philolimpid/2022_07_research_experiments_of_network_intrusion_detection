project_dir = r'E:\project\2022_07_research_of_network_intrusion_detection\2022_07_research_experiments_of_network_intrusion_detection'

kTrainProportion = 0.7
kValProportion = 0.1
kTestProportion = 0.2

random_seed = 101



input_feature_list = {}
input_feature_list['kdd99'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                  '13', '14', '15', '16', '17', '18', '20', '21', '22', '23', '24', '25',
                  '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                  '38', '39', '40']
input_feature_list['nb15'] = ['sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                      'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
                      'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
                      'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                      'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
input_feature_list['nb17'] = ['MI_dir_L5_weight','MI_dir_L5_mean','MI_dir_L5_variance','MI_dir_L3_weight',
                                      'MI_dir_L3_mean','MI_dir_L3_variance','MI_dir_L1_weight','MI_dir_L1_mean',
                                      'MI_dir_L1_variance','MI_dir_L0.1_weight','MI_dir_L0.1_mean','MI_dir_L0.1_variance',
                                      'MI_dir_L0.01_weight','MI_dir_L0.01_mean','MI_dir_L0.01_variance','H_L5_weight',
                                      'H_L5_mean','H_L5_variance','H_L3_weight','H_L3_mean','H_L3_variance','H_L1_weight',
                                      'H_L1_mean','H_L1_variance','H_L0.1_weight','H_L0.1_mean','H_L0.1_variance',
                                      'H_L0.01_weight','H_L0.01_mean','H_L0.01_variance','HH_L5_weight','HH_L5_mean',
                                      'HH_L5_std','HH_L5_magnitude','HH_L5_radius','HH_L5_covariance','HH_L5_pcc',
                                      'HH_L3_weight','HH_L3_mean','HH_L3_std','HH_L3_magnitude','HH_L3_radius',
                                      'HH_L3_covariance','HH_L3_pcc','HH_L1_weight','HH_L1_mean','HH_L1_std',
                                      'HH_L1_magnitude','HH_L1_radius','HH_L1_covariance','HH_L1_pcc','HH_L0.1_weight',
                                      'HH_L0.1_mean','HH_L0.1_std','HH_L0.1_magnitude','HH_L0.1_radius',
                                      'HH_L0.1_covariance','HH_L0.1_pcc','HH_L0.01_weight','HH_L0.01_mean',
                                      'HH_L0.01_std','HH_L0.01_magnitude','HH_L0.01_radius','HH_L0.01_covariance',
                                      'HH_L0.01_pcc','HH_jit_L5_weight','HH_jit_L5_mean','HH_jit_L5_variance','HH_jit_L3_weight',
                                      'HH_jit_L3_mean','HH_jit_L3_variance','HH_jit_L1_weight','HH_jit_L1_mean',
                                      'HH_jit_L1_variance','HH_jit_L0.1_weight','HH_jit_L0.1_mean','HH_jit_L0.1_variance',
                                      'HH_jit_L0.01_weight','HH_jit_L0.01_mean','HH_jit_L0.01_variance','HpHp_L5_weight',
                                      'HpHp_L5_mean','HpHp_L5_std','HpHp_L5_magnitude','HpHp_L5_radius','HpHp_L5_covariance',
                                      'HpHp_L5_pcc','HpHp_L3_weight','HpHp_L3_mean','HpHp_L3_std','HpHp_L3_magnitude',
                                      'HpHp_L3_radius','HpHp_L3_covariance','HpHp_L3_pcc','HpHp_L1_weight','HpHp_L1_mean',
                                      'HpHp_L1_std','HpHp_L1_magnitude','HpHp_L1_radius','HpHp_L1_covariance','HpHp_L1_pcc',
                                      'HpHp_L0.1_weight','HpHp_L0.1_mean','HpHp_L0.1_std','HpHp_L0.1_magnitude',
                                      'HpHp_L0.1_radius','HpHp_L0.1_covariance','HpHp_L0.1_pcc','HpHp_L0.01_weight',
                                      'HpHp_L0.01_mean','HpHp_L0.01_std','HpHp_L0.01_magnitude','HpHp_L0.01_radius',
                                      'HpHp_L0.01_covariance','HpHp_L0.01_pcc']
input_feature_list['nb17_danmini'] = input_feature_list['nb17']
input_feature_list['nb17_ecobee'] = input_feature_list['nb17']
input_feature_list['nb17_philips'] = input_feature_list['nb17']
input_feature_list['nb18'] = ['proto', 'seq', 'stddev', 'N_IN_Conn_P_SrcIP',
                              'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP',
                              'drate', 'srate', 'max']