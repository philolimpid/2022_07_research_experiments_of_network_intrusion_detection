project_dir = r'E:\project\2022_07_research_of_network_intrusion_detection\2022_07_research_experiments_of_network_intrusion_detection'

kTrainProportion = 0.7
kValProportion = 0.1
kTestProportion = 0.2

random_seed = 101


nb15_input_features = ['sport', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                      'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
                      'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
                      'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                      'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
kdd99_input_features = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                  '13', '14', '15', '16', '17', '18', '20', '21', '22', '23', '24', '25',
                  '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                  '38', '39', '40']