#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from flowcontainer.extractor import extract
import scapy.all as scapy
import math
import pdb
from multiprocessing import Pool
import csv
import json
# ========= 参数配置 =========
#input_dirs = [
#    "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj",]

input_dirs= ["/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/training_data/USTC-TFC2016/Facetime"]

output_dir = "/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary"
iat_out = os.path.join(output_dir, "iat_corpus_2_7.csv")
len_out = os.path.join(output_dir, "len_corpus_2_7.csv")
dir_out =  os.path.join(output_dir, "dir_corpus_2_7.csv")

BATCH_SIZE = 100      # 每多少个文件保存一次
max_packets = 40      # 每个流最多取前40个包
NUM_WORKERS = 1   


# ========= 写入文件 =========
def append_csv(path, rows):
    if not rows:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows(rows)

# ========= 提取流特征 =========
def get_flow_features(pcap_path, max_packets=40):
    length_list=[]
    iat_list=[]
    dir_list=[]
    total_packets=0
    print("pcap_file_path:-------------------------------------")
    print(pcap_path)
    # pdb.set_trace()
    try:
        packets = scapy.rdpcap(pcap_path)
        print(f"packets length:{len(packets)}")
        #pdb.set_trace()
        if len(packets) < 5:
            return -1, 0, 0  # 返回包数为0
        feature_result = extract(
            pcap_path,
            filter='tcp',
            extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type'],
            split_flag=True
        )
        
        if not feature_result:
            feature_result = extract(pcap_path, filter='udp')
            if not feature_result:
                return -1, 0, 0
    
        print(feature_result.keys())
        # pdb.set_trace()
        for flow_key,flow_obj in feature_result.items():
            total_packets += len(flow_obj.ip_lengths)  # 实际包数
            # 截取前 max_packets 个
            lengths = np.array(flow_obj.ip_lengths[:max_packets]).tolist()
            lengths = [abs(i) for i in lengths]
            directions = [int(np.sign(x)) for x in flow_obj.ip_lengths[:max_packets]]
            timestamps = np.array(flow_obj.ip_timestamps[:max_packets])
            if len(lengths)<5:
                print(f"flow contains {len(lengths)} pcpas, less then min value 5, which will be filter")
                continue
            #print(len(lengths),len(timestamps))
            iats = np.diff(timestamps).tolist() if len(timestamps) > 1 else [0]
            iats = [round(i*100,1) for i in iats]
            iats.insert(0,0.0)
            length_list.append(lengths)
            iat_list.append(iats)
            dir_list.append(directions)
        if len(length_list)>1:
            pass
        return (length_list, dir_list, iat_list), total_packets,pcap_path
    except Exception as e:
        return -1, 0, 0

def mark_processed(log_file_path,pcap_file_list):
    with open(log_file_path,'w') as f:
        json.dump(pcap_file_list,f)
        f.close()
        
def get_processed_pcapfile(log_file_path):
    try:
        with open(log_file_path,'r') as f:
            file_list = json.load(f)
            f.close()
        return file_list
    except:
        return []

# ========= 主程序 =========
if __name__=="__main__":
  
    len_buf, dir_buf, iat_buf, path_buf = [], [], [], []
    processed_log = os.path.join(output_dir, "processed_files.json")
    total_files = 0
    # 清空文件
    total_packet_list=[]
    total_flows=0
    flow_cnt=0
    pcap_file_list=[]
    processed_pcap_files=get_processed_pcapfile(processed_log)
    for input_dir in input_dirs:
        print(f"\n[+] Processing directory: {input_dir}")
        dir_files, dir_packets = 0, 0
        for root, dirs, files in os.walk(input_dir):
            for file in sorted(files):
                if not file.endswith(".pcap") :
                    continue
                # if file in total_packet_list:
                #     continue   
                total_packet_list.append(file)
                pcap_path = os.path.join(root, file)
                if pcap_path in processed_pcap_files:
                    continue 
                pcap_file_list.append(pcap_path)
                
    print(f"pcaps number:{len(pcap_file_list)}")
    import pdb
    # pdb.set_trace()
                
    with Pool(NUM_WORKERS) as pool:
        
        for result in pool.imap_unordered(get_flow_features, pcap_file_list):
            print(f"result: {result}")
            feature_results,total_pcaps,fpath= result[0],result[1],result[2]
            if feature_results==-1:
                continue
            print(f"feature_results: {feature_results}")
            path_buf.append(fpath)
            lengths, directions, iats = feature_results
            for i in range(len(lengths)):
                len_buf.append(lengths[i])
                dir_buf.append(directions[i])
                iat_buf.append(iats[i])
            flow_cnt += 1
            print(f"[+] Flow {flow_cnt}: {fpath}")
            if flow_cnt % BATCH_SIZE == 0:
                append_csv(len_out, len_buf)
                append_csv(dir_out, dir_buf)
                append_csv(iat_out, iat_buf)
                mark_processed(processed_log, path_buf)

                print(f"[✓] {flow_cnt} flows written.")

                len_buf, dir_buf, iat_buf, path_buf = [], [], [], []



