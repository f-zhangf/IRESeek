########################################################
#
#  
#   Process Seq Data
#
#
#########################################################
import os
import torch
import dgl
import pickle
import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from utils.rna_utils import load_seq,load_mat_bpe,load_mat_bpp,_convert2dglgraph
from utils.generate_bpp_bpe import fold_rna_from_file
def get_fasta_name(file_path):
    allowed_extensions = {'.fa', '.fasta'}
    matched_files = []
    for item in os.listdir(file_path):
        item_path = file_path + item
        
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in allowed_extensions:
                matched_files.append(item)
    return matched_files


def get_graph(file,bpe_data_path,bpp_data_path,save_path):
    id,seq = load_seq(file)
    fold_rna_from_file(file,bpp_data_path,bpe_data_path)
        
    #lable list
    label_list = np.array([-1] * len(id))
    graph_labels = torch.tensor(label_list)
        
    #bpe list
    bpe_matrix = load_mat_bpe(bpe_data_path,id)
    bpe_graph = _convert2dglgraph(seq, bpe_matrix)
        
    #bpp list
    bpp_matrix = load_mat_bpp(bpp_data_path,id)
    bpp_graph = _convert2dglgraph(seq,bpp_matrix)

    #ss list
    ss_graph = [dgl.graph(([], [])) for _ in len(label_list)]


    print("###################Begin generate pkl ###################")
    for idx, value in tqdm(enumerate(label_list), total=len(label_list), desc="Saving PKL files"):
        data = (id[idx],seq[idx],ss_graph[idx],bpe_graph[idx],bpp_graph[idx],value)
        with open(save_path +'/'+ id[idx]+'.pkl', 'wb') as f:
            pickle.dump(data, f)
    print("###################end generate pkl ###################")



if __name__ == '__main__':
    basedir = os.getcwd()
    #生成所有测试预处理数据
    test_save_path = basedir + "/all_test/newtest"
    get_graph(test_save_path)
    #生成训练集预处理数据
    #train_save_path = basedir + "/RNAFinalData/normal/train"
    #make_train_graph(train_save_path)
    #train_save_path = basedir + "/test_no_connect/undersampling/ensemble_3/train"
    #make_train_graph(train_save_path)