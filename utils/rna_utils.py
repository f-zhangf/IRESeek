import os
import sys
import torch
import dgl
import gzip
from tqdm import tqdm
import numpy as np
def load_fasta_format(file):
    all_id = []
    all_seq = []
    seq = ''
    for row in file:
        if type(row) is bytes:
            row = row.decode('utf-8')
        row = row.rstrip()
        if row.startswith('>'):
            all_id.append(row.lstrip('>'))
            if seq != '':
                all_seq.append(seq)
                seq = ''
        else:
            seq += row
    all_seq.append(seq)
    return all_id, all_seq

def load_seq_bppe(filepath):
    if filepath.endswith(('.fa',',fasta')):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    for i in range(len(all_seq)):
        seq = all_seq[i]
        all_seq[i] = seq.upper().replace('T', 'U')
    return all_id, all_seq

def load_seq(filepath):
    if filepath.endswith(('.fa',',fasta')):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    for i in range(len(all_seq)):
        seq = all_seq[i]
        all_seq[i] = seq.upper()
    return all_id, all_seq


def _constructGraph(seq, csr_matrix):
    seq_upper = seq.upper()
    d = {'A': torch.tensor([[1., 0., 0., 0.]]),
         'G': torch.tensor([[0., 1., 0., 0.]]),
         'C': torch.tensor([[0., 0., 1., 0.]]),
         'U': torch.tensor([[0., 0., 0., 1.]]),
         'T': torch.tensor([[0., 0., 0., 1.]])}
    grh = dgl.DGLGraph(csr_matrix)
    grh.ndata['feat'] = torch.zeros((grh.number_of_nodes(), 4))
    for i in range(len(seq)):
        grh.ndata['feat'][i] = d[seq_upper[i]]
    grh.edata['feat'] = torch.tensor(csr_matrix.data)
    grh.edata['feat'] = grh.edata['feat'].unsqueeze(1) 
    for i in range(len(seq)):
        for j in range(-2, 3):
            if j == 0 or j == -1 or j == 1:
                continue
            if i + j < 0 or i + j > len(seq) - 1:
                continue
            if grh.has_edges_between(i, i + j):
                continue
            grh.add_edges(i, i + j)
            grh.edges[i, i + j].data['feat'] = torch.tensor([[1/j]], dtype=torch.float64)
    return grh

def _convert2dglgraph(seq_list, csr_matrixs):
    dgl_graph_list = []
    for i in tqdm(range(len(csr_matrixs))):
        dgl_graph_list.append(_constructGraph(seq_list[i], csr_matrixs[i]))
    return dgl_graph_list
#secondary structure Graph
def _constructSsGraph(seq, csr_matrix):
    seq_upper = seq.upper()
    d = {'A': torch.tensor([[1., 0., 0., 0.]]),
         'G': torch.tensor([[0., 1., 0., 0.]]),
         'C': torch.tensor([[0., 0., 1., 0.]]),
         'U': torch.tensor([[0., 0., 0., 1.]]),
         'T': torch.tensor([[0., 0., 0., 1.]])}
    grh = dgl.DGLGraph(csr_matrix)
    grh.ndata['feat'] = torch.zeros((grh.number_of_nodes(), 4))
    for i in range(len(seq)):
        grh.ndata['feat'][i] = d[seq_upper[i]]
    grh.edata['feat'] = torch.tensor(csr_matrix.data)
    grh.edata['feat'] = grh.edata['feat'].unsqueeze(1) 
    return grh
def _convert2ssdglgraph(seq_list, csr_matrixs):
    dgl_graph_list = []
    for i in tqdm(range(len(csr_matrixs))):
        dgl_graph_list.append(_constructSsGraph(seq_list[i], csr_matrixs[i]))
    return dgl_graph_list

#bpe
def load_mat_bpe(file_path,seq_id):
    np.set_printoptions(threshold=np.inf)
    sp_eng_matrix = []
    np.set_printoptions(suppress=True, precision=8,threshold=np.inf)
    print("seq_id len:",len(seq_id))
    for id in seq_id:
        id = id.split("_")[0]
        #print("id:",id) 
        m_file_path = file_path + '/' + id + '.npy'
        if not os.path.exists(m_file_path):
            print(f"文件 '{m_file_path}' 不存在，请检查路径！")
            continue
        data = np.load(m_file_path).squeeze(0)
        for i in range(174 - 1):
            data[i,i+1] = 1.
            data[i+1,i] = 1.
        sp_eng_matrix.append(sp.csc_matrix(data))
    print("sp_eng_matrix size:",len(sp_eng_matrix))
    return sp_eng_matrix

#bpp
def load_mat_bpp(file_path,seq_id):
    np.set_printoptions(threshold=np.inf)
    sp_bpp_matrix = []
    np.set_printoptions(suppress=True, precision=8,threshold=np.inf)
    print("seq_id len:",len(seq_id))
    for id in seq_id:
        id = id.split("_")[0]
        #print("id:",id) 
        m_file_path = file_path + '/' + id + '.npy'
        if not os.path.exists(m_file_path):
            print(f"文件 '{m_file_path}' 不存在，请检查路径！")
            continue
        data = np.load(m_file_path)
        sp_bpp_matrix.append(sp.csc_matrix(data))
    print("sp_bpp_matrix size:",len(sp_bpp_matrix))
    return sp_bpp_matrix

#secondary structure
def load_mat_ss(file_path,seq_id):
    np.set_printoptions(threshold=np.inf)
    sp_ss_matrix = []
    np.set_printoptions(suppress=True, precision=8,threshold=np.inf)
    print("seq_id len:",len(seq_id))
    for id in seq_id:
        id = id.split("_")[0]
        #print("id:",id) 
        m_file_path = file_path + '/' + id + '.npy'
        if not os.path.exists(m_file_path):
            print(f"文件 '{m_file_path}' 不存在，请检查路径！")
            continue
        data = np.load(m_file_path).squeeze(0)
        sp_ss_matrix.append(sp.csc_matrix(data))
    print("sp_ss_matrix size:",len(sp_ss_matrix))
    return sp_ss_matrix

if __name__ == "__main__":
    print("test")
