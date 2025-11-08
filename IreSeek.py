import os
import sys
import subprocess
import torch
import time
import argparse
import dgl
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.load_net import get_model 
from data.RNAGraph import RNADataset,collate_all
from utils.generate_bpp_bpe import fold_rna_from_file
from utils.process_data import get_graph
########################################################
#
#
#   Predict IRES
#
#
#########################################################
d = {'A': torch.tensor([[1., 0., 0., 0.]]),
     'G': torch.tensor([[0., 1., 0., 0.]]),
     'C': torch.tensor([[0., 0., 1., 0.]]),
     'U': torch.tensor([[0., 0., 0., 1.]]),
     'T': torch.tensor([[0., 0., 0., 1.]])}


def save_date(save_path,id,seq,scores,lables):
    
    save_dir = save_path
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    df = pd.DataFrame({
    "ID": id,
    "Sequence": seq,
    "Label": lables,
    "Score": scores
})

    df.to_csv(save_dir + "/data.csv", index=False)

def select_free_gpu():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,nounits,noheader"],
            stderr=subprocess.STDOUT
        )
        result = result.decode("utf-8").strip().splitlines()
        #print(result)
        #sys.exit()
        gpu_info = []
        for line in result:
            index, free_mem, total_mem = line.split(', ')
            free_mem = int(free_mem)
            total_mem = int(total_mem)
            gpu_info.append((index, free_mem, total_mem))
        print(gpu_info)
        gpu_info = sorted(gpu_info, key=lambda x: x[1], reverse=True)
        selected_gpu_id = gpu_info[0][0]
        free_memory = gpu_info[0][1]
        print(f"Selected GPU: {selected_gpu_id}, Free Memory: {free_memory} MiB")
        return selected_gpu_id

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e.output.decode()}")
        return None  
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:' ,torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def device_set():
    #gpu_id = select_free_gpu()
    #if gpu_id is not None:
        #gpu_use = True
        #print(f"Selected GPU: {gpu_id}")
    #else:
         #gpu_use = False
         #print("No suitable GPU found.")
    gpu_id = 0 
    gpu_use = torch.cuda.is_available()
    return gpu_id,gpu_setup(gpu_use,gpu_id)
def load_model(MODEL_NAME, net_params,path,device):

    model = get_model(MODEL_NAME, net_params)
    PATH = path
    model.load_state_dict(torch.load(PATH,map_location=device))
    return model

def soft_voting(models,device, data_loader, use_fea,path = "",use_softmax=False):
    for model in models:
        model.eval()
    batch_id_array,batch_seq_array,batch_scores_array, batch_labels_array = np.array([]), np.array([]),np.array([]),np.array([])
    start_flag = 1
    with torch.no_grad():
        for iter, (batch_ids,batch_seqs,labels,ss_graphs,bpe_graphs,bpp_graphs) in enumerate(data_loader):
            bpe_batch_graphs = dgl.batch(bpe_graphs)
            bpe_batch_graphs = bpe_batch_graphs.to(device)
            bpe_batch_graphs.ndata['feat'] = bpe_batch_graphs.ndata['feat'].to(device)
            bpe_batch_graphs.edata['feat'] = bpe_batch_graphs.edata['feat'].to(device)
            
            batch_x = bpe_batch_graphs.ndata['feat'] # num x feat

            bpp_batch_graphs = dgl.batch(bpp_graphs)
            bpp_batch_graphs = bpp_batch_graphs.to(device)
            bpp_batch_graphs.ndata['feat'] = bpp_batch_graphs.ndata['feat'].to(device)
            bpp_batch_graphs.edata['feat'] = bpp_batch_graphs.edata['feat'].to(device)

            ss_batch_graphs = dgl.batch(ss_graphs)
            ss_batch_graphs = ss_batch_graphs.to(device)
            ss_batch_graphs.ndata['feat'] = ss_batch_graphs.ndata['feat'].to(device)
            ss_batch_graphs.edata['feat'] = ss_batch_graphs.edata['feat'].to(device)
          
            batch_labels = labels.to(device)
       
            batch_scores =torch.stack([model.forward(batch_x,bpe_batch_graphs,None,bpp_batch_graphs) for model in models],0).mean(0)

            batch_id_numpy = batch_ids
            batch_seq_numpy = batch_seqs
            
            if use_softmax:
                batch_scores_numpy = F.softmax(batch_scores, dim=1).detach().cpu().numpy()
            else :
                batch_scores_numpy = batch_scores.squeeze().detach().cpu().numpy() 
            batch_labels_numpy = batch_labels.cpu().numpy()
            if start_flag:
                batch_id_array,batch_seq_array= batch_id_numpy,batch_seq_numpy
                batch_scores_array, batch_labels_array = batch_scores_numpy, batch_labels_numpy
                start_flag = 0
            else:
                batch_id_array = np.hstack((batch_id_array, batch_id_numpy))
                batch_seq_array = np.hstack((batch_seq_array, batch_seq_numpy))                
                batch_scores_array = np.vstack((batch_scores_array, batch_scores_numpy))
                batch_labels_array = np.hstack((batch_labels_array, batch_labels_numpy))

        if use_softmax:
            pred_lables = np.argmax(batch_scores_array, axis=1)
            pred_scores = batch_scores_array[:, 1]
            save_date(path,batch_id_array,batch_seq_array,pred_scores,pred_lables)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',default=None, type=str,help='Preprocessed dataset filename.')
    parser.add_argument('-i','--input_file', default=None, metavar='', type=str, help='please make sure your input file is .fa or .fasta format ')
    parser.add_argument('-s','--split',action='store_true',help='If set, the input sequences will be split,')
    parser.add_argument('-w','--window',default=174, help='when split is enabled, the window will be used.', metavar='', type=int)
    parser.add_argument('-g', '--interval',default=10,help='when split is enabled, the Sliding window interval be used .',metavar='',type=int)
    args = parser.parse_args()
    
    input_file = args.input_file
    #seq ,bpp ,bpe path
    base_path = os.getcwd()
    seq_path = os.getcwd() + '/ProcessData/IRES_Seq/'
    bpe_data_path = base_path +'/ProcessData/IRES_BPE_matrix'
    bpp_data_path = base_path +'/ProcessData/IRES_BPP_matrix'
    sava_pkl_path = base_path +'/ProcessData/IRES_Pkl'
    sava_pred_path = base_path + '/ProcessData/IRES_Predict' 

    if args.dataset is None:
        if args.split:
            input_file_path = seq_path + input_file
            output_file_path = seq_path + f'process_{input_file}'
            cmd = f'seqkit sliding -s {args.interval} -W {args.window} -C {input_file_path} > {output_file_path}'
            subprocess.call(cmd, shell=True)
            input_file = f'process_{input_file}'
        #step 1 generate bpp and bpe
        fold_rna_from_file(seq_path + input_file,bpp_data_path,bpe_data_path)
        #step 2 process data to get pkl 
        get_graph(seq_path + input_file,bpe_data_path,bpp_data_path,sava_pkl_path)
    else:
        sava_pkl_path = sava_pkl_path +'/args.dataset'

    #step 3 begin predict ires
    id,device = device_set()
    base_path = os.getcwd()
    net_params = {
        "L": 4,
        "in_dim":4,
        "hidden_dim": 32,
        "out_dim": 32,
        "residual": False,
        "readout": "mean",
        "in_feat_dropout": 0.1,
        "dropout": 0.1,
        "batch_norm": True,
        "self_loop": False,
        "embedding_dim":0,
        "gpu_id": id,
        "type": "GCN",
        "n_classes": 2,
        "use_softmax":True,
        'device': device,
        "fea_info": 4
    }
    print("net_params:",net_params)
    
    #model para path
    model_base_path = os.getcwd() +'/save_model/'
    data_set = RNADataset(sava_pkl_path)
    data_loader = DataLoader(data_set, batch_size= 128, shuffle=False, collate_fn=collate_all)

    over_path = model_base_path + 'over/model_75.pth'
    under1_path = model_base_path + 'under1/model_6.pth'
    under2_path = model_base_path + 'under2/model_195.pth'
    under3_path = model_base_path + 'under3/model_1.pth'


    over_model = load_model("IreeSeek_conv",net_params,over_path,device).to(device)
    under1_model = load_model("IreeSeek_conv",net_params,under1_path,device).to(device)
    under2_model = load_model("IreeSeek_conv",net_params,under2_path,device).to(device)
    under3_model = load_model("IreeSeek_conv",net_params,under3_path,device).to(device)

    models =[over_model,under1_model,under2_model,under3_model]
 
    soft_voting(models,
                device,
                data_loader,
                use_fea=net_params['fea_info'],
                path = sava_pred_path,
                use_softmax =net_params["use_softmax"])