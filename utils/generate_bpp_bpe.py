import os
import gzip
import random
import subprocess
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from functools import partial
from utils.rna_utils import load_seq_bppe
from BPfold.BPfold.util.base_pair_motif import BPM_energ
def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    np.random.seed(random.seed())
    name = str(np.random.rand())
    # Call RNAplfold on command line.
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, w, l, cutoff, name, no_lonely_bps_str)
    ret = subprocess.call(cmd, shell=True)

    # assemble adjacency matrix
    row_col, link, prob = [], [], []
    length = len(seq)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            link.append(1)
            prob.append(1.)
        if i != 0:
            row_col.append((i, i - 1))
            link.append(2)
            prob.append(1.)

    # Extract base pair information.
    name += '_0001_dp.ps'
    start_flag = False
    with open(name) as f:
        for line in f:
            if start_flag:
                values = line.split()
                if len(values) == 4:
                    source_id = int(values[0]) - 1
                    dest_id = int(values[1]) - 1
                    avg_prob = float(values[2])
                    # source_id < dest_id
                    row_col.append((source_id, dest_id))
                    link.append(3)
                    prob.append(avg_prob**2)
                    row_col.append((dest_id, source_id))
                    link.append(4)
                    prob.append(avg_prob**2)
            if 'start of base pair probability data' in line:
                start_flag = True
    os.remove(name)

    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)))
########################################################
#
#
#   Generate bpp and bpe
#
#
#########################################################
def fold_rna_from_file(seq_path,sav_bpp_path,sava_bpe_path):
    all_id,all_seq = load_seq_bppe(seq_path)
    winsize =  150
    print('running rnaplfold with winsize %d' % (winsize))
    print("all_id:",len(all_id),"all_seq:",len(all_seq))
    BPM = BPM_energy()
    for idx, seq in tqdm(enumerate(all_seq), total=len(all_seq), desc="Processing sequences"):
        #get bpp
        link_csr_matrix , bpp_csr_matrix = fold_seq_rnaplfold(seq,w=winsize, l=min(winsize, 150), cutoff=1e-4, no_lonely_bps=True)
        np.save(sav_bpp_path + "/" + all_id[idx] + ".npy",bpp_csr_matrix.toarray())
        #get bpe
        mat2 = BPM.get_energy(seq, normalize_energy=False, dispart_outer_inner=False)
        np.save(sava_bpe_path + "/" + all_id[idx] + ".npy",mat2)
if __name__ == '__main__':
    basedir = os.getcwd()
    #read_negative_file =basedir +"/all_test/IRES_newtest_seq/55k_newtest_negative.fa"
    #sava_negative =basedir + "/all_test/IRES_newtest_BPP/negative"
    #fold_rna_from_file(read_negative_file,sava_negative)
    #read_pos_file =basedir +"/all_test/IRES_newtest_seq/55k_newtest_positive.fa"
    #sava_pos =basedir + "/all_test/IRES_newtest_BPP/positive"
    #fold_rna_from_file(read_pos_file,sava_pos)