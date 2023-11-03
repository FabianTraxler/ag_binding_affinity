""" Converts residue IDs as found in the OF embeddings file to correspond to
    unique residue IDs per chain ID as found in PDB files, and saves them as
    separate torch files for each data point

Args:
    emb_file: File containing the OF embeddings 
    out_dir: Directory where the converted files containing the embeddings are saved
    pdbs_dir: Directory containing the PDB files from the original ABDB dataset 
"""

import torch
import os
import sys
import copy
import shutil
import pandas as pd
import ast
import numpy as np

emb_file = sys.argv[1]  # file containing OF embeddings as obtained from guided diffusion
summ_file = sys.argv[2]  # location of the new summary file, needs to be linked in config as well
pdbs_dir = sys.argv[3]  # directory containing the PDB files from the original ABDB dataset
out_dir = sys.argv[4]  # directory where the converted files containing the embeddings are saved

chain_id_order = ['h', 'l', 'a', 'b', 'c']

present_pdbs_dir = '/'.join(pdbs_dir.split('/')[:-1] + ['emb_pdbs'])
print(present_pdbs_dir)

if not os.path.exists(present_pdbs_dir):
    os.mkdir(present_pdbs_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

dat = torch.load(emb_file, map_location='cpu')

summ_df = pd.read_csv(summ_file)

files = os.listdir(pdbs_dir)
files = [f[:4] for f in files]
present_data = []
print('files', files)
for d in dat:
    pdb_id = d['pdb_fn']
    if  pdb_id[:4].lower() in files:
        if '_' in pdb_id:
            if pdb_id[-1:] != '1':
                continue
        pdb_id = pdb_id[:4].lower()
        df_idx = np.where(summ_df.pdb == pdb_id)[0][0]
        pdb_row = summ_df.iloc[df_idx]
        chain_dict = ast.literal_eval(pdb_row.chain_infos)
        chain_keys = list(chain_dict.keys())
        if len(chain_keys) < 3:
            continue
        new_chain_dict = {chain_id_order[i]: chain_dict[chain_keys[i]] 
                          for i in range(len(chain_keys))}
        pdb_row.chain_infos = new_chain_dict
        d_new = copy.deepcopy(d) 
        ch_type = d['input_data']['context_chain_type']
        res_id = d['input_data']['residue_index']
        max_id = torch.max(res_id[torch.logical_or(ch_type == 2, ch_type == 1)]) 
        max_L_id = torch.max(res_id[ch_type==2]) 
        res_id[ch_type == 2] += 1
        res_id[ch_type == 1] += -max_L_id - 15
        res_id[ch_type == 3] += -max_id - 200
        d_new['input_data']['residue_index'] = res_id
        assert d_new['input_data']['residue_index'] == d_new['input_data']['residue_index_pdb']
        torch.save(d_new, os.path.join(out_dir, pdb_id + '.pt'))
        shutil.copyfile(os.path.join(pdbs_dir, pdb_id + '.pdb'), os.path.join(present_pdbs_dir, pdb_id + '.pdb'))
        present_data.append(pdb_row)

fname_out = summ_file[:-4] + '_OF.csv'
new_summ_df = pd.DataFrame(present_data)
print(new_summ_df)
new_summ_df.set_index('Unnamed: 0', inplace=True)
print(new_summ_df)
new_summ_df.to_csv(fname_out)
