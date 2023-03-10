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

emb_file = sys.argv[1]
out_dir = sys.argv[2]
pdbs_dir = sys.argv[3]

present_pdbs_dir = '/'.join(pdbs_dir.split('/')[:-1] + ['emb_pdbs'])
print(present_pdbs_dir)

if not os.path.exists(present_pdbs_dir):
    os.mkdir(present_pdbs_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

dat = torch.load(emb_file, map_location='cpu')

files = os.listdir(pdbs_dir)
files = [f[:4] for f in files]
print('files', files)
for d in dat:
    pdb_id = d['pdb_fn'][:4].lower()
    if  pdb_id in files:
        print('pdb id', pdb_id)
        d_new = copy.deepcopy(d) 
        ch_type = d['input_data']['context_chain_type']
        res_id = d['input_data']['residue_index']
        max_id = torch.max(res_id[torch.logical_or(ch_type == 2, ch_type == 1)]) 
        max_L_id = torch.max(res_id[ch_type==2]) 
        print('ch_type', ch_type)
        print('res_id_before', res_id)
        print('max id', max_id)
        print('max L id', max_L_id)
        res_id[ch_type == 2] += 1
        res_id[ch_type == 1] += -max_L_id - 15
        res_id[ch_type == 3] += -max_id - 200
        print('res_id_after', res_id)
        d_new['input_data']['residue_index'] = res_id
        torch.save(d_new, os.path.join(out_dir, pdb_id + '.pt'))
        shutil.copyfile(os.path.join(pdbs_dir, pdb_id + '.pdb'), os.path.join(present_pdbs_dir, pdb_id + '.pdb'))
