import torch
import os
import sys
import copy
import shutil

emb_file = sys.argv[1]
out_file = sys.argv[2]
pdbs_dir = sys.argv[3]

present_pdbs_dir = '/'.join(pdbs_dir.split('/')[:-1] + ['emb_pdbs'])
print(present_pdbs_dir)

if not os.path.exists(present_pdbs_dir):
    os.mkdir(present_pdbs_dir)

dat = torch.load(emb_file, map_location='cpu')

of_emb = {}
files = os.listdir(pdbs_dir)
files = [f[:4] for f in files]
print('files', files)
for d in dat:
    pdb_id = d['pdb_fn'][:4].lower()
    if  pdb_id in files:
        print('pdb id', pdb_id)
        d_new = copy.deepcopy(d) 
        d_new.pop('pdb_fn', None) 
        ch_type = d['input_data']['context_chain_type']
        res_id = d['input_data']['residue_index']
        max_id = torch.max(res_id[torch.logical_or(ch_type == 2, ch_type == 1)]) 
        max_L_id = torch.max(res_id[ch_type==2]) 
        print('ch_type', ch_type)
        print('res_id_before', res_id)
        print('max id', max_id)
        print('max L id', max_L_id)
        res_id[ch_type == 2] += 1
        res_id[ch_type == 1] += -max_L_id - 15 - 2
        res_id[ch_type == 3] += -max_id - 200 - 2
        print('res_id_after', res_id)
        d_new['input_data']['residue_index'] = res_id
        of_emb[pdb_id] = d_new
        shutil.copyfile(os.path.join(pdbs_dir, pdb_id + '.pdb'), os.path.join(present_pdbs_dir, pdb_id + '.pdb'))

torch.save(of_emb, out_file)
