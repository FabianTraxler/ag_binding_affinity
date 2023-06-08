import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('rf_emb_location', type=str)
parser.add_argument('rf_emb_save_dir', type=str)

args = parser.parse_args()

dat_rf = np.load(args.rf_emb_location, allow_pickle=True).item()
mapping = {'state_prev': 'single',
           'input_data': {'true_x0': 'positions',
                          'aatype': 'aatype',
                          'context_chain_type': 'context_chain_type'}}

if not os.path.exists(args.rf_emb_save_dir):
    os.mkdir(args.rf_emb_save_dir)

chain_type_order = [2, 1, 3]

for key in dat_rf.keys():
    data = dat_rf[key]
    data_conv = {}
    for d_key in data.keys():
        if d_key in mapping.keys():
            if d_key == 'input_data':
                data_conv['input_data'] = {}
                input_data = data['input_data']
                for i_key in input_data.keys():
                    if i_key in mapping['input_data'].keys():
                        data_conv['input_data'][mapping['input_data'][i_key]] = \
                            torch.tensor(input_data[i_key]).unsqueeze(0)
                    if i_key == 'ref':
                        data_conv['input_data']['chain_id_pdb'] = []
                        data_conv['input_data']['residue_index_pdb'] = []
                        for res in input_data['ref']:
                            data_conv['input_data']['chain_id_pdb'].append(ord(res[0]))
                            data_conv['input_data']['residue_index_pdb'].append(res[1])
                        data_conv['input_data']['chain_id_pdb'] = \
                            torch.tensor(data_conv['input_data']['chain_id_pdb']).unsqueeze(0)
                        data_conv['input_data']['residue_index_pdb'] = \
                            torch.tensor(data_conv['input_data']['residue_index_pdb']).unsqueeze(0)
                chain_type_idx = [torch.where(data_conv['input_data']['context_chain_type'][0, :] == i)[0]
                                  for i in chain_type_order]
                chain_type_idx = torch.cat(chain_type_idx)
                print(data_conv['input_data']['context_chain_type'])
                print(data_conv['input_data']['chain_id_pdb'])
                print(data_conv['input_data']['residue_index_pdb'])
                print(data_conv['input_data']['aatype'])
                data_conv['input_data']['context_chain_type'] = \
                    data_conv['input_data']['context_chain_type'][:, chain_type_idx]
                data_conv['input_data']['chain_id_pdb'] = \
                    data_conv['input_data']['chain_id_pdb'][:, chain_type_idx]
                data_conv['input_data']['residue_index_pdb'] = \
                    data_conv['input_data']['residue_index_pdb'][:, chain_type_idx]
                data_conv['input_data']['aatype'] = \
                    data_conv['input_data']['aatype'][:, chain_type_idx]
                print(data_conv['input_data']['context_chain_type'])
                print(data_conv['input_data']['chain_id_pdb'])
                print(data_conv['input_data']['residue_index_pdb'])
                print(data_conv['input_data']['aatype'])
            else:
                data_conv[mapping[d_key]] = torch.tensor(data[d_key])

    save_file = os.path.join(args.rf_emb_save_dir, key[:4].lower() + '.pt')
    torch.save(data_conv, save_file)
