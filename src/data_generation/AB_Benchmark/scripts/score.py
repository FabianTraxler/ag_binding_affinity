import os

import pandas as pd


def get_energy_score(pdb_path: str):
    with open(pdb_path) as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if lines[-i][:20] == "rosetta_energy_score":
            break
    else:
        return None
    score = lines[-i].split(" ")[-1].strip()
    try:
        return float(score)
    except:
        return None


pdb_ids = snakemake.params["pdb_ids"]
relaxed_pdb_path = snakemake.params["relaxed_pdb_path"]
unrelaxed_pdb_path = snakemake.params["unrelaxed_pdb_path"]

summary_df = pd.DataFrame()

for idx, pdb_id in enumerate(pdb_ids):
    pdb_info = {
        "pdb_id": pdb_id.split("_")[0].lower(),
        "file_name": pdb_id + ".pdb"
    }

    bound_score = get_energy_score(os.path.join(relaxed_pdb_path, "{}_b.pdb".format(pdb_id)))
    pdb_info["relaxed_bound_score"] = bound_score
    receptor_score = get_energy_score(os.path.join(relaxed_pdb_path, "{}_r_u.pdb".format(pdb_id)))
    pdb_info["relaxed_receptor_score"] = receptor_score
    ligand_score = get_energy_score(os.path.join(relaxed_pdb_path, "{}_l_u.pdb".format(pdb_id)))
    pdb_info["relaxed_ligand_score"] = ligand_score

    binding_affinity = bound_score - receptor_score - ligand_score

    pdb_info["relaxed_binding_energy"] = binding_affinity


    bound_score = get_energy_score(os.path.join(unrelaxed_pdb_path, "{}_b.pdb".format(pdb_id)))
    pdb_info["unrelaxed_bound_score"] = bound_score
    receptor_score = get_energy_score(os.path.join(unrelaxed_pdb_path, "{}_r_u.pdb".format(pdb_id)))
    pdb_info["unrelaxed_receptor_score"] = receptor_score
    ligand_score = get_energy_score(os.path.join(unrelaxed_pdb_path, "{}_l_u.pdb".format(pdb_id)))
    pdb_info["unrelaxed_ligand_score"] = ligand_score

    binding_affinity = bound_score - receptor_score - ligand_score

    pdb_info["unrelaxed_binding_energy"] = binding_affinity
    summary_df = summary_df.append(pdb_info, ignore_index=True)

summary_df.to_csv(snakemake.output[0], index=False)