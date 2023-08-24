import yaml, subprocess, tempfile

with open(snakemake.input[0]) as f:
    if snakemake.wildcards.publication.startswith('mason21_'):
        publication = yaml.safe_load(f.read())['mason21_optim_therap_antib_by_predic']
    else:
        publication = yaml.safe_load(f.read())[snakemake.wildcards.publication]
target_complex = snakemake.wildcards.complex.split(':')[0]
for complex in publication['complexes']:
    if complex['antibody']['name'] == target_complex.split('_')[0] and complex['antigen']['name'] == target_complex.split('_')[1]:
        break
else:
    raise ValueError(f'{target_complex.split("_")} does not exist in metadata for publication {snakemake.wildcards.publication}')

if "template" in complex:  # This is to please phillips21, which was generated slightly differently from the others
    pdb_id = complex["template"]['id'].lower()
    chains = ','.join(complex["template"]['chains']['antigen'] + complex["template"]['chains']['antibody'])
else:
    pdb_id = complex["template"]['pdb'].lower()
    chains = ','.join(list(complex["pdb"]['chains']['antigen'].values()) + list(complex["pdb"]['chains']['antibody'].values()))

# TODO use pdb_fetch {pdb_id} --output {pdb_id}.unprep --add-atoms=none --replace-nonstandard <- if they become an issue
with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp_pdb, tempfile.NamedTemporaryFile(suffix=".error") as tmp_error:
    subprocess.run(f'''mkdir {snakemake.output["template_dir"]} && pdb_fetch {pdb_id} > {tmp_pdb.name} 2> {tmp_error.name}''', shell=True)
    if subprocess.run(f'grep Error {tmp_error.name}', shell=True).returncode == 0:
        subprocess.run(f'wget https://files.rcsb.org/pub/pdb/compatible/pdb_bundle/cd/{pdb_id}/{pdb_id}-pdb-bundle.tar.gz && tar xzf {pdb_id}-pdb-bundle.tar.gz && mv {pdb_id}-pdb-bundle1.pdb {tmp_pdb.name}', shell=True)

    subprocess.run(f'''grep '^ATOM' {tmp_pdb.name} | \
        pdb_sort | \
        pdb_tidy | \
        pdb_selchain -{chains} | \
        pdb_fixinsert | \
        pdb_delhetatm | \
        pdb_seg  > {snakemake.output["template_dir"]}/{pdb_id}.pdb''', shell=True, check=True)
    # chainbows destroys chain identifiers (because of the TERs inserted by pdb_tidy OMG)
