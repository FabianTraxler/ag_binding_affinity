

for i in $(seq 100); do SNAKEMAKE_PROFILE= snakemake -j200 --use-conda -k --batch all=$i/100; done
