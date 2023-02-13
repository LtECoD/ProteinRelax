#! /bin/bash

prot=2KL8
data_dir="/home/ys/workspace/ProteinAutoEncoder/result"

for epoc in 25 50 75 100 200 300 499
do
    python main.py \
        --pdb ${data_dir}/${prot}-${epoc}.pdb \
        --outdir ${data_dir} \
        --device 1
done