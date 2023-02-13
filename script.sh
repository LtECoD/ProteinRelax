#! /bin/bash

prot=AF-A7N9H7

for epoc in 25 50 75 100 200 300 499
do
    python main.py \
        --pdb "/lustre/home/yangsen/workspace/ProteinAutoEncoder/result/${prot}-${epoc}.pdb" \
        --outdir "/lustre/home/yangsen/workspace/ProteinAutoEncoder/result"
done