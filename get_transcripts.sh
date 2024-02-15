#!/bin/bash

# This script is used to create the annotation file for the human genome
# The annotation python scripts were downloaded from GLORI GitHub repository

# Download the human genome annotation file
mkdir -p data/annot_ref/
wget  https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/annotation_releases/GCF_000001405.40-RS_2023_03/GCF_000001405.40_GRCh38.p14_assembly_report.txt 

wget https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/annotation_releases/GCF_000001405.40-RS_2023_03/GCF_000001405.40_GRCh38.p14_genomic.gtf.gz 

conda activate thesis

# Patch the GTF file based on the assembly report
python GLORI/get_anno/change_UCSCgtf.py -i data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf -j data/annot_ref/GCF_000001405.40_GRCh38.p14_assembly_report.txt -o data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens

# Get exon start and end
python GLORI/get_anno/gtf2anno.py -i data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens -o data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens.tbl

# Filtering the unknown transcript
awk '$3!~/_/&&$3!="na"' data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens.tbl | sed '/unknown_transcript/d'  > data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens.tbl2

# Get the longest transcript for each gene
python GLORI/get_anno/selected_longest_transcrpts_fa.py -anno data/annot_ref/GCF_000001405.40_GRCh38.p14_genomic.gtf_change2Ens.tbl2  --outname_prx data/annot_ref/GCF_000001405.39_GRCh38.p13_rna2