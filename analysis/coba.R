
# Load all of the packages that you end up using
# in your analysis in this code chunk.
library(tidyverse)
library(readxl)
library(biomaRt)
library(fuzzyjoin)
library(ggpubr) 
library(ggrepel)
library(ggplot2)
library(ggvenn)
library(latex2exp)
# GFF/GTF related libs
library(rtracklayer) # Interacting with UCSC genome browser
library(GenomicRanges)
library(plyranges)
library(GenomicFeatures)
library(ensembldb)
library(Guitar)
# For dealing with bamfiles
library(Rsamtools)
library(universalmotif)
library(Biostrings)
dna_filter_1 <- readDNAStringSet("/binf-isilon/renniegrp/vpx267/ucph_thesis/analysis/sequence_filter_0.fasta")



nt_freqs <- c("A"=1/4,"C"=1/4,"G"=1/4,"T"=1/4)
motif1 <- universalmotif::create_motif(dna_filter_1, 
                            type="PCM", 
                            name= paste0("filter_",1), 
                            pseudocount = 1, 
                            bkg = nt_freqs[c("A","C","G","T")])
view(motif1)