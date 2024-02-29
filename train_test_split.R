#myGR is a GRanges object - contains 
#strand, single nucleotide positions of width 1
#id_num: a numeric id (1:num_positions) 
#gene: the containing gene for each position
# #label: training labels (in my case a continuous proportion, called PD_score in the below)
library(Biostrings)
library(tidyverse)
library(readxl)
library(GenomicFeatures)
library(BSgenome)
library(fuzzyjoin)
library(rsample) # For kvold split
library(BSgenome.Hsapiens.NCBI.GRCh38)
library(TxDb.Hsapiens.UCSC.hg38.knownGene) #v44 Gencode
library(BSgenome.Mmusculus.UCSC.mm39) # For MEF 

# Subject to change
DATA_FOLDER <- '/Users/faqih/Documents/UCPH/Thesis/code/data/'

# ===== Heat Shock MEF Data ===== #
heat_shock_MEF_up <- readxl::read_excel(paste(DATA_FOLDER, 
                                              'GLORI_data/heat_shock_m6A_sites_MEF.xlsx', 
                                              sep=''),
                                        sheet = 'HS UP m6A')
heat_shock_MEF_down <- readxl::read_excel(paste(DATA_FOLDER, 
                                                'GLORI_data/heat_shock_m6A_sites_MEF.xlsx', 
                                                sep=''),
                                          sheet = 'HS Down m6A')
heat_shock_MEF_unchange <- readxl::read_excel(paste(DATA_FOLDER, 
                                                    'GLORI_data/heat_shock_m6A_sites_MEF.xlsx', 
                                                    sep=''),
                                              sheet = 'HS un-change m6A')

heat_shock_MEF <- rbind(heat_shock_MEF_up, heat_shock_MEF_down, heat_shock_MEF_unchange)
# Int conversion
heat_shock_MEF$start <- as.integer(heat_shock_MEF$start)
heat_shock_MEF$end <- as.integer(heat_shock_MEF$end)
heat_shock_MEF$log10pvalue <- as.numeric(heat_shock_MEF$log10pvalue)

heat_shock_MEF_GR <- makeGRangesFromDataFrame(heat_shock_MEF |> dplyr::select(seqnames, start, end))



# =====Hypoxia HeLa Data ===== #
hypoxia_HeLa_up <- readxl::read_excel(paste(DATA_FOLDER, 
                                            'GLORI_data/hypoxia_m6A_sites_HeLa.xlsx', 
                                            sep=''),
                                      sheet = 'Hypoxia UP m6A')
hypoxia_HeLa_down <- readxl::read_excel(paste(DATA_FOLDER, 
                                              'GLORI_data/hypoxia_m6A_sites_HeLa.xlsx', 
                                              sep=''),
                                        sheet = 'Hypoxia Down m6A')
hypoxia_HeLa_unchange <- readxl::read_excel(paste(DATA_FOLDER, 
                                                  'GLORI_data/hypoxia_m6A_sites_HeLa.xlsx', 
                                                  sep=''),
                                            sheet = 'Hypoxia un-change m6A')

hypoxia_HeLa <- rbind(hypoxia_HeLa_up, hypoxia_HeLa_down, hypoxia_HeLa_unchange)
# Int conversion
hypoxia_HeLa$start <- as.integer(hypoxia_HeLa$start)
hypoxia_HeLa$end <- as.integer(hypoxia_HeLa$end)
hypoxia_HeLa$log10pvalue <- as.numeric(hypoxia_HeLa$log10pvalue)
# Renaming direction since there is two values: Hypoxia and hypoxia 
hypoxia_HeLa$direction <- str_to_sentence(hypoxia_HeLa$direction)




# ===== TxD Related ===== #
# Get annotation to the txDb (including the validity checking)
human_txDb <- makeTxDbFromGFF(paste0(DATA_FOLDER, 'annot_ref/gencode.v45.annotation.gtf'))
# Get from rtracklayer to extract metadata
gtf <- rtracklayer::import.gff(con=paste0(DATA_FOLDER, 'annot_ref/gencode.v45.annotation.gtf'), 
                        format="gtf", genome="GRCh38.p14")

metadata <- as.tibble(elementMetadata(gtf)[ , c("gene_id", "gene_name", 
                                                "gene_type", "transcript_id", 
                                                "transcript_name")])
metadata <- na.omit(metadata)
metadata <- metadata[!duplicated(metadata), ]


# Promoters
human_Tx_promoters <-  promoters(human_txDb, upstream=1000, downstream=0)

# Tx
human_Tx <- transcripts(human_txDb)
# Transcript length summary
human_Tx_length <- transcriptLengths(human_txDb,
                                             with.cds_len=TRUE, with.utr5_len=TRUE, with.utr3_len=TRUE)


human_Tx_data <- S4Vectors::merge(human_Tx, metadata, 
                                          by.x = 'tx_name', by.y = 'transcript_id', all.x=TRUE)
human_Tx_data <- S4Vectors::merge(human_Tx_data, human_Tx_length |> 
                                    dplyr::select(tx_name, nexon, tx_len,cds_len, utr5_len, utr3_len), 
                                  by.x = 'tx_name', by.y = 'tx_name', all.x=TRUE)

# Filtering the longest transcript (width) only for each gene
# it still contain numbers of non mRNA gene: i.e. rRNA, tRNA, Mt_RNA, etc.
human_Tx_data_longest <- human_Tx_data |> 
  dplyr::group_by(gene_id) |> 
  dplyr::filter(width == max(width)) |> 
  # Since several transcripts could have the same start and end.
  # need to filter by tx_len (CDS + UTRs)
  dplyr::filter(tx_len == max(tx_len)) |> 
  dplyr::ungroup() |> 
  dplyr::select(!c(tx_id)) |> 
  dplyr::rename('tx_name'='transcript_name', 'tx_id'='tx_name') |> 
  dplyr::relocate(c('tx_id','tx_name','nexon','tx_len'), 
                  .after = c('strand'))
human_Tx_data_longest$seqnames <- as.character(human_Tx_data_longest$seqnames,
                                               upstream=1000, downstream=0)

# Extract promoters for the longest transcripts
human_Tx_longest_promoters <- S4Vectors::merge(human_Tx_promoters, 
                                               (human_Tx_data_longest |> 
                                                 dplyr::select(tx_id, tx_name, gene_name, gene_id)), 
                                               by.x = 'tx_name', by.y = 'tx_id', 
                                               all.x=TRUE)
human_Tx_longest_promoters <- as.tibble(human_Tx_longest_promoters |> 
                                          dplyr::select(!c('tx_id'))
                                        ) |> 
  dplyr::rowwise() |>
  # UCSC Format for BSgenome
  dplyr::mutate( seqnames=ifelse(strsplit(as.character(seqnames), "chr")[[1]][2] == "M", "MT",
                                   strsplit(as.character(seqnames), "chr")[[1]][2])) |> 
  dplyr::ungroup() |> 
  # Remove unknown filter
  filter(!is.na(tx_name.y)) |> 
  rename(tx_id=tx_name, tx_name=tx_name.y) |> 
  mutate(tx_name.y=NULL, tx_name=NULL) |>
  arrange(seqnames, start,end)
  



# ===== Joining Hypoxia with Annot ===== #
annotated_hypoxia_Hela <- fuzzyjoin::genome_join(
  hypoxia_HeLa, 
  human_Tx_data_longest |> 
    dplyr::select(seqnames, start, end, gene_id, gene_name, tx_id, tx_name),
  by = c("seqnames", "start", "end"), 
  mode = "left"
)
annotated_hypoxia_Hela <- annotated_hypoxia_Hela |> 
  dplyr::rowwise() |>
  # UCSC Format for BSgenome
  dplyr::mutate( seqnames.x=ifelse(strsplit(as.character(seqnames.x), "chr")[[1]][2] == "M", "MT",
                            strsplit(as.character(seqnames.x), "chr")[[1]][2])) |> 
  dplyr::ungroup() |> 
  dplyr::rename(seqnames=seqnames.x, start=start.x, end=end.x)|> 
  # HERE WE ONLY GET THE TX THA MAPPED INTO GENE 
  dplyr::filter(!is.na(gene_name)) |> 
  # Sort 
  dplyr::arrange(seqnames,start,end, gene_id, tx_id) |> 
  dplyr::mutate(seqnames.y=NULL, start.y=NULL, end.y=NULL) 

# == m6A sites GR ==
annotated_hypoxia_Hela_GR <- makeGRangesFromDataFrame(annotated_hypoxia_Hela)
# Adding metadata to GR
values(annotated_hypoxia_Hela_GR) <- annotated_hypoxia_Hela |> 
  dplyr::select(gene_id, gene_name, tx_id, tx_name, meth_control, meth_case, label)


# == Promoters GR ==
annotated_promoters_Hela <- annotated_hypoxia_Hela |> 
  dplyr::rename(site_start=start,site_end=end) |> 
  dplyr::left_join(human_Tx_longest_promoters |> 
              dplyr::select(tx_id, seqnames,start,end), 
            dplyr::join_by(tx_id)) |> 
  dplyr::mutate(seqnames.y=NULL) |> 
  dplyr::rename(seqnames=seqnames.x) |> 
  dplyr::select(seqnames,start,end,tx_id,tx_name,gene_id,gene_name,site_start,site_end)

annotated_promoters_Hela_GR <- makeGRangesFromDataFrame(annotated_promoters_Hela)
values(annotated_promoters_Hela_GR) <- annotated_promoters_Hela |> 
  dplyr::select(tx_id, tx_name, gene_id, gene_name,site_start,site_end)




# ===== Getting Sequence from BSgenome ===== #
# Credit to Sarah Rennie
bs_human <- BSgenome.Hsapiens.NCBI.GRCh38
bs_mouse <- BSgenome.Mmusculus.UCSC.mm39
bs_info_human <- as.data.frame(seqinfo(BSgenome.Hsapiens.NCBI.GRCh38))
bs_info_mouse <- as.data.frame(seqinfo(BSgenome.Mmusculus.UCSC.mm39))


bs_info_human <- tibble::rownames_to_column(bs_info_human, "seqnames")
#bs_info_mouse <- tibble::rownames_to_column(bs_info_mouse, "seqnames")

# +/- 2 from methylation sites -> DRACH specificity
hypoxia_HeLa_seq <- BSgenome::getSeq(bs_human, annotated_hypoxia_Hela_GR+2)
mcols(annotated_hypoxia_Hela_GR)$seq <- hypoxia_HeLa_seq
hypoxia_HeLa_promoters_seq <- BSgenome::getSeq(bs_human,annotated_promoters_Hela_GR)
mcols(annotated_promoters_Hela_GR)$promoter_seq <- hypoxia_HeLa_promoters_seq


#sanity check the center point is the expected base
seqmat <- as.matrix(hypoxia_HeLa_seq)
print(table(seqmat))
seqmat <- as.matrix(hypoxia_HeLa_promoters_seq)
print(table(seqmat)) # 'N' value will be taken care on one-hot encoding

GR <-annotated_hypoxia_Hela_GR 
mcols(GR)$promoter_seq <- annotated_promoters_Hela_GR$promoter_seq




# ===== Creating train-test split ===== #
temp <- tibble(id=c(1:length(GR)),gene_id = GR$gene_id, gene_name=GR$gene_name, 
               tx_id=GR$tx_id, meth_control=GR$meth_control, meth_case=GR$meth_case) 

set.seed(123)
#allocate a group randomly to each gene_id, not position
gene_group <- sample(1:5,length(temp$gene_id),replace=T)
names(gene_group) <- temp$gene_id
temp$gene_group <- gene_group[as.vector(temp$gene_id)]

#create the five folds stratified by gene_group
#all of the groups consiste of roughly equal number of gene
split_tibble <- temp |> rsample::group_vfold_cv(group = gene_group)

for(i in 1:nrow(split_tibble))
{
  # Write out the training/testing labels
  label_split_TRAIN <- rsample::training(split_tibble$splits[[i]]) 
  label_split_TEST <- rsample::testing(split_tibble$splits[[i]]) 

  jsonlite::write_json(label_split_TRAIN, 
                       paste0(DATA_FOLDER,
                              "train_test_data/train_label_SPLIT_",i,".json"))
  jsonlite::write_json(label_split_TEST, 
                       paste0(DATA_FOLDER, 
                              "train_test_data/test_label_SPLIT_",i,".json"))

  motif_split_TRAIN <- GR$seq[training(split_tibble$splits[[i]]) |> pull(id)]
  motif_split_TEST <- GR$seq[testing(split_tibble$splits[[i]]) |> pull(id)]
  
  promoter_split_TRAIN <- GR$promoter_seq[training(split_tibble$splits[[i]]) |> pull(id)]
  promoter_split_TEST <- GR$promoter_seq[testing(split_tibble$splits[[i]]) |> pull(id)]

  ## Write an XStringSet object to a FASTA (or FASTQ) file:
  writeXStringSet(motif_split_TRAIN, paste0(DATA_FOLDER,"train_test_data/motif_fasta_train_SPLIT_",i,".fasta"), append=FALSE,
                  compress=FALSE, compression_level=NA, format="fasta")

  writeXStringSet(motif_split_TEST, paste0(DATA_FOLDER,"train_test_data/motif_fasta_test_SPLIT_",i,".fasta"), append=FALSE,
                  compress=FALSE, compression_level=NA, format="fasta")

    writeXStringSet(promoter_split_TRAIN, paste0(DATA_FOLDER,"promoter_fasta_train_SPLIT_",i,".fasta"), append=FALSE,
                  compress=FALSE, compression_level=NA, format="fasta")

  writeXStringSet(promoter_split_TEST, paste0(DATA_FOLDER,"train_test_data/promoter_fasta_test_SPLIT_",i,".fasta"), append=FALSE,
                  compress=FALSE, compression_level=NA, format="fasta")

}

save(temp,file="data_tibble_PD.Rdat")
save(split_tibble,file="data_tibble_splits_PD.Rdat")


## NEXT 
# Converting into: 
# Tensor([batch_num, 4, seq_length])
# tensor([[[0., 1., 1., 0., 0.],
#          [0., 0., 0., 1., 1.],      #GAACC
#          [1., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]],
#         
#         [[0., 0., 1., 0., 1.],
#          [0., 0., 0., 1., 0.],      #GGACA
#          [1., 1., 0., 0., 0.],
#          [0., 0., 0., 0., 0.]],
#         
#         ...,