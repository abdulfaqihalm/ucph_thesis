library(Biostrings)
library(tidyverse)
library(readxl)
library(GenomicFeatures)
library(BSgenome)
library(fuzzyjoin)
library(rsample) # For kvold split
library(BSgenome.Hsapiens.NCBI.GRCh38)
library(TxDb.Hsapiens.UCSC.hg38.knownGene) #v44 Gencode
# library(BSgenome.Mmusculus.UCSC.mm39) # For MEF 

# Subject to change
DATA_FOLDER <- '/Users/faqih/Documents/UCPH/Thesis/code/data/'
data_path <- '/Users/faqih/Documents/UCPH/Thesis/code/data/'

# A rate serving as the m6A level
# Has transcript and gene info
GRE_folder = '/binf-isilon/renniegrp/vpx267/ucph_thesis/data/GLORI_data'
hela_1 <- read.csv(paste0(GRE_folder,'GSM6432595_Hela-1_35bp_m2.totalm6A.FDR.csv'), sep = "\t")
hela_2 <- read.csv(paste0(GRE_folder,'GSM6432596_Hela-2_35bp_m2.totalm6A.FDR.csv'), sep = "\t")
mef_1 <- read.csv(paste0(GRE_folder,'GSM6432601_MEF-1_35bp_m2.totalm6A.FDR.csv'), sep = "\t")
mef_2 <- read.csv(paste0(GRE_folder,'GSM6432602_MEF-2_35bp_m2.totalm6A.FDR.csv'), sep = "\t")
hypox_hela_1 <- read.csv(paste0(GRE_folder,'GSM6432598_hypoxia-1_35bp_m2.totalm6A.FDR.csv'), sep = '\t')
hypox_hela_2 <- read.csv(paste0(GRE_folder,'GSM6432599_hypoxia-2_35bp_m2.totalm6A.FDR.csv'), sep = '\t')
hs_mef_1 <- read.csv(paste0(GRE_folder,'GSM6432604_HS-1_35bp_m2.totalm6A.FDR.csv.gz'), sep = '\t')
hs_mef_2 <- read.csv(paste0(GRE_folder,'GSM6432605_HS-2_35bp_m2.totalm6A.FDR.csv.gz'), sep = '\t')

hela_raw_intrsct <- hela_1 |> 
  dplyr::inner_join(hela_2, by=c("Chr", "Sites")) |> 
  dplyr::rename(Strand=Strand.x, Gene=Gene.x) |> 
  dplyr::select(Chr, Sites, Strand, Gene, Ratio.x, Ratio.y) |> 
  rowwise() |> 
  mutate(meth_ctrl_mean = mean(c_across(c('Ratio.x', 'Ratio.y')))) |> 
  ungroup() |> 
  dplyr::select(!c(Ratio.x, Ratio.y))
hela_raw_hypoxia_intrsct <-  hypox_hela_1 |> 
  dplyr::inner_join(hypox_hela_2, by=c("Chr", "Sites")) |> 
  dplyr::rename(Strand=Strand.x, Gene=Gene.x) |> 
  dplyr::select(Chr, Sites, Strand, Gene, Ratio.x, Ratio.y) |> 
  rowwise() |> 
  mutate(meth_case_mean = mean(c_across(c('Ratio.x', 'Ratio.y')))) |> 
  ungroup() |> 
  dplyr::select(!c(Ratio.x, Ratio.y))

hela_hypoxia_intrsct <- hela_raw_intrsct |> 
  dplyr::full_join(hela_raw_hypoxia_intrsct, by=c("Chr", "Sites")) |> 
  dplyr::mutate(strand = ifelse(is.na(Strand.x),Strand.y,Strand.x),
                meth_control = ifelse(is.na(meth_ctrl_mean), 0, meth_ctrl_mean),
                meth_case = ifelse(is.na(meth_case_mean), 0, meth_case_mean),
                end = Sites,
                gene= ifelse(is.na(Gene.x), Gene.y, Gene.x)) |> 
  dplyr::rename(seqnames=Chr, start=Sites) |> 
  dplyr::select(c(seqnames, start, end, strand, gene, meth_case, meth_control))

#Sanity check, should be zero --> no duplicate sites, thus safe for training-test split
hela_raw_hypoxia_intrsct |> dplyr::group_by(Chr, Sites) |> dplyr::summarise(count=dplyr::n()) |> dplyr::filter(count>1)




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
  # STILL CONTAIN DUPLICATE. CHECK: human_Tx_data_longest |>  dplyr::inner_join(human_Tx_data_longest |> dplyr::select(gene_name, gene_type) |> group_by(gene_name) |> dplyr::summarise(count = dplyr::n()) |> dplyr::filter(count>1), by=c("gene_name")) |> arrange(gene_name,tx_id)
  # Thus add filter below
  dplyr::group_by(gene_name) |>
  dplyr::filter(cds_len == max(cds_len)) |> 
  dplyr::ungroup() |> 
  # Arbitary filter by tx_name since they already have the save width, cds len
  dplyr::group_by(gene_name) |>
  dplyr::filter(tx_name == max(tx_name)) |> 
  dplyr::ungroup() |> 
  dplyr::group_by(gene_name) |>
  dplyr::filter(gene_id == max(gene_id)) |> 
  dplyr::ungroup() |> 
  dplyr::select(!c(tx_id)) |> 
  dplyr::rename('tx_name'='transcript_name', 'tx_id'='tx_name') |> 
  dplyr::relocate(c('tx_id','tx_name','nexon','tx_len'), 
                  .after = c('strand'))
# FILTERING PSEUDOGENE, MISC, VAULT, TO BE CONFIRMED (TEC), ANTIBODY GENES 
human_Tx_data_longest |> dplyr::filter(!grepl('pseudogene|misc_RNA|vault|TR_|IG|TEC|artifact', gene_type)) |> dplyr::distinct(gene_type)

### Sanity chec. Should be zero! 
human_Tx_data_longest |>  dplyr::inner_join(human_Tx_data_longest |> dplyr::select(gene_name, gene_type) |> group_by(gene_name) |> dplyr::summarise(count = dplyr::n()) |> dplyr::filter(count>1), by=c("gene_name")) |> arrange(gene_name,tx_id)
# NOTE: "ELSE" is not known gene from GLORI pipeline



# ===== Joining Hypoxia with Annot ===== #
#Here we want to know the longest transcript which annotated from GLORI
#Expected several NA gene_type from known gene i.e. LOC112268063
annotated_hypoxia_Hela <- hela_hypoxia_intrsct |> 
  dplyr::left_join(human_Tx_data_longest |> 
                     dplyr::select(gene_name, gene_type, tx_id, width, start, end, strand),
                                                                   by=dplyr::join_by(gene==gene_name)) |>
  dplyr::rename(start = start.x, end = end.x, strand = strand.x, 
                start_gene = start.y, end_gene = end.y, strand_gene = strand.y)
  

# We want to still annotate several sites which might not joined with the longest transcript data 
# to get the promoters.
# Note: the non-joined sites are those which have gene_name == "ELSE" and other genes which somehow couldn't be found 
# from the TxDb. There are ~33,238 stes: annotated_hypoxia_Hela |> filter(is.na(start_gene))
# Other named gene from GLORI migh not be joined due to change in the name. Hence, we will join it by location and take
# the longest transcript for each gen
annotated_hypoxia_Hela_temp_na <- (annotated_hypoxia_Hela |> dplyr::filter(is.na(start_gene)) |> 
     dplyr::select(!c(gene_type, tx_id, width, start_gene, end_gene, strand_gene)) |> dplyr::mutate(site=start)) |> 
  dplyr::left_join(human_Tx_data_longest |> 
                     #dplyr::filter(!grepl('vault|TEC|artifact', gene_type)) |> 
                     dplyr::select(seqnames, start, end, strand, gene_name, gene_type, tx_id,tx_len,width),
                   by=dplyr::join_by(seqnames==seqnames, site>=start, site<=end)) |> 
  dplyr::rename(start = start.x, end = end.x, strand = strand.x, 
                start_gene = start.y, end_gene = end.y, strand_gene = strand.y) |> 
  group_by(seqnames,start,end) |> 
  dplyr::mutate(tx_len=ifelse(is.na(tx_len), 0, tx_len)) |> 
  dplyr::filter(tx_len == max(tx_len)) |> 
  dplyr::ungroup() 


# There are around 7,086  sites which still can't be joined
# where 6,508 sites came from "ELSE" genes. annotated_hypoxia_Hela_temp_na |> dplyr::filter(is.na(gene_name))
# Need to check from the other transcript version. Not only the longest.
# However, if it is joined with several genes, take the longest tx_len.
annotated_hypoxia_Hela_temp_na2 <- 
  annotated_hypoxia_Hela_temp_na |> 
  dplyr::filter(is.na(gene_name)) |> 
  dplyr::select(c(seqnames,start,end,strand,gene,meth_case,meth_control,site)) |> dplyr::left_join(  
    human_Tx_data |> 
      dplyr::select(seqnames, start, end, strand, gene_name, gene_type, tx_name,tx_len,width) |> 
      dplyr::rename(tx_id=tx_name),
    by=dplyr::join_by(seqnames==seqnames, site>=start, site<=end)
  )|> 
  dplyr::rename(start = start.x, end = end.x, strand = strand.x, 
                start_gene = start.y, end_gene = end.y, strand_gene = strand.y) |> 
  group_by(seqnames,start,end) |> 
  dplyr::mutate(tx_len=ifelse(is.na(tx_len), 0, tx_len)) |> 
  dplyr::filter(tx_len == max(tx_len)) |> 
  dplyr::ungroup() 
# Still 6077 sites not joind! annotated_hypoxia_Hela_temp_na2 |> filter(is.na(start_gene))  
# detail 
# annotated_hypoxia_Hela_temp_na2 |> filter(is.na(start_gene)) |> mutate(flag = case_when(meth_case==0 ~ "control", meth_control==0 ~ "case", .default= "both")) |> group_by(flag) |> count()

dplyr::bind_rows(annotated_hypoxia_Hela |> dplyr::filter(!is.na(start_gene)), 
                 )

temp <- annotated_hypoxia_Hela_temp_na |> 
  dplyr::filter(!is.na(gene_name)) |> 
  dplyr::bind_rows(annotated_hypoxia_Hela_temp_na2) |> 
  dplyr::mutate(gene=ifelse(!is.na(start_gene) & gene!=gene_name, gene_name, gene)) |> 
  dplyr::select(c(seqnames, start, end, strand, gene, meth_case, meth_control, gene_type, tx_id, width, start_gene, end_gene, strand_gene))


# Final annotation
annotated_hypoxia_Hela <- dplyr::bind_rows(annotated_hypoxia_Hela |> dplyr::filter(!is.na(start_gene)), 
                 temp) |> 
  dplyr::arrange(seqnames,start,end)


# All methylation data if it's available has minimum of 10% 
min(annotated_hypoxia_Hela |> dplyr::filter(meth_case!=0) |> pull(meth_case))
min(annotated_hypoxia_Hela |> dplyr::filter(meth_control!=0) |> pull(meth_control))

# Buffer
temp_annotated_hypoxia_Hela <- annotated_hypoxia_Hela



annotated_hypoxia_Hela <- temp_annotated_hypoxia_Hela
# Since we want to know the promoters, we drop the unkonwn start_gene sites
annotated_hypoxia_Hela <- annotated_hypoxia_Hela |> filter(!is.na(start_gene))


STREAM <- 500
WIDTH <- 1001
annotated_hypoxia_Hela$data_width <- WIDTH


# Initialize the array of m6A prb
annotated_hypoxia_Hela$m6A_prob_control <- lapply(annotated_hypoxia_Hela$data_width, function(x) rep(0.0, x))
annotated_hypoxia_Hela$m6A_flag_control <- annotated_hypoxia_Hela$m6A_prob_control 
annotated_hypoxia_Hela$m6A_prob_case <- annotated_hypoxia_Hela$m6A_prob_control 
annotated_hypoxia_Hela$m6A_flag_case <- annotated_hypoxia_Hela$m6A_prob_control 

# Sort annotated_hypoxia_Hela by gene_id
annotated_hypoxia_Hela <- annotated_hypoxia_Hela |> dplyr::arrange(seqnames, start)

temp_buffer <- annotated_hypoxia_Hela


# Initialize a list to store the most recently updated m6A_prob vectors
#annotated_hypoxia_Hela <- temp_buffer 



# Parallel - Based on Sarah's Feedback
library(parallel)
library(doMC)
library(foreach)
n_cores <- parallel::detectCores()
doMC::registerDoMC(n_cores - 1)

#Check registered DoPar 
foreach::getDoParRegistered() == TRUE 
foreach::getDoParWorkers() == 6 # In my MBP M1 

annotation_df <- temp_buffer
start <- proc.time()[[3]]
m6A_data <- foreach(i=seq_len(nrow(annotated_hypoxia_Hela)), .combine='rbind') %dopar% {
  # Get methylation sites around ith site (for each loop)
  if (i %% 1000 == 0){
    print(paste0("Iteration : ",i))
  }
  idxs <- which(
    (
      (annotation_df$start >= (annotation_df$start[i]-STREAM)) 
    ) & 
      (annotation_df$start <= (annotation_df$start[i]+STREAM)&
         (annotation_df$seqnames == annotation_df$seqnames[i]))
  )
  
  if(length(idxs) > 0) {
    m6A_prob_control_vec <- annotation_df$m6A_prob_control[[i]]
    m6A_prob_case_vec <- annotation_df$m6A_prob_case[[i]]
    m6A_flag_control_vec <- annotation_df$m6A_prob_control[[i]]
    m6A_flag_case_vec <- annotation_df$m6A_prob_case[[i]]
    for(idx in idxs){
      relative_pos <- annotation_df$start[idx] - (annotation_df$start[i] - STREAM) + 1
      # Update the m6A_prob vector
      if (annotation_df$meth_control[idx] > 0) {
        m6A_prob_control_vec[relative_pos] <- annotation_df$meth_control[idx]
        m6A_flag_control_vec[relative_pos] <- 1
        
      }
      if ( annotation_df$meth_case[idx] > 0) {
        m6A_prob_case_vec[relative_pos] <- annotation_df$meth_case[idx]
        m6A_flag_case_vec[relative_pos] <- 1
      }
    }
  } 
  
  return(list(m6A_prob_control_vec, m6A_flag_control_vec,
              m6A_prob_case_vec, m6A_flag_case_vec))
}
elapsed <- proc.time()[[3]] - start
print(paste0("Elapsed time for generating array of m6A: ", elapsed))
#[1] "Elapsed time for generating array of m6A: 1007.591"
rownames(m6A_data) <- NULL
colnames(m6A_data) <- c("m6A_prob_control", "m6A_flag_control",
                        "m6A_prob_case", "m6A_flag_case")

#parallel::stopCluster()

rm(temp_buffer, annotation_df)

# Sanity Check 
# Location should be the same
which(m6A_data[1, 1:2]$m6A_flag_control>0) == which(m6A_data[1, 1:2]$m6A_prob_control>0)
which(m6A_data[2, 3:4]$m6A_flag_case>0) == which(m6A_data[2, 3:4]$m6A_prob_case>0)
# Flag should be 1
m6A_data[1, 1:2]$m6A_flag_control[which(m6A_data[1, 1:2]$m6A_flag_control>0)] == 1
print(m6A_data[1, 1:2]$m6A_prob_control[which(m6A_data[1, 1:2]$m6A_prob_control>0)])
m6A_data[2, 3:4]$m6A_flag_case[which(m6A_data[2, 3:4]$m6A_flag_case>0)] == 1
print(m6A_data[2, 3:4]$m6A_prob_case[which(m6A_data[2, 3:4]$m6A_prob_case>0)] )


annotated_hypoxia_Hela$m6A_prob_control <- m6A_data[,"m6A_prob_control"]
annotated_hypoxia_Hela$m6A_flag_control <- m6A_data[,"m6A_flag_control"]
annotated_hypoxia_Hela$m6A_prob_case <- m6A_data[,"m6A_prob_case"]
annotated_hypoxia_Hela$m6A_flag_case <- m6A_data[,"m6A_flag_case"]





## == Sanity Check Start ==## 
annotated_hypoxia_Hela$m6A_prob_case[[14]][501] == annotated_hypoxia_Hela$meth_case[14]
rel_pos_check1 <- (annotated_hypoxia_Hela$start[15] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_case[[14]][rel_pos_check1] == annotated_hypoxia_Hela$meth_case[15]
annotated_hypoxia_Hela$m6A_flag_case[[14]][rel_pos_check1] == 1
rel_pos_check2 <- (annotated_hypoxia_Hela$start[16] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_case[[14]][rel_pos_check2] == annotated_hypoxia_Hela$meth_case[16]
annotated_hypoxia_Hela$m6A_flag_case[[14]][rel_pos_check2] == 1


annotated_hypoxia_Hela$m6A_prob_control[[14]][501] == annotated_hypoxia_Hela$meth_control[14]
rel_pos_check1 <- (annotated_hypoxia_Hela$start[12] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_control[[14]][rel_pos_check1] == annotated_hypoxia_Hela$meth_control[12]
annotated_hypoxia_Hela$m6A_flag_control[[14]][rel_pos_check1] == 1
rel_pos_check2 <- (annotated_hypoxia_Hela$start[13] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_control[[14]][rel_pos_check2] == annotated_hypoxia_Hela$meth_control[13]
annotated_hypoxia_Hela$m6A_flag_control[[14]][rel_pos_check2] == 1
rel_pos_check1 <- (annotated_hypoxia_Hela$start[15] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_control[[14]][rel_pos_check1] == annotated_hypoxia_Hela$meth_control[15]
annotated_hypoxia_Hela$m6A_flag_control[[14]][rel_pos_check1] == 1
rel_pos_check2 <- (annotated_hypoxia_Hela$start[16] - (annotated_hypoxia_Hela$start[14]-STREAM) +1)
annotated_hypoxia_Hela$m6A_prob_control[[14]][rel_pos_check2] == annotated_hypoxia_Hela$meth_control[16]
annotated_hypoxia_Hela$m6A_flag_control[[14]][rel_pos_check2] == 1
# We will check those position whether A or not on the next code chunk
## == Sanity Check End ==##





# == Make the GenomicRanges for genes and promoters ==#
GR <- makeGRangesFromDataFrame(
  annotated_hypoxia_Hela |> 
    dplyr::rowwise() |>
    # UCSC Format for BSgenome
    dplyr::mutate( seqnames=ifelse(strsplit(as.character(seqnames), "chr")[[1]][2] == "M", 
                                   "MT",
                                   strsplit(as.character(seqnames), "chr")[[1]][2])) |> 
    dplyr::ungroup())
#Adding window 
GR <- GR + 500

# We already filtered the sites which don't belong to any TxDb data.
# Hence, it is safe to get the promoter
promoters_GR <- promoters(GR, upstream=1000, downstream=0)




# ===== Getting Sequence from BSgenome ===== #
# Credit to Sarah Rennie
bs_human <- BSgenome.Hsapiens.NCBI.GRCh38
#bs_mouse <- BSgenome.Mmusculus.UCSC.mm39
bs_info_human <- as.data.frame(seqinfo(BSgenome.Hsapiens.NCBI.GRCh38))
#bs_info_mouse <- as.data.frame(seqinfo(BSgenome.Mmusculus.UCSC.mm39))

bs_info_human <- tibble::rownames_to_column(bs_info_human, "seqnames")
#bs_info_mouse <- tibble::rownames_to_column(bs_info_mouse, "seqnames")

# Get sequence from GR using BSgenome
hypoxia_HeLa_seq <- BSgenome::getSeq(bs_human, GR)
hypoxia_Hela_promoters_seq <- BSgenome::getSeq(bs_human,promoters_GR)


# == Sanity check sequence ==
# Should be A
substring(as.character(hypoxia_HeLa_seq[[2001]]), 501, 501)
substring(as.character(hypoxia_HeLa_seq[[13]]), 501, 501)
# Checking the methylated position 
# Should be A
substring(as.character(hypoxia_HeLa_seq[[14]]), rel_pos_check1, rel_pos_check1)
substring(as.character(hypoxia_HeLa_seq[[14]]), rel_pos_check2, rel_pos_check2)




# ===== Creating train-test split ===== #

# For a single model 
case_metadata <- annotated_hypoxia_Hela |> dplyr::filter(meth_case!=0)
case_idx <- which(annotated_hypoxia_Hela |> dplyr::pull(meth_case) !=0)
case_seq <- hypoxia_HeLa_seq[case_idx]
case_prom <- hypoxia_Hela_promoters_seq[case_idx]
  
control_metadata <- annotated_hypoxia_Hela |> dplyr::filter(meth_control!=0)
control_idx <- which(annotated_hypoxia_Hela |> dplyr::pull(meth_control) !=0)
control_seq <- hypoxia_HeLa_seq[control_idx]
control_prom <- hypoxia_Hela_promoters_seq[control_idx]

make_split <- function(df, seq, prom_seq, data_folder){
  set.seed(123123)
  #allocate a group randomly to each gene_id, not position
  df$id <- c(1:dim(df)[1])
  gene_group <- sample(1:5,length(unique(df$gene)),replace=T)
  names(gene_group) <- unique(df$gene)
  df$gene_group <- gene_group[as.vector(df$gene)]
  
  # Sanity Check 
  # Grouping by sites check - expected uneven dist. since several genes might have higher number of sites than others
  print(df |> group_by(gene_group) |> count())
  # Grouping by distinct gene check - expected even number
  print(df |> 
          dplyr::select(gene, gene_group) |> 
          dplyr::distinct() |>
          dplyr::group_by(gene_group) |> count())
  
  
  
  #create the five folds stratified by gene_group
  #all of the groups consiste of roughly equal number of gene
  split_tibble <- df |> rsample::group_vfold_cv(group = gene_group)
  
  
  for(i in 1:nrow(split_tibble)){
    # Write out the training/testing labels
    data_split_TRAIN <- rsample::training(split_tibble$splits[[i]]) 
    data_split_TEST <- rsample::testing(split_tibble$splits[[i]]) 
    
    # Verbose 
    print(paste0("Train sites: ",dim(data_split_TRAIN)[1]))
    print(paste0("Train sites prop: ", sprintf("%.2f",(dim(data_split_TRAIN)[1])/(dim(data_split_TRAIN)[1] + dim(data_split_TEST)[1]))))
    print( data_split_TRAIN |> group_by(gene_group) |> count() |> unite("id",gene_group:n) |> pull(id))
    print(paste0("Test: ",dim(data_split_TEST)[1]))
    print(paste0("Test sites prop: ", sprintf("%.2f",(dim(data_split_TEST)[1])/(dim(data_split_TRAIN)[1] + dim(data_split_TEST)[1]))))
    print( data_split_TEST |> group_by(gene_group) |> count() |> unite("id",gene_group:n) |> pull(id))
    
    jsonlite::write_json(data_split_TRAIN , 
                         paste0(data_folder,
                                "train_meta_data_SPLIT_",i,".json"))
    jsonlite::write_json(data_split_TEST, 
                         paste0(data_folder, 
                                "test_meta_data_SPLIT_",i,".json"))
    
    jsonlite::write_json(data_split_TRAIN |> dplyr::select(c(m6A_prob_control)), 
                         paste0(data_folder,
                                "train_control_m6A_prob_data_SPLIT_",i,".json"))
    jsonlite::write_json(data_split_TEST |> dplyr::select(c(m6A_prob_control)), 
                         paste0(data_folder, 
                                "test_control_m6A_prob_data_SPLIT_",i,".json"))
    
    jsonlite::write_json(data_split_TRAIN |> dplyr::select(c(m6A_flag_control)), 
                         paste0(data_folder,
                                "train_control_m6A_flag_data_SPLIT_",i,".json"))
    jsonlite::write_json(data_split_TEST |> dplyr::select(c(m6A_flag_control)), 
                         paste0(data_folder, 
                                "test_control_m6A_flag_data_SPLIT_",i,".json"))
    
    jsonlite::write_json(data_split_TRAIN |> dplyr::select(c(m6A_prob_case)), 
                         paste0(data_folder,
                                "train_case_m6A_prob_data_SPLIT_",i,".json"))
    jsonlite::write_json(data_split_TEST |> dplyr::select(c(m6A_prob_case)), 
                         paste0(data_folder, 
                                "test_case_m6A_prob_data_SPLIT_",i,".json"))
    
    jsonlite::write_json(data_split_TRAIN |> dplyr::select(c(m6A_prob_case)), 
                         paste0(data_folder,
                                "train_case_m6A_flag_data_SPLIT_",i,".json"))
    jsonlite::write_json(data_split_TEST |> dplyr::select(c(m6A_prob_case)), 
                         paste0(data_folder, 
                                "test_case_m6A_flag_data_SPLIT_",i,".json"))
    
    seq_split_TRAIN <- seq[training(split_tibble$splits[[i]]) |> pull(id)]
    seq_split_TEST <- seq[testing(split_tibble$splits[[i]]) |> pull(id)]
    
    promoter_split_TRAIN <- prom_seq[training(split_tibble$splits[[i]]) |> pull(id)]
    promoter_split_TEST <- prom_seq[testing(split_tibble$splits[[i]]) |> pull(id)]
    
    ## Write an XStringSet object to a FASTA (or FASTQ) file:
    writeXStringSet(seq_split_TRAIN, paste0(data_folder,"motif_fasta_train_SPLIT_",i,".fasta"), append=FALSE,
                    compress=FALSE, compression_level=NA, format="fasta")
    
    writeXStringSet(seq_split_TEST, paste0(data_folder,"motif_fasta_test_SPLIT_",i,".fasta"), append=FALSE,
                    compress=FALSE, compression_level=NA, format="fasta")
    
    writeXStringSet(promoter_split_TRAIN, paste0(data_folder,"promoter_fasta_train_SPLIT_",i,".fasta"), append=FALSE,
                    compress=FALSE, compression_level=NA, format="fasta")
    
    writeXStringSet(promoter_split_TEST, paste0(data_folder,"promoter_fasta_test_SPLIT_",i,".fasta"), append=FALSE,
                    compress=FALSE, compression_level=NA, format="fasta")
    
  }
}




case_metadata <- annotated_hypoxia_Hela |> dplyr::filter(meth_case!=0)
case_idx <- which(annotated_hypoxia_Hela |> dplyr::pull(meth_case) !=0)
case_seq <- hypoxia_HeLa_seq[case_idx]
case_prom <- hypoxia_Hela_promoters_seq[case_idx]

make_split(case_metadata, case_seq, case_prom, paste0(DATA_FOLDER, "train_test_data_500_3/case/"))

control_metadata <- annotated_hypoxia_Hela |> dplyr::filter(meth_control!=0)
control_idx <- which(annotated_hypoxia_Hela |> dplyr::pull(meth_control) !=0)
control_seq <- hypoxia_HeLa_seq[control_idx]
control_prom <- hypoxia_Hela_promoters_seq[control_idx]
make_split(control_metadata, control_seq, control_prom, paste0(DATA_FOLDER, "train_test_data_500_3/control/"))


# for double outputs model 
make_split(annotated_hypoxia_Hela, hypoxia_HeLa_seq, hypoxia_Hela_promoters_seq, paste0(DATA_FOLDER, "double_outputs/"))














save(annotated_hypoxia_Hela,file="data_tibble_PD.Rdat")
save(split_tibble,file="data_tibble_splits_PD.Rdat")


analysis 


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