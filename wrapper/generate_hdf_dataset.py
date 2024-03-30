import h5py 
import numpy as np 
import pandas as pd 
from Bio import SeqIO
from gene2vec_embedding import gene2vec
from gensim.models import Word2Vec
import time 
PROJECT_FOLDER = "/binf-isilon/renniegrp/vpx267/ucph_thesis"
data_folder = f"{PROJECT_FOLDER}/data/dual_outputs"
output_folder = f"{PROJECT_FOLDER}/data/dual_outputs/hdf5"


#for mode in ["train","test"]:
for mode in ["train"]:
    for seq_type in ["motif"]:
        for i in range(1, 6):
            # Load embedding

            start_time = time.time() 
            path_to_embedding = f"{PROJECT_FOLDER}/data/embeddings/gene2vec/dual_outputs/split_{i}.model"
            embedding = Word2Vec.load(path_to_embedding)
            print(f"Mode : {mode}, Seq Type: {seq_type}, Split: {i}")
            file_name = f"{data_folder}/{seq_type}_fasta_{mode}_SPLIT_{i}.fasta"
            seq_records = list(SeqIO.parse(file_name, format="fasta"))
            data = []
            for seq_record in seq_records:
                try:
                    data.append(gene2vec(str(seq_record.seq), embedding))
                except ValueError as e:
                    print(e)
            data = np.array(data, dtype=np.float32)
            print(f"Data shape: {data.shape}")
            with h5py.File(f"{output_folder}/gene2vec.hdf5", "a") as file:
                file.create_dataset(f"{mode}/{seq_type}_SPLIT_{i}", data=data)
            
            end_time = time.time() 
            print(end_time-start_time)