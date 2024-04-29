from Bio import SeqIO
import numpy as np
import logging
import sys
from time import time
import multiprocessing
from gensim.models import Word2Vec
from argparse import ArgumentParser
import glob
import os
import random
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor


def get_kmers(sequence: str, kmer: int = 3) -> list[str]:
    """
    Get k-mers from a sequence.

    param: sequence: sequence of nucleotides.
    param: kmer: length of k-mer.
    return: list of k-mers.
    """
    return [sequence[i:i+kmer] for i in range(len(sequence) - 2)]


def generate_kmers(path_to_fasta: str) -> list:
    """
    Generate list of k-mers from a FASTA file.

    param: path_to_fasta: path to FASTA file containing sequences.
    return: training_data: list of k-mers.
    """
    sequences = []
    for seq_record in SeqIO.parse(path_to_fasta, format="fasta"):
        sequences.append(str(seq_record.seq))
    training_data = []
    for sequence in sequences:
        kmers = get_kmers(sequence)
        training_data.append(kmers)
    return training_data


def gene2vec(seq: str, embedding: Word2Vec) -> np.ndarray:
    """
    Gene2Vec encode a sequence
    """
    result = get_kmers(seq, 3)
    transformed_seq = []
    for kmer in result:
        if kmer in embedding.wv:
            transformed_seq.append(embedding.wv[kmer])
        else:
            # Unknown kmer
            transformed_seq.append([0.0] * embedding.wv.vector_size)

    result = np.array(transformed_seq)
    return result.T


def gene2vec2(seq, embedding: Word2Vec) -> np.ndarray:
    """
    Gene2Vec encode a sequence
    """
    result = get_kmers(str(seq.seq), 3)
    transformed_seq = []
    for kmer in result:
        if kmer in embedding.wv:
            transformed_seq.append(embedding.wv[kmer])
        else:
            # Unknown kmer
            transformed_seq.append([0.0] * embedding.wv.vector_size)

    result = np.array(transformed_seq)
    return result.T


def process_sequences(sequences: list[str], embedding: Word2Vec):
    num_processes = (2 * os.cpu_count()//3)
    # with Pool(num_processes) as pool:
    #     results = pool.starmap(gene2vec, [(str(seq.seq), embedding) for seq in sequences])
    from functools import partial
    start_time = time.time()
    pool = multiprocessing.Pool(4)
    temp = partial(gene2vec, "input_dir", "output_dir")
    results = pool.map(func=temp, iterable=sequences)
    pool.close()
    pool.join()
    end_time = time.time()
    print(end_time-start_time)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(
            gene2vec, [(str(seq.seq), embedding) for seq in sequences]))
    return results


if __name__ == "__main__":
    """
    EXAMPLE 
    -----

    python wrapper/gene2vec_embedding.py --data_folder data/dual_outputs --fasta_file_pattern "motif_fasta_train_SPLIT_*.fasta" --output_folder data/embeddings/gene2vec/double_outputs
    """
    # Set logging template
    logging.basicConfig(format='%(asctime)s::%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    parser = ArgumentParser(
        description="Create Gene2Vec embeddings from FASTA file(s)"
    )
    parser.add_argument(
        "--data_folder", help="Train test Data Folder", required=True)
    parser.add_argument("--fasta_file_pattern",
                        help="Fasta file pattern", required=True)
    parser.add_argument("--output_folder", help="Output Folder", required=True)

    args = parser.parse_args()
    full_pattern = os.path.join(args.data_folder, args.fasta_file_pattern)
    matching_files = sorted(glob.glob(full_pattern))

    SEED = 1234
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

    for idx, file_path in enumerate(matching_files, start=1):
        logging.info(f"\n\n=== Processing file: {idx} {file_path} ===")
        sequences_corpus_train = generate_kmers(file_path)
        cores = multiprocessing.cpu_count()
        t = time()
        w2v_model = Word2Vec(sentences=sequences_corpus_train,
                             window=5,
                             seed=SEED,
                             vector_size=300,
                             workers=(cores//3)*2,  # uses 2/3 cores of CPU
                             epochs=15
                             )
        w2v_model.save(os.path.join(args.output_folder, f"split_{idx}.model"))
        logging.info('Time to build vocab: {} mins'.format(
            round((time() - t) / 60, 2)))
        logging.info(f"=== Finished processing file: {idx} {file_path} ===")
