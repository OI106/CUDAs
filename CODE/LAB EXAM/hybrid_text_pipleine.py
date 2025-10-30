from mpi4py import MPI
import pandas as pd
from collections import Counter
import numpy as np
import re
import time as Time
from multiprocessing import Pool

def process_subchunk(sub_df):
    
    df_col = sub_df["review"].astype(str) + " " + sub_df["sentiment"].astype(str)
    words = re.findall(r'\w+', ' '.join(df_col).lower())
    return len(words)

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        
        df = pd.read_csv('IMDB Dataset.csv')
        df_pos = df[df["sentiment"].str.lower() == "positive"]
        df_neg = df[df["sentiment"].str.lower() == "negative"]
        
        comm.send(df_neg, dest=1, tag=0)
        local_df = df_pos
        type = "positive"
                
    elif rank == 1:
        local_df = comm.recv(source = 0 , tag = 0)
        type = "negative"
    
    start_time = Time.time()

    num_threads = 4
    subchunk = np.array_split(local_df, num_threads)
    
    with Pool(processes=num_threads) as p:
        word_counts = p.map(process_subchunk, subchunk)

    local_word_count = sum(word_counts)
    
    end_time = Time.time()
    
    local_time = end_time - start_time

    print(f"Process {rank} completed processing {local_word_count} words of {type} sentiments in {local_time:.3f} s", flush=True)

    total_word_count = comm.reduce(local_word_count, op=MPI.SUM, root=0)

    comm.barrier()
    
    if rank == 0:
        print(f"\nTotal word count across all processes: {total_word_count}")
        