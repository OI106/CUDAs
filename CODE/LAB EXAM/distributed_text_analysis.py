from mpi4py import MPI
import pandas as pd
from collections import Counter
import numpy as np
import re
import time as Time

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Rank 0 reads and splits the data
    if rank == 0:
        df = pd.read_csv('IMDB Dataset.csv')
        df_splits = np.array_split(df, size)
    else:
        df_splits = None  # other ranks have nothing yet

    # Step 2: Distribute data chunks automatically
    local_df = comm.scatter(df_splits, root=0)

    # Step 3: Process each chunk locally
    start_time = Time.time()

    df_col = local_df["review"].astype(str) + " " + local_df["sentiment"].astype(str)
    words = re.findall(r'\w+', ' '.join(df_col).lower())
    word_count = len(words)

    end_time = Time.time()
    local_time = end_time - start_time

    print(f"Process {rank} completed processing {word_count} words in {local_time:.3f} s", flush=True)

    # Step 4: Reduce total word count (sum)
    total_word_count = comm.reduce(word_count, op=MPI.SUM, root=0)

    # Step 5: Synchronize and measure global time
    comm.barrier()  # wait until everyone finishes
    if rank == 0:
        print(f"\nTotal word count across all processes: {total_word_count}")

