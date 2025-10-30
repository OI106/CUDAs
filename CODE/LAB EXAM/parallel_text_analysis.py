
# TASK 2 
 
import pandas as pd
from collections import Counter
from multiprocessing import Pool, current_process
import re
import time as Time

def process_chunk(chunk):

    start_time = Time.time()
    worker_id = current_process().name

    chunk_col = chunk["review"].astype(str) + " " + chunk["sentiment"].astype(str)
    
    words = re.findall(r'\w+', ' '.join(chunk_col).lower())
    
    word_count = len(words)
    
    end_time = Time.time()
    process_time = end_time - start_time
    
    count = Counter(words)
    efficiency = word_count / process_time
    
    return  {
        "processing_time": process_time,
        "efficiency": efficiency,
        "counter": count
    }

if __name__ == '__main__':
    
    df = pd.read_csv('IMDB Dataset.csv')
    num_prc = 4
    
    chunk_size = len(df) // num_prc + 1
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with Pool(processes=num_prc) as p:
        results = p.map(process_chunk, chunks)
    
    print("Combining results from all processes...")
    total_counter = Counter()
    
    for counter in results:
        total_counter.update(counter["counter"])
        
    df = pd.DataFrame([{
    
        "Processing Time" : counter["processing_time"],
        "Word Count": counter["counter"],
        "Efficiency (words/sec)": round(counter["efficiency"], 2)
    
    }for counter in results])
    
    print(df)
    
    